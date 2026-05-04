"""Integration tests for TurboAdam optimizer — compress-every-step architecture.

Covers:
- API compatibility with torch.optim.Adam
- Convergence on a simple quadratic loss
- State structure (compressed v, CoState m)
- Closeness to standard Adam within tolerance
- Ablation flags (compress_m, compress_v)
"""

import copy

import pytest
import torch
import torch.nn as nn

from turboadam import TurboAdam
from turboadam.costate import CoStateManager
from turboadam.oneq import decompress_v


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_params(seed=42):
    """Return a list of parameters initialized from a fixed seed."""
    torch.manual_seed(seed)
    return [
        torch.nn.Parameter(torch.randn(10, 10)),
        torch.nn.Parameter(torch.randn(20)),
    ]


def _step_with_grad(opt, params):
    """Run one step with fresh random gradients."""
    for p in params:
        p.grad = torch.randn_like(p)
    opt.step()


# ---------------------------------------------------------------------------
# 1. API compatibility
# ---------------------------------------------------------------------------

class TestAPICompatibility:
    def test_accepts_adam_args(self):
        params = _make_params()
        opt = TurboAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
        assert opt.defaults["lr"] == 1e-3
        assert opt.defaults["betas"] == (0.9, 0.999)
        assert opt.defaults["eps"] == 1e-8
        assert opt.defaults["weight_decay"] == 1e-2

    def test_accepts_turboadam_args(self):
        params = _make_params()
        opt = TurboAdam(params, block_size=64, v_bits=6)
        assert opt.defaults["block_size"] == 64
        assert opt.defaults["v_bits"] == 6

    def test_default_args(self):
        params = _make_params()
        opt = TurboAdam(params)
        assert opt.defaults["lr"] == 1e-3
        assert opt.defaults["betas"] == (0.9, 0.999)
        assert opt.defaults["eps"] == 1e-8
        assert opt.defaults["weight_decay"] == 0.0
        assert opt.defaults["block_size"] == 128
        assert opt.defaults["v_bits"] == 4
        assert opt.defaults["compress_m"] is True
        assert opt.defaults["compress_v"] is True

    def test_step_callable(self):
        params = _make_params()
        opt = TurboAdam(params)
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()


# ---------------------------------------------------------------------------
# 2. Convergence on quadratic f(x) = sum(x^2)
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_converges_quadratic_200_steps(self):
        """TurboAdam should drive sum(x^2) close to 0 within 200 steps."""
        torch.manual_seed(0)
        x = nn.Parameter(torch.randn(50))
        opt = TurboAdam([x], lr=1e-2)

        initial_loss = (x ** 2).sum().item()
        for _ in range(200):
            opt.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            opt.step()

        final_loss = (x ** 2).sum().item()
        assert final_loss < 0.10 * initial_loss, (
            f"Expected final_loss < 10% of initial, "
            f"got initial={initial_loss:.4f}, final={final_loss:.6f}"
        )

    def test_converges_quadratic_no_compression(self):
        """With both compressions disabled, should match Adam convergence."""
        torch.manual_seed(0)
        x = nn.Parameter(torch.randn(50))
        opt = TurboAdam([x], lr=1e-2, compress_m=False, compress_v=False)

        initial_loss = (x ** 2).sum().item()
        for _ in range(200):
            opt.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            opt.step()

        final_loss = (x ** 2).sum().item()
        assert final_loss < 0.10 * initial_loss


# ---------------------------------------------------------------------------
# 3. State structure — compressed v and CoState m
# ---------------------------------------------------------------------------

class TestStateStructure:
    def test_compressed_v_present_after_first_step(self):
        """After first step, state should contain compressed_v dict."""
        torch.manual_seed(0)
        params = _make_params()
        opt = TurboAdam(params)
        _step_with_grad(opt, params)

        for p in params:
            state = opt.state[p]
            assert "compressed_v" in state, "compressed_v missing from state"
            assert isinstance(state["compressed_v"], dict)

    def test_compressed_v_roundtrip_shape(self):
        """Decompressed v should match parameter shape."""
        torch.manual_seed(0)
        params = _make_params()
        opt = TurboAdam(params)
        _step_with_grad(opt, params)

        for p in params:
            v_recon = decompress_v(opt.state[p]["compressed_v"])
            assert v_recon.shape == p.shape

    def test_compressed_v_updates_each_step(self):
        """compressed_v should change between steps (v is updated every step)."""
        torch.manual_seed(0)
        params = _make_params()
        opt = TurboAdam(params)

        _step_with_grad(opt, params)
        v1 = decompress_v(opt.state[params[0]]["compressed_v"]).clone()

        _step_with_grad(opt, params)
        v2 = decompress_v(opt.state[params[0]]["compressed_v"])

        assert not torch.allclose(v1, v2), "compressed_v should change between steps"

    def test_no_exp_avg_sq_when_compress_v(self):
        """With compress_v=True, state should NOT contain exp_avg_sq."""
        torch.manual_seed(0)
        params = _make_params()
        opt = TurboAdam(params, compress_v=True)
        _step_with_grad(opt, params)

        for p in params:
            assert "exp_avg_sq" not in opt.state[p]

    def test_exp_avg_sq_when_no_compress_v(self):
        """With compress_v=False, state should have fp32 exp_avg_sq."""
        torch.manual_seed(0)
        params = _make_params()
        opt = TurboAdam(params, compress_v=False)
        _step_with_grad(opt, params)

        for p in params:
            state = opt.state[p]
            assert "exp_avg_sq" in state
            assert state["exp_avg_sq"].dtype == torch.float32
            assert "compressed_v" not in state

    def test_m_mgr_present_for_large_params(self):
        """m compression manager should be in state for large params when compress_m=True."""
        torch.manual_seed(0)
        params = [nn.Parameter(torch.randn(64, 64))]  # 4096 elements
        opt = TurboAdam(params, compress_m=True)
        _step_with_grad(opt, params)

        assert "m_mgr" in opt.state[params[0]]

    def test_small_params_skip_costate(self):
        """Small params (< 4096 elements) should use fp32 m even with compress_m=True."""
        torch.manual_seed(0)
        params = [nn.Parameter(torch.randn(20))]
        opt = TurboAdam(params, compress_m=True)
        _step_with_grad(opt, params)

        assert "exp_avg" in opt.state[params[0]]
        assert "costate_mgr" not in opt.state[params[0]]

    def test_exp_avg_when_no_compress_m(self):
        """With compress_m=False, state should have fp32 exp_avg."""
        torch.manual_seed(0)
        params = _make_params()
        opt = TurboAdam(params, compress_m=False)
        _step_with_grad(opt, params)

        for p in params:
            state = opt.state[p]
            assert "exp_avg" in state
            assert state["exp_avg"].dtype == torch.float32
            assert "costate_mgr" not in state

    def test_step_counter_increments(self):
        """State step counter should increment correctly."""
        torch.manual_seed(0)
        params = _make_params()
        opt = TurboAdam(params)
        for step_num in range(1, 6):
            _step_with_grad(opt, params)
            for p in params:
                assert opt.state[p]["step"] == step_num


# ---------------------------------------------------------------------------
# 4. Closeness to standard Adam
# ---------------------------------------------------------------------------

class TestAdamComparison:
    def _run_optimizer(self, opt_class, params, grads_sequence, opt_kwargs=None):
        """Run optimizer for len(grads_sequence) steps, return final param values."""
        if opt_kwargs is None:
            opt_kwargs = {}
        opt = opt_class(params, lr=1e-3, **opt_kwargs)
        for grads in grads_sequence:
            for p, g in zip(params, grads):
                p.grad = g.clone()
            opt.step()
        return [p.data.clone() for p in params]

    def test_no_compression_matches_adam(self):
        """With both compressions disabled, TurboAdam should match Adam exactly."""
        torch.manual_seed(7)
        n_steps = 10

        grad_sequence = []
        for _ in range(n_steps):
            grads = [torch.randn(10, 10), torch.randn(20)]
            grad_sequence.append(grads)

        torch.manual_seed(42)
        adam_params = [nn.Parameter(torch.ones(10, 10)), nn.Parameter(torch.ones(20))]
        adam_results = self._run_optimizer(
            torch.optim.Adam, adam_params, grad_sequence, opt_kwargs={"betas": (0.9, 0.999)}
        )

        torch.manual_seed(42)
        turbo_params = [nn.Parameter(torch.ones(10, 10)), nn.Parameter(torch.ones(20))]
        turbo_results = self._run_optimizer(
            TurboAdam, turbo_params, grad_sequence,
            opt_kwargs={"betas": (0.9, 0.999), "compress_m": False, "compress_v": False},
        )

        for i, (a_val, t_val) in enumerate(zip(adam_results, turbo_results)):
            assert torch.allclose(a_val, t_val, atol=1e-6), (
                f"Param {i}: TurboAdam(no compression) should exactly match Adam"
            )

    def test_close_to_adam_10_steps(self):
        """TurboAdam with compression should be within ~15% of Adam after 10 steps."""
        torch.manual_seed(7)
        n_steps = 10

        grad_sequence = []
        for _ in range(n_steps):
            grads = [torch.randn(10, 10), torch.randn(20)]
            grad_sequence.append(grads)

        torch.manual_seed(42)
        adam_params = [nn.Parameter(torch.ones(10, 10)), nn.Parameter(torch.ones(20))]
        adam_results = self._run_optimizer(
            torch.optim.Adam, adam_params, grad_sequence, opt_kwargs={"betas": (0.9, 0.999)}
        )

        torch.manual_seed(42)
        turbo_params = [nn.Parameter(torch.ones(10, 10)), nn.Parameter(torch.ones(20))]
        turbo_results = self._run_optimizer(
            TurboAdam, turbo_params, grad_sequence, opt_kwargs={"betas": (0.9, 0.999)}
        )

        for i, (a_val, t_val) in enumerate(zip(adam_results, turbo_results)):
            a_norm = a_val.norm().item()
            diff_norm = (a_val - t_val).norm().item()
            rel_error = diff_norm / (a_norm + 1e-8)
            assert rel_error < 0.15, (
                f"Param {i}: relative error {rel_error:.4f} exceeds 15% tolerance"
            )

    def test_param_values_change_after_step(self):
        """Parameters should actually be updated after optimizer step."""
        torch.manual_seed(0)
        x = nn.Parameter(torch.ones(10))
        initial = x.data.clone()
        opt = TurboAdam([x], lr=1e-2)
        x.grad = torch.ones(10)
        opt.step()
        assert not torch.allclose(x.data, initial), "Parameters did not change after step"


# ---------------------------------------------------------------------------
# 5. Weight decay
# ---------------------------------------------------------------------------

class TestWeightDecay:
    def test_weight_decay_reduces_params(self):
        """With weight_decay, parameters should be pulled toward zero."""
        torch.manual_seed(0)
        x = nn.Parameter(torch.ones(20) * 10.0)
        opt = TurboAdam([x], lr=1e-3, weight_decay=0.1)
        for _ in range(10):
            x.grad = torch.zeros(20)
            opt.step()
        assert x.data.abs().mean().item() < 10.0, "Weight decay did not reduce params"


# ---------------------------------------------------------------------------
# 6. Multiple param groups
# ---------------------------------------------------------------------------

class TestParamGroups:
    def test_multiple_param_groups(self):
        """Optimizer should handle multiple param groups correctly."""
        torch.manual_seed(0)
        p1 = nn.Parameter(torch.randn(5, 5))
        p2 = nn.Parameter(torch.randn(10))
        opt = TurboAdam(
            [
                {"params": [p1], "lr": 1e-2},
                {"params": [p2], "lr": 1e-4},
            ]
        )
        p1.grad = torch.randn_like(p1)
        p2.grad = torch.randn_like(p2)
        opt.step()
        assert p1 in opt.state
        assert p2 in opt.state


# ---------------------------------------------------------------------------
# 7. V compression quality
# ---------------------------------------------------------------------------

class TestVCompression:
    def test_compressed_v_nonnegative(self):
        """Decompressed v should be non-negative (EMA of squared gradients)."""
        torch.manual_seed(0)
        params = _make_params()
        opt = TurboAdam(params)
        for _ in range(10):
            _step_with_grad(opt, params)

        for p in params:
            v = decompress_v(opt.state[p]["compressed_v"])
            assert (v >= 0).all(), "Decompressed v contains negative values"

    def test_v_bits_parameter(self):
        """Different v_bits values should work."""
        for bits in [4, 6, 8, 10]:
            torch.manual_seed(0)
            params = _make_params()
            opt = TurboAdam(params, v_bits=bits)
            _step_with_grad(opt, params)
            v = decompress_v(opt.state[params[0]]["compressed_v"])
            assert v.shape == params[0].shape

    def test_convergence_at_various_bits(self):
        """Should converge on quadratic at 4, 6, and 8 bits (within 500 steps)."""
        for bits in [4, 6, 8]:
            torch.manual_seed(0)
            x = nn.Parameter(torch.randn(50))
            opt = TurboAdam([x], lr=1e-2, v_bits=bits)

            initial_loss = (x ** 2).sum().item()
            for _ in range(500):
                opt.zero_grad()
                loss = (x ** 2).sum()
                loss.backward()
                opt.step()

            final_loss = (x ** 2).sum().item()
            assert final_loss < 0.10 * initial_loss, (
                f"Failed to converge at {bits}-bit: "
                f"initial={initial_loss:.4f}, final={final_loss:.6f}"
            )


# ---------------------------------------------------------------------------
# 8. Ablation flags
# ---------------------------------------------------------------------------

class TestAblationFlags:
    def test_compress_m_only(self):
        """compress_m=True, compress_v=False should work (CoState only)."""
        torch.manual_seed(0)
        x = nn.Parameter(torch.randn(50))
        opt = TurboAdam([x], lr=1e-2, compress_m=True, compress_v=False)

        initial_loss = (x ** 2).sum().item()
        for _ in range(200):
            opt.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            opt.step()

        final_loss = (x ** 2).sum().item()
        assert final_loss < 0.10 * initial_loss

    def test_compress_v_only(self):
        """compress_m=False, compress_v=True should work (v compression only)."""
        torch.manual_seed(0)
        x = nn.Parameter(torch.randn(50))
        opt = TurboAdam([x], lr=1e-2, compress_m=False, compress_v=True)

        initial_loss = (x ** 2).sum().item()
        for _ in range(200):
            opt.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            opt.step()

        final_loss = (x ** 2).sum().item()
        assert final_loss < 0.10 * initial_loss

    def test_no_compression(self):
        """Both disabled should behave like standard Adam."""
        torch.manual_seed(0)
        x = nn.Parameter(torch.randn(50))
        opt = TurboAdam([x], lr=1e-2, compress_m=False, compress_v=False)

        initial_loss = (x ** 2).sum().item()
        for _ in range(200):
            opt.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            opt.step()

        final_loss = (x ** 2).sum().item()
        assert final_loss < 0.10 * initial_loss


# ---------------------------------------------------------------------------
# 9. Closure support
# ---------------------------------------------------------------------------

class TestClosure:
    def test_closure_returns_loss(self):
        """step(closure) should return the loss value."""
        torch.manual_seed(0)
        x = nn.Parameter(torch.randn(10))
        opt = TurboAdam([x], lr=1e-2)

        def closure():
            opt.zero_grad()
            loss = (x ** 2).sum()
            loss.backward()
            return loss

        loss = opt.step(closure)
        assert loss is not None
        assert loss.item() > 0
