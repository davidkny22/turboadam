"""Integration tests for TurboAdam optimizer.

Covers:
- Drop-in replacement for torch.optim.Adam (API compatibility)
- Phase A / Phase B transition behavior
- Combined 1Q + CoState update loop
- State dict save/load roundtrip
- Parameter group handling
"""
