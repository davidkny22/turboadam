"""Memory profiling instrumentation.

Tracks per-component memory allocation at each training step:
- Weights, gradients, master weights
- Optimizer state (m + v compressed)
- Peak vs. steady-state memory

Produces the primary figure for the paper: memory profile over training.
"""
