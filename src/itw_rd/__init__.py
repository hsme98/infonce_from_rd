"""Finite-state InfoNCE and fixed-output rate-distortion experiments."""

from .core import fixed_output_rd, free_output_rd, sinkhorn_log

__version__ = "1.0.0"
__all__ = ["fixed_output_rd", "free_output_rd", "sinkhorn_log"]
