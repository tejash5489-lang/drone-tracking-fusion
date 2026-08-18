"""
Differentiable (PyTorch) Covariance Intersection — Phase 3, Approach 2.

Same two equations as fusion.covariance_intersection (the numpy version you
implemented for Phase 1), reimplemented with torch ops:

    Pcc^-1 = omega * Paa^-1 + (1 - omega) * Pbb^-1
    xcc    = Pcc @ (omega * Paa^-1 @ xa + (1 - omega) * Pbb^-1 @ xb)

Why a second implementation of the same formula: Approach 2's LSTM outputs
omega(t) and is trained end-to-end (blueprint 2.3 — "trained end-to-end to
minimize OSPA and SIAP tracking-quality metrics"), meaning gradients have to
flow backward from a loss on the *fused* track, through the CI equations,
through omega, and into the LSTM's weights. The numpy version can't do
that — numpy has no autograd. This module is that gradient path; the numpy
version stays the one everything else in the project (Phase 1 baseline,
Approach 1, and Approach 2's own final evaluation — see
fusion/approach2_train.py) uses for actual fusion, since it's the one that
interoperates with Stone Soup's (numpy-based) Track/GaussianState objects.

Vectorized over a whole sequence at once: xa/xb have shape (T, n, 1),
Pa/Pb have shape (T, n, n), omega has shape (T,) — one call fuses an entire
trajectory, rather than looping per time step in Python.
"""
import torch


def covariance_intersection_torch(xa, Pa, xb, Pb, omega):
    """Batched, differentiable CI.

    Parameters
    ----------
    xa, xb : torch.Tensor, shape (..., n, 1)
    Pa, Pb : torch.Tensor, shape (..., n, n)
    omega : torch.Tensor, shape (...,) — one value per leading batch/time
        index, broadcasting against the trailing (n, n) / (n, 1) dims.

    Returns
    -------
    xc, Pc : torch.Tensor, shapes matching xa and Pa.
    """
    Pa_inv = torch.linalg.inv(Pa)
    Pb_inv = torch.linalg.inv(Pb)

    omega = omega.reshape(*omega.shape, 1, 1)  # -> (..., 1, 1), broadcasts over (n, n)
    Pc_inv = omega * Pa_inv + (1 - omega) * Pb_inv
    Pc = torch.linalg.inv(Pc_inv)
    xc = Pc @ (omega * Pa_inv @ xa + (1 - omega) * Pb_inv @ xb)
    return xc, Pc
