"""
Covariance Intersection (CI) fusion — the mechanism every one of the paper's
four approaches feeds an omega into. Blueprint 2.2:

    "Covariance Intersection (CI): a fusion technique for combining two
    estimates whose cross-correlation is unknown; combines the two estimates
    in inverse-covariance (information) space, weighted by omega:

        Pcc^-1 = omega * Paa^-1 + (1 - omega) * Pbb^-1

    This is the fusion 'engine' that all four approaches feed omega into."

That equation gives the fused covariance. The fused *mean* (paper's Eq. 3-4,
also standard CI) is:

    xcc = Pcc @ (omega * Paa^-1 @ xa + (1 - omega) * Pbb^-1 @ xb)

This module is deliberately left for you to implement — it's the one
equation the entire rest of the project (Phases 2-6) is built on top of:
every approach (LSTM-forecast, dynamic-fusing LSTM, IMM-LSTM, CMA-ES) is just
a different way of choosing omega(t) before calling this same function.
Getting comfortable with what CI actually does mathematically here will make
those four approaches much more legible later.
"""
import numpy as np


def covariance_intersection(xa, Pa, xb, Pb, omega):
    """Fuse two Gaussian estimates (xa, Pa) and (xb, Pb) via Covariance
    Intersection with a fixed mixing parameter ``omega``.

    Parameters
    ----------
    xa, xb : np.ndarray, shape (n, 1)
        State estimate (mean) from sensor A and sensor B, same state space.
    Pa, Pb : np.ndarray, shape (n, n)
        Corresponding state covariances.
    omega : float
        Mixing parameter in [0, 1]. omega=1 trusts sensor A completely,
        omega=0 trusts sensor B completely, omega=0.5 is the paper's fixed
        baseline.

    Returns
    -------
    xc : np.ndarray, shape (n, 1)
        Fused mean.
    Pc : np.ndarray, shape (n, n)
        Fused covariance.

    Notes
    -----
    CI is deliberately conservative: unlike a naive Kalman-style fusion, it
    makes no assumption about the cross-correlation between the two
    estimates (which is unknown here, since both sensors independently
    filter the same target). That's *why* omega is needed at all — it's the
    knob that controls how much each source is trusted, in the absence of
    a principled way to compute the "correct" fusion weight from the
    cross-covariance directly.

    TODO(you): implement this from the two equations in the module
    docstring (Pcc^-1 = ... and xcc = ...). A few things worth thinking
    about while you write it:
      - You'll need the *inverse* of each covariance matrix (information
        form) — `np.linalg.inv` is fine here; these are small (n x n),
        well-conditioned matrices in this project.
      - Compute Pc first, then use it to get xc — the mean equation needs
        Pc, not Pa/Pb directly.
      - Sanity check once it's done: with omega=1 you should recover
        exactly (xa, Pa); with omega=0, exactly (xb, Pb). That's a good
        unit test to write for yourself.
    """
    raise NotImplementedError("Implement Covariance Intersection — see the docstring above.")
