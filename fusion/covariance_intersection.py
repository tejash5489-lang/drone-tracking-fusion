"""
Covariance Intersection (CI) fusion.

CI combines two Gaussian estimates when their cross-correlation is unknown:

    Pc^-1 = omega * Pa^-1 + (1 - omega) * Pb^-1
    xc    = Pc @ (omega * Pa^-1 @ xa + (1 - omega) * Pb^-1 @ xb)

All four omega-selection approaches in this project feed their chosen omega
into this same fusion engine.
"""
import numpy as np


def covariance_intersection(xa, Pa, xb, Pb, omega):
    """Fuse two Gaussian estimates via Covariance Intersection.

    Parameters
    ----------
    xa, xb : np.ndarray, shape (n, 1)
        State estimates from sensor A and sensor B.
    Pa, Pb : np.ndarray, shape (n, n)
        State covariance matrices for each estimate.
    omega : float
        Mixing parameter in [0, 1]. omega=1 recovers sensor A, omega=0
        recovers sensor B, and omega=0.5 is the fixed baseline.

    Returns
    -------
    xc : np.ndarray, shape (n, 1)
        Fused state estimate.
    Pc : np.ndarray, shape (n, n)
        Fused covariance.
    """
    xa = np.asarray(xa, dtype=float)
    xb = np.asarray(xb, dtype=float)
    Pa = np.asarray(Pa, dtype=float)
    Pb = np.asarray(Pb, dtype=float)
    omega = float(omega)

    if not 0.0 <= omega <= 1.0:
        raise ValueError(f"omega must be in [0, 1], got {omega!r}")

    Pa_inv = np.linalg.inv(Pa)
    Pb_inv = np.linalg.inv(Pb)

    Pc_inv = omega * Pa_inv + (1.0 - omega) * Pb_inv
    Pc = np.linalg.inv(Pc_inv)
    xc = Pc @ (omega * Pa_inv @ xa + (1.0 - omega) * Pb_inv @ xb)

    return xc, Pc
