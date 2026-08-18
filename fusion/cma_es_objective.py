"""
CMA-ES omega objective — Phase 5, Approach 4 (blueprint 2.3):

    "CMA-ES searches for the omega that minimizes log(det(Pcc)) (the fused
    covariance's log-determinant, i.e., the tightest/most confident fused
    estimate), with grid-searched hyperparameters (x0 = 0.5, sigma0 = 0.5,
    max iterations = 100)."

Uses the real (numpy) fusion.covariance_intersection — unlike Approaches
2/3, this doesn't need a gradient through omega (CMA-ES is derivative-free
by design), so there's no differentiable-twin problem here the way there
was for Approach 2/3's training.

pycma quirk worth knowing about: omega is a single scalar, but pycma's own
CMAEvolutionStrategy explicitly warns that 1-D optimization "is not
supported and may bail or work poorly" (see
github.com/CMA-ES/pycma/issues/86 and /302) — CMA-ES's whole mechanism is
adapting a *covariance matrix* of a multivariate search distribution, which
degenerates in 1-D and hits real bugs in this library. The standard
workaround (used here) is to pad the search vector to 2 dimensions with an
unused dummy coordinate; only the first coordinate (omega) is read back out.
"""
import numpy as np
import cma

from fusion.covariance_intersection import covariance_intersection


def log_det_fused_covariance(xa, Pa, xb, Pb, omega):
    """log(det(Pcc)) — paper's Eq. 14. Smaller (more negative) means a
    tighter, more confident fused covariance."""
    _, Pc = covariance_intersection(xa, Pa, xb, Pb, omega)
    _, logdet = np.linalg.slogdet(Pc)
    return logdet


def optimize_omega_cma_es(xa, Pa, xb, Pb, x0=0.5, sigma0=0.5, maxiter=100, seed=None):
    """Find the omega in [0, 1] minimizing log(det(Pcc)) via CMA-ES, using
    the paper's specified starting point/step-size/iteration budget.

    Returns
    -------
    float
    """
    def objective(params):
        omega = float(np.clip(params[0], 0.0, 1.0))
        return log_det_fused_covariance(xa, Pa, xb, Pb, omega)

    xbest, _ = cma.fmin2(
        objective,
        [x0, x0],  # padded to 2-D — see module docstring
        sigma0,
        options={"bounds": [0.0, 1.0], "maxiter": maxiter, "verbose": -9, "seed": seed},
    )
    return float(np.clip(xbest[0], 0.0, 1.0))
