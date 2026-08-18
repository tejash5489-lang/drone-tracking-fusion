"""
Nearest-neighbor lookup — Phase 5, Approach 4 (blueprint 2.3):

    "At inference, the current sensor states are compared to every row in
    the database using a normalized Euclidean distance (Eq. 16-17) and the
    omega of the closest match is applied — no optimization is run live."

"Normalized" here means z-score normalized against the knowledge base's own
feature statistics before computing Euclidean distance — the 42 covariance-
triangle features span very different scales (position vs. velocity
variances, diagonal vs. off-diagonal terms), so an un-normalized Euclidean
distance would be dominated by whichever raw feature happens to have the
largest numeric range rather than the most informative one.
"""
import numpy as np


def flatten_covariance_np(P):
    """Upper-triangular (incl. diagonal) elements of a symmetric covariance
    matrix: (..., n, n) -> (..., n*(n+1)/2). Numpy twin of
    models.dynamic_fusing_lstm.flatten_covariance, kept numpy-native here
    since Approach 4 has no PyTorch dependency otherwise — CMA-ES is
    explicitly the non-deep-learning approach.
    """
    n = P.shape[-1]
    rows, cols = np.triu_indices(n)
    return P[..., rows, cols]


class OmegaKnowledgeBase:
    """Fitted knowledge base: stores feature normalization stats + the
    (normalized feature, omega) records, and answers nearest-neighbor
    lookups against them.
    """

    def __init__(self, features, omegas):
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        self.std[self.std == 0] = 1.0  # avoid divide-by-zero on a constant feature
        self._normalized = (features - self.mean) / self.std
        self.omegas = omegas

    @classmethod
    def load(cls, path="data/cma_es_knowledgebase.npz"):
        with np.load(path) as data:
            return cls(data["features"], data["omegas"])

    def lookup(self, Pa, Pb):
        """Return (omega, distance) for the closest-matching knowledge-base
        record to this (Pa, Pb) pair — no optimization, just a search."""
        feature = np.concatenate([flatten_covariance_np(Pa), flatten_covariance_np(Pb)])
        normalized = (feature - self.mean) / self.std
        distances = np.linalg.norm(self._normalized - normalized, axis=1)
        best = np.argmin(distances)
        return float(self.omegas[best]), float(distances[best])
