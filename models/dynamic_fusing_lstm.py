"""
Approach 2 architecture — Dynamic Fusing LSTM (blueprint 2.3):

    "Rather than learning omega from a pre-optimized label sequence, this
    LSTM (50 hidden units + linear output layer) takes the two sensors'
    raw track covariances directly as input and predicts omega(t) itself
    every step."

Feature engineering (not specified in the paper beyond "raw track
covariances"): each 6x6 covariance matrix is symmetric, so its upper
triangle (21 values, including the diagonal) carries all the information —
feeding the full 36-value flatten would just hand the network 15 duplicate
inputs per sensor. Both sensors' 21-value triangles are concatenated into a
42-dim feature vector per time step.

As in Approach 1's model (models/omega_lstm.py), the paper's "linear output
layer" is followed here by a sigmoid to keep omega in [0, 1] — CI's
information-space combination isn't a meaningful "fusion" outside that
range, and this is the same choice made there for the same reason.
"""
import torch
import torch.nn as nn


def flatten_covariance(P):
    """Upper-triangular (incl. diagonal) elements of a batch of symmetric
    covariance matrices: (..., n, n) -> (..., n*(n+1)/2)."""
    n = P.shape[-1]
    rows, cols = torch.triu_indices(n, n)
    return P[..., rows, cols]


def make_features(Pa_seq, Pb_seq):
    """Pa_seq, Pb_seq: (..., n, n) -> (..., n*(n+1)) concatenated feature vector."""
    return torch.cat([flatten_covariance(Pa_seq), flatten_covariance(Pb_seq)], dim=-1)


class DynamicFusingLSTM(nn.Module):
    def __init__(self, state_dim=6, hidden_size=50):
        super().__init__()
        input_size = 2 * (state_dim * (state_dim + 1) // 2)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """x: (batch, seq_len, input_size) -> omega: (batch, seq_len), each in [0, 1]."""
        lstm_out, _ = self.lstm(x)
        return torch.sigmoid(self.output(lstm_out)).squeeze(-1)
