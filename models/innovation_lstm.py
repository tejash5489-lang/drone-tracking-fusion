"""
Innovation-minimization LSTM (LSTM-1 / LSTM-2) — Phase 4, Approach 3
(blueprint 2.3):

    "LSTM-1 and LSTM-2 each process one sensor's predictions/measurements
    and are trained to minimize a combined objective of innovation MAE
    (difference between predicted and measured state) plus the trace of
    the innovation covariance — effectively this jointly tunes the JPDA
    tracker's gating threshold per sensor while training the network."

INTERPRETIVE NOTE — read before trusting this file: the paper analysis
this project is built from doesn't carry the closed-form Eq. 12, and
re-deriving what "tunes the JPDA gating threshold" means mechanically needs
an interpretive choice, not just a translation. JPDA's Mahalanobis gate is
a discrete, non-differentiable hyperparameter (which detections get
considered at all) — you can't backpropagate a loss through which
detections got gated out, the same non-differentiability problem
Approach 2 hit with OSPA/SIAP (see fusion/approach2_train.py). Once you've
read the paper's actual Eq. 12 and Fig. 13, revisit this file — it will
likely need reshaping to match.

The interpretation used here: rather than re-running JPDA with a
network-predicted discrete gate threshold, this LSTM outputs a continuous,
differentiable per-step *trust weight* r(t) in [0, 1] — the differentiable
analogue of a gate. A tight gate distrusts a surprising detection and
effectively falls back toward pure dynamics prediction (r near 0); a loose
gate accepts it (r near 1). See fusion/approach3_innovation.py for how r(t)
is used to blend the tracked estimate against a pure-dynamics prediction,
and how minimizing the resulting innovation trains this network to do
exactly what a well-tuned gate threshold would.
"""
import torch.nn as nn
import torch


class InnovationLSTM(nn.Module):
    def __init__(self, state_dim=6, hidden_size=20):
        super().__init__()
        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, innovation_seq):
        """innovation_seq: (batch, seq_len, state_dim) -> trust: (batch, seq_len), each in [0, 1]."""
        lstm_out, _ = self.lstm(innovation_seq)
        return torch.sigmoid(self.output(lstm_out)).squeeze(-1)
