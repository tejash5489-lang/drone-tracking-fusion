"""
Approach 1 forecasting LSTM — Phase 2 (blueprint 2.3):

    "Step 2: train an LSTM (2 LSTM layers x 20 units, TimeDistributed
    Dense(10), MSE/RMSE loss, Adam optimizer, 20 epochs) as a
    sequence-to-sequence model to predict future omega values from
    historical omega sequences."

This is a univariate sequence-to-sequence model: input is a window of past
omega values (shape: batch x seq_len x 1), output is a same-length sequence
of predicted omega values (one prediction per input timestep — standard
seq2seq-via-shifted-target training, not just a single next-step scalar).

Translating "TimeDistributed Dense(10)" from Keras to PyTorch: in Keras,
TimeDistributed explicitly wraps a layer to apply it independently at every
timestep of a (batch, seq, features) tensor. In PyTorch, nn.Linear already
does this implicitly — applied to a 3D tensor, it multiplies only the last
dimension and broadcasts over every leading dimension (batch and seq alike)
— so a plain nn.Linear(20, 10) here *is* the TimeDistributed Dense(10),
no wrapper needed.

The paper doesn't specify a final activation. Omega must land in [0, 1] to
be valid for Covariance Intersection (see fusion/covariance_intersection.py),
so this model ends in a sigmoid — a deliberate addition, not something taken
directly from the paper's description.
"""
import torch
import torch.nn as nn


class OmegaForecastLSTM(nn.Module):
    def __init__(self, hidden_size=20, dense_size=10, num_lstm_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size,
            num_layers=num_lstm_layers, batch_first=True,
        )
        self.dense = nn.Linear(hidden_size, dense_size)
        self.activation = nn.ReLU()
        self.output = nn.Linear(dense_size, 1)

    def forward(self, x):
        """x: (batch, seq_len, 1) -> (batch, seq_len, 1), each value in [0, 1]."""
        lstm_out, _ = self.lstm(x)
        dense_out = self.activation(self.dense(lstm_out))
        return torch.sigmoid(self.output(dense_out))
