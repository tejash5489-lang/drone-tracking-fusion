import torch
import torch.nn as nn


class CausalOmegaTransformer(nn.Module):
    """Predicts omega_t in (0,1) from a window of per-step fusion features.
    Causal mask: step t only attends to steps <= t (usable online)."""

    def __init__(self, in_dim, d_model=64, nhead=4, n_layers=2,
                 dim_ff=128, dropout=0.1, max_len=512):
        super().__init__()
        self.inp = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):                         # x: (B, T, in_dim)
        T = x.size(1)
        h = self.inp(x) + self.pos[:, :T]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.enc(h, mask=mask)
        return 0.01 + 0.98 * torch.sigmoid(self.head(h)).squeeze(-1)   # (B, T)


class LSTMOmega(nn.Module):
    """LSTM baseline trained with the exact same loss, for a fair comparison."""

    def __init__(self, in_dim, hidden=64, layers=2):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        return 0.01 + 0.98 * torch.sigmoid(self.head(h)).squeeze(-1)