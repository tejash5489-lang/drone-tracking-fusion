"""
Phase 7 — alternative sequence models for Approach 2 (Dynamic Fusing).

Both classes are drop-in replacements for models.dynamic_fusing_lstm.
DynamicFusingLSTM: same constructor arguments, same input
(batch, seq_len, 2 * n*(n+1)/2) of upper-triangle covariance features from
models.dynamic_fusing_lstm.make_features, same output (batch, seq_len)
omega in [0, 1]. That means fusion.approach2_train's train(),
run_sequence() and evaluate_approach2() work on them unchanged.

Both are *causal*: omega at step t depends only on steps <= t, exactly like
the unidirectional LSTM. That matters because the fused track is meant to be
usable step by step, and because a model that peeked at future covariances
would look better in evaluation than it could ever be in use.

  DynamicFusingGRU  one GRU layer, hidden_size=50 (same width as the LSTM).
  DynamicFusingTCN  temporal convolutional network: dilated causal 1-D
                    convolutions with residual connections. Kernel 3 with
                    dilations (1, 2, 4) gives each output a receptive field
                    of 1 + 2*(1+2+4)*2 = 29 steps (two convs per block), which
                    covers the ~15-16 step runs used here.

Run  python -m models.sequence_variants  for a self-test (shapes, omega
range, and a causality check: changing the input at step k must not change
the output at any earlier step).
"""
import torch
import torch.nn as nn


class DynamicFusingGRU(nn.Module):
    def __init__(self, state_dim=6, hidden_size=50):
        super().__init__()
        input_size = 2 * (state_dim * (state_dim + 1) // 2)
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """x: (batch, seq_len, input_size) -> omega: (batch, seq_len) in [0, 1]."""
        out, _ = self.gru(x)
        return torch.sigmoid(self.output(out)).squeeze(-1)


class _CausalConv1d(nn.Module):
    """Conv1d that pads only on the left, so output[t] never sees input[>t]."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):  # x: (batch, channels, time)
        return self.conv(nn.functional.pad(x, (self.left_pad, 0)))


class _TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = _CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.act = nn.ReLU()
        self.skip = (nn.Conv1d(in_channels, out_channels, 1)
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        y = self.conv2(self.act(self.conv1(x)))
        return self.act(y + self.skip(x))


def signed_log(x):
    """sign(x) * log(1 + |x|): compresses the huge dynamic range of raw
    covariance entries while keeping order and sign. Elementwise, so it
    cannot break causality."""
    return torch.sign(x) * torch.log1p(x.abs())


class DynamicFusingTCN(nn.Module):
    """input_transform:
        "none"        features go straight into the convolutions.
        "signed_log"  features pass through signed_log first.

    Why the option exists: LSTM/GRU squash inputs through bounded gates, so
    they tolerate raw covariance magnitudes. Convolution + ReLU has no such
    bound, so raw inputs can saturate the sigmoid output and stall training.
    A first trial showed exactly that pattern for the "none" TCN; the
    "signed_log" variant tests whether input scale was the cause.
    """

    def __init__(self, state_dim=6, hidden_size=50, kernel_size=3, dilations=(1, 2, 4),
                 input_transform="none"):
        super().__init__()
        if input_transform not in ("none", "signed_log"):
            raise ValueError(f"unknown input_transform {input_transform!r}")
        self.input_transform = input_transform
        input_size = 2 * (state_dim * (state_dim + 1) // 2)
        blocks, in_ch = [], input_size
        for d in dilations:
            blocks.append(_TemporalBlock(in_ch, hidden_size, kernel_size, d))
            in_ch = hidden_size
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """x: (batch, seq_len, input_size) -> omega: (batch, seq_len) in [0, 1]."""
        if self.input_transform == "signed_log":
            x = signed_log(x)
        h = self.blocks(x.transpose(1, 2)).transpose(1, 2)  # (batch, time, hidden)
        return torch.sigmoid(self.output(h)).squeeze(-1)


def _self_test():
    from models.dynamic_fusing_lstm import DynamicFusingLSTM

    torch.manual_seed(0)
    x = torch.randn(3, 16, 42)
    variants = [
        ("LSTM", DynamicFusingLSTM), ("GRU", DynamicFusingGRU),
        ("TCN (raw)", DynamicFusingTCN),
        ("TCN (log input)", lambda: DynamicFusingTCN(input_transform="signed_log")),
    ]
    for label, cls in variants:
        model = cls().eval()
        n_params = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            omega = model(x)
            assert omega.shape == (3, 16), (label, omega.shape)
            assert float(omega.min()) >= 0.0 and float(omega.max()) <= 1.0

            # Causality: perturb steps >= k, outputs before k must not move.
            k = 9
            x2 = x.clone()
            x2[:, k:, :] += torch.randn_like(x2[:, k:, :])
            diff_before = (model(x2)[:, :k] - omega[:, :k]).abs().max().item()
            diff_after = (model(x2)[:, k:] - omega[:, k:]).abs().max().item()
            assert diff_before < 1e-6, f"{label} is not causal ({diff_before})"
            assert diff_after > 1e-6, f"{label} ignores its input"
        print(f"{label:16s} params={n_params:6d}  shape ok, omega in [0,1], causal ok")
    v = torch.tensor([-1e6, -1.0, 0.0, 1.0, 1e6])
    out = signed_log(v)
    assert torch.all(out[:2] < 0) and out[2] == 0 and torch.all(out[3:] > 0)
    assert float(out.abs().max()) < 15 and torch.all(out[1:] >= out[:-1])  # compressed and monotonic
    print("signed_log ok (compresses, monotonic, sign-preserving)")
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    _self_test()