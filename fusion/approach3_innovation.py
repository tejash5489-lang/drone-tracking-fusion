"""
Approach 3, LSTM-1 / LSTM-2 training — innovation minimization.

See models/innovation_lstm.py for the interpretive design note this
builds on before trusting the specifics here.

"Innovation" is defined in *state* space (not raw measurement space):
how far a step's tracked state estimate lands from what pure dynamics
alone would have predicted from the previous *refined* estimate. Small
innovation means the sensor's update roughly agreed with where the target
should be anyway; large innovation means something surprising happened
(a noisy detection, clutter pulling the track, a maneuver) — exactly what
a gating threshold is meant to be skeptical of.

Recurrence per sequence, given tracked states x(t) and covariances P(t)
from one sensor, and the (fixed, known) NCV transition matrix F:

    raw_innovation(t)   = x(t) - F @ x(t-1)          [x(-1) := x(0)]
    trust(t)             = LSTM(raw_innovation sequence)     — this is the
                            *only* LSTM forward pass; it's a plain
                            vectorized call, not part of the recurrence
                            below, since raw_innovation doesn't depend on
                            refined(t-1).
    predicted_state(t)  = F @ refined(t-1)
    predicted_covar(t)  = F @ P_refined(t-1) @ F^T   (deterministic part of
                            covariance prediction; process-noise growth is
                            omitted — a simplification, not a claim that
                            uncertainty doesn't grow between updates)
    refined(t)           = trust(t) * x(t) + (1 - trust(t)) * predicted_state(t)
    P_refined(t)         = trust(t)^2 * P(t) + (1 - trust(t))^2 * predicted_covar(t)
                            (variance of a weighted sum of independent
                            estimates — standard, if trust and P(t) are
                            treated as independent, which is an
                            approximation)
    innovation(t)        = x(t) - predicted_state(t)  — this is what's
                            actually minimized: the gap between the tracked
                            estimate and what the *refined* trajectory
                            predicted, which trust(t) directly controls
                            through predicted_state(t)'s dependence on
                            refined(t-1).

Loss = mean(|innovation|) + mean(trace(P_refined)) — both terms genuinely
depend on the LSTM's trust weights, so both contribute real gradient.
"""
from datetime import timedelta

import numpy as np
import torch

from sim.scenario import build_transition_model
from models.innovation_lstm import InnovationLSTM


def transition_matrix_torch():
    F = np.asarray(
        build_transition_model().matrix(time_interval=timedelta(seconds=1)), dtype=np.float32)
    return torch.from_numpy(F)


def refine_sequence(model, x_seq, P_seq, F):
    """Run one sensor's (x_seq, P_seq) through the trust-weight recurrence.

    Parameters
    ----------
    x_seq : torch.Tensor, shape (T, n, 1)
    P_seq : torch.Tensor, shape (T, n, n)
    F : torch.Tensor, shape (n, n)

    Returns
    -------
    refined_seq, P_refined_seq : torch.Tensor, same shapes as x_seq, P_seq
    innovation : torch.Tensor, shape (T, n)
    trust : torch.Tensor, shape (T,)
    """
    T = x_seq.shape[0]

    predicted_from_raw = F @ x_seq[:-1]  # (T-1, n, 1)
    raw_innovation = torch.cat(
        [torch.zeros_like(x_seq[:1]), x_seq[1:] - predicted_from_raw], dim=0
    ).squeeze(-1)  # (T, n)
    trust = model(raw_innovation.unsqueeze(0)).squeeze(0)  # (T,)

    refined = [x_seq[0]]
    P_refined = [P_seq[0]]
    innovations = [torch.zeros_like(x_seq[0].squeeze(-1))]
    for t in range(1, T):
        predicted_state = F @ refined[-1]
        predicted_covar = F @ P_refined[-1] @ F.T

        r = trust[t]
        refined.append(r * x_seq[t] + (1 - r) * predicted_state)
        P_refined.append((r ** 2) * P_seq[t] + ((1 - r) ** 2) * predicted_covar)
        innovations.append((x_seq[t] - predicted_state).squeeze(-1))

    refined_seq = torch.stack(refined, dim=0)
    P_refined_seq = torch.stack(P_refined, dim=0)
    innovation = torch.stack(innovations, dim=0)
    return refined_seq, P_refined_seq, innovation, trust


def innovation_loss(innovation, P_refined_seq, alpha=1.0, beta=1.0):
    mae = innovation.abs().mean()
    trace_term = torch.diagonal(P_refined_seq, dim1=-2, dim2=-1).sum(-1).mean()
    return alpha * mae + beta * trace_term


def train_innovation_lstm(model, sequences, x_key, P_key, F, epochs=20, lr=0.001):
    """Train one sensor's InnovationLSTM (blueprint: 20 epochs, Adam lr=0.001).

    sequences : list of dict, from sim.generate_approach2_dataset — reused
    as-is since Approach 3 needs exactly the same (per-sensor state,
    covariance, truth) shape Approach 2's dataset already provides.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for seq in sequences:
            x_seq = torch.from_numpy(seq[x_key])
            P_seq = torch.from_numpy(seq[P_key])

            optimizer.zero_grad()
            _, P_refined_seq, innovation, _ = refine_sequence(model, x_seq, P_seq, F)
            loss = innovation_loss(innovation, P_refined_seq)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(sequences))
    return losses
