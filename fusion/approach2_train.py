"""
Approach 2 training — Phase 3 (blueprint 2.3):

    "It is trained end-to-end with a custom loss that penalizes OSPA and
    SIAP discrepancy between the LSTM-fused track and ground truth (with a
    non-negativity constraint via max(0, ·) on both terms), using alpha and
    beta weighting coefficients. Trained over 500 time steps, 50 epochs,
    Adam (lr = 0.001)."

Blueprint's own flagged risk (section 4.6) is the reason this file exists
in this shape rather than just calling Stone Soup's OSPA/SIAP generators
directly inside the loss:

    "Differentiable OSPA/SIAP for Approach 2's training loss: these metrics
    are not natively differentiable, so a custom/non-standard training loop
    (or a smooth surrogate loss) will likely be needed instead of a plain
    Keras .fit() call."

The surrogate used here: for our current *single-target* scenario, OSPA
distance and SIAP's positional/velocity-accuracy submetrics all reduce to
plain Euclidean position/velocity error (OSPA's extra machinery — optimal
assignment, a cardinality penalty — only does anything when the number of
tracks vs. truths can differ, which doesn't arise with one confirmed
target). So the training loss below is that reduction, which — unlike the
real metrics — is differentiable end-to-end through the LSTM -> omega ->
covariance_intersection_torch chain. It is deliberately *not* the same code
path as eval.metrics: the real OSPA/SIAP (multi-target-capable, exactly
what Phase 6 will compare methods on) are computed separately at
evaluation time in evaluate_approach2() below, via the real (numpy)
covariance_intersection — training and evaluation use different
implementations of "how good is this track", by design, because only one
of them can be backpropagated through.

The paper's max(0, ·) non-negativity constraint is redundant for a
Euclidean-norm surrogate (already >= 0 by construction) so it's omitted
here rather than applied for its own sake — noted for when this loss is
revisited against a truer OSPA/SIAP surrogate later.
"""
import numpy as np
import torch

from models.dynamic_fusing_lstm import DynamicFusingLSTM, make_features
from fusion.covariance_intersection_torch import covariance_intersection_torch

POSITION_MAPPING = (0, 2, 4)
VELOCITY_MAPPING = (1, 3, 5)


def approach2_loss(xc_seq, truth_seq, alpha=1.0, beta=1.0):
    """Differentiable position/velocity-error surrogate for OSPA/SIAP
    discrepancy (see module docstring for why this isn't the real metric).

    xc_seq, truth_seq : torch.Tensor, shape (T, n, 1)
    """
    position_error = (xc_seq[:, POSITION_MAPPING, 0] - truth_seq[:, POSITION_MAPPING, 0])
    velocity_error = (xc_seq[:, VELOCITY_MAPPING, 0] - truth_seq[:, VELOCITY_MAPPING, 0])
    position_term = position_error.norm(dim=-1)  # (T,) euclidean distance per step
    velocity_term = velocity_error.norm(dim=-1)
    return (alpha * position_term + beta * velocity_term).mean()


def _to_tensors(sequence):
    return {k: torch.from_numpy(v) for k, v in sequence.items()}


def run_sequence(model, sequence):
    """Forward one sequence through the LSTM -> differentiable CI, returning
    the fused state trajectory xc_seq (T, n, 1)."""
    t = _to_tensors(sequence)
    features = make_features(t["Pa"], t["Pb"]).unsqueeze(0)  # (1, T, input_size)
    omega = model(features).squeeze(0)  # (T,)
    xc_seq, _ = covariance_intersection_torch(t["xa"], t["Pa"], t["xb"], t["Pb"], omega)
    return xc_seq, t["truth"]


def train(model, train_sequences, val_sequences, epochs=50, lr=0.001, alpha=1.0, beta=1.0):
    """Adam, 50 epochs (blueprint's specified setup for Approach 2).

    One gradient step per sequence (full trajectory) rather than per
    windowed batch — sequences here are whole simulation runs of differing
    length, and the loss genuinely depends on the *entire* trajectory's
    fused states (not a single step), so there's no shorter unit of work
    to batch over without padding/masking, which isn't worth the added
    complexity at this dataset size.

    Returns
    -------
    train_losses, val_losses : list of float, one entry per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for sequence in train_sequences:
            optimizer.zero_grad()
            xc_seq, truth_seq = run_sequence(model, sequence)
            loss = approach2_loss(xc_seq, truth_seq, alpha=alpha, beta=beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_sequences))

        model.eval()
        with torch.no_grad():
            val_loss = sum(
                approach2_loss(*run_sequence(model, seq), alpha=alpha, beta=beta).item()
                for seq in val_sequences
            ) / len(val_sequences)
        val_losses.append(val_loss)

    return train_losses, val_losses


def evaluate_approach2(model, num_steps=16, seed=None, scenario=None):
    """Real (non-differentiable) evaluation, matching how Phase 1's
    baseline and Approach 1 are reported — see sim/run_baseline.py.

    Feeds a fresh simulation's covariance sequence through the trained LSTM
    to get predicted omega(t), then fuses using the *real* numpy
    covariance_intersection (not the differentiable training-only one) to
    build an actual Stone Soup Track, and reports OSPA/SIAP via
    eval.metrics — the same metrics every other approach in this project is
    judged on.

    Parameters
    ----------
    scenario : sim.scenarios.ScenarioConfig, optional
        None (default) uses Phase 1's baseline scenario (unchanged
        behaviour) — pass one of sim.scenarios.SINGLE_TARGET_SCENARIOS for
        Phase 6's cross-scenario comparison.
    """
    from fusion.covariance_intersection import covariance_intersection
    from sim.two_radar_simulation import run_two_radar_simulation, run_single_target_scenario
    from eval.metrics import build_metric_manager, compute_metrics
    from stonesoup.types.state import GaussianState
    from stonesoup.types.track import Track

    if scenario is None:
        truth, steps, _, _ = run_two_radar_simulation(num_steps=num_steps, seed=seed)
    else:
        truth, steps, _, _ = run_single_target_scenario(scenario, seed=seed)

    valid_steps = [(ts, a, g) for ts, a, g in steps if a is not None and g is not None]
    if not valid_steps:
        raise RuntimeError("No steps with both sensors active — can't evaluate.")

    Pa_seq = torch.from_numpy(np.stack([a.covar for _, a, _ in valid_steps]).astype(np.float32))
    Pb_seq = torch.from_numpy(np.stack([g.covar for _, _, g in valid_steps]).astype(np.float32))
    model.eval()
    with torch.no_grad():
        omega_seq = model(make_features(Pa_seq, Pb_seq).unsqueeze(0)).squeeze(0).numpy()

    fused_states = []
    for (timestamp, airborne_state, ground_state), omega in zip(valid_steps, omega_seq):
        xa, Pa = airborne_state.state_vector, airborne_state.covar
        xb, Pb = ground_state.state_vector, ground_state.covar
        xc, Pc = covariance_intersection(xa, Pa, xb, Pb, float(omega))
        fused_states.append(GaussianState(xc, Pc, timestamp=timestamp))

    fused_track = Track(fused_states)
    manager = build_metric_manager()
    return compute_metrics(manager, {fused_track}, {truth})


def main():
    from sim.generate_approach2_dataset import generate_dataset

    sequences = generate_dataset(num_runs=30, num_steps=16, seed_start=0)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(sequences))
    n_val = max(1, int(len(sequences) * 0.2))
    val_sequences = [sequences[i] for i in idx[:n_val]]
    train_sequences = [sequences[i] for i in idx[n_val:]]
    print(f"{len(sequences)} sequences ({len(train_sequences)} train / {len(val_sequences)} val)")

    model = DynamicFusingLSTM()
    train_losses, val_losses = train(model, train_sequences, val_sequences)
    for epoch in (1, 10, 25, 50):
        if epoch <= len(train_losses):
            print(f"  epoch {epoch:2d}  train_loss={train_losses[epoch-1]:.4f}  "
                  f"val_loss={val_losses[epoch-1]:.4f}")

    torch.save(model.state_dict(), "data/dynamic_fusing_lstm.pt")
    np.savez("data/approach2_loss_curves.npz",
             train_loss=np.array(train_losses), val_loss=np.array(val_losses))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("position + velocity error (surrogate loss)")
    ax.legend()
    fig.savefig("data/approach2_loss_curves.png")
    print("Saved model to data/dynamic_fusing_lstm.pt, loss curves to "
          "data/approach2_loss_curves.{npz,png}")

    print("\n=== Approach 2 (real OSPA/SIAP, via covariance_intersection.py) ===")
    for k, v in evaluate_approach2(model, num_steps=16, seed=2026).items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
