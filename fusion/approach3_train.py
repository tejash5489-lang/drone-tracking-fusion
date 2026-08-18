"""
Approach 3 — full IMM-LSTM pipeline (blueprint 2.3 & 3, Phase 4):

    "Wire the three-LSTM pipeline together per Fig. 13's block diagram:
    Sensor1/2 -> LSTM1/2 -> innovation-error check -> fusing LSTM ->
    fused track -> gating-threshold update loop."

Staged (not joint) training: LSTM-1/2 are trained first (see
fusion/approach3_innovation.py) to convergence, then frozen; LSTM-3 is
trained on top of their *refined* outputs. This is the natural reading of
the diagram's left-to-right flow, and is far more tractable than training
all three simultaneously — the blueprint's "gating-threshold update loop"
suggests the paper may alternate/iterate between stages, which would be a
reasonable next step on top of this, not something this scaffold attempts.

LSTM-3 reuses models.dynamic_fusing_lstm.DynamicFusingLSTM as-is (same
covariance-features -> omega(t) architecture Approach 2 uses) rather than
a near-duplicate class, since the computational role is identical — only
the covariances it's fed differ (P_refined from LSTM-1/2, not raw sensor
covariances).

LSTM-3's loss is plain RMSE between the fused track and ground truth
(blueprint's Eq. 13) — notably simpler than Approach 2's loss, since the
paper doesn't ask this one to approximate OSPA/SIAP.

Epoch count: the blueprint states "20 epochs for LSTM-1/2" but doesn't
give LSTM-3's count. 30 is used below as a reasonable default, not a value
taken from the paper — revisit once you've read Eq. 13's surrounding text.
"""
import numpy as np
import torch

from models.innovation_lstm import InnovationLSTM
from models.dynamic_fusing_lstm import DynamicFusingLSTM, make_features
from fusion.approach3_innovation import (
    transition_matrix_torch, refine_sequence, train_innovation_lstm,
)
from fusion.covariance_intersection_torch import covariance_intersection_torch


def rmse_loss(xc_seq, truth_seq):
    return torch.sqrt(((xc_seq - truth_seq) ** 2).mean())


def refine_dataset(model_a, model_b, sequences, F):
    """Run (frozen) LSTM-1/2 over every sequence, returning a new list of
    dicts with refined states/covariances added alongside the originals."""
    model_a.eval()
    model_b.eval()
    refined = []
    with torch.no_grad():
        for seq in sequences:
            xa = torch.from_numpy(seq["xa"])
            Pa = torch.from_numpy(seq["Pa"])
            xb = torch.from_numpy(seq["xb"])
            Pb = torch.from_numpy(seq["Pb"])
            refined_xa, refined_Pa, _, _ = refine_sequence(model_a, xa, Pa, F)
            refined_xb, refined_Pb, _, _ = refine_sequence(model_b, xb, Pb, F)
            refined.append({
                **seq,
                "refined_xa": refined_xa.numpy(), "refined_Pa": refined_Pa.numpy(),
                "refined_xb": refined_xb.numpy(), "refined_Pb": refined_Pb.numpy(),
            })
    return refined


def run_fusing_sequence(model3, seq):
    refined_xa = torch.from_numpy(seq["refined_xa"])
    refined_Pa = torch.from_numpy(seq["refined_Pa"])
    refined_xb = torch.from_numpy(seq["refined_xb"])
    refined_Pb = torch.from_numpy(seq["refined_Pb"])
    truth = torch.from_numpy(seq["truth"])

    features = make_features(refined_Pa, refined_Pb).unsqueeze(0)
    omega = model3(features).squeeze(0)
    xc_seq, _ = covariance_intersection_torch(refined_xa, refined_Pa, refined_xb, refined_Pb, omega)
    return xc_seq, truth


def train_fusing_lstm(model3, train_sequences, val_sequences, epochs=30, lr=0.001):
    optimizer = torch.optim.Adam(model3.parameters(), lr=lr)
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model3.train()
        epoch_loss = 0.0
        for seq in train_sequences:
            optimizer.zero_grad()
            xc_seq, truth = run_fusing_sequence(model3, seq)
            loss = rmse_loss(xc_seq, truth)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_sequences))

        model3.eval()
        with torch.no_grad():
            val_loss = sum(
                rmse_loss(*run_fusing_sequence(model3, seq)).item() for seq in val_sequences
            ) / len(val_sequences)
        val_losses.append(val_loss)
    return train_losses, val_losses


def evaluate_approach3(model_a, model_b, model3, F, num_steps=16, seed=None, scenario=None):
    """Real (non-differentiable) evaluation — mirrors
    fusion.approach2_train.evaluate_approach2: feed a fresh simulation
    through the trained three-LSTM pipeline, fuse with the real numpy
    covariance_intersection, report real OSPA/SIAP.

    Parameters
    ----------
    scenario : sim.scenarios.ScenarioConfig, optional
        None (default) uses Phase 1's baseline scenario — pass one of
        sim.scenarios.SINGLE_TARGET_SCENARIOS for Phase 6's comparison.
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

    xa = torch.from_numpy(np.stack([a.state_vector for _, a, _ in valid_steps]).astype(np.float32))
    Pa = torch.from_numpy(np.stack([a.covar for _, a, _ in valid_steps]).astype(np.float32))
    xb = torch.from_numpy(np.stack([g.state_vector for _, _, g in valid_steps]).astype(np.float32))
    Pb = torch.from_numpy(np.stack([g.covar for _, _, g in valid_steps]).astype(np.float32))

    model_a.eval()
    model_b.eval()
    model3.eval()
    with torch.no_grad():
        refined_xa, refined_Pa, _, _ = refine_sequence(model_a, xa, Pa, F)
        refined_xb, refined_Pb, _, _ = refine_sequence(model_b, xb, Pb, F)
        features = make_features(refined_Pa, refined_Pb).unsqueeze(0)
        omega_seq = model3(features).squeeze(0).numpy()

    fused_states = []
    for (timestamp, _, _), xa_i, Pa_i, xb_i, Pb_i, omega in zip(
            valid_steps, refined_xa.numpy(), refined_Pa.numpy(),
            refined_xb.numpy(), refined_Pb.numpy(), omega_seq):
        xc, Pc = covariance_intersection(xa_i, Pa_i, xb_i, Pb_i, float(omega))
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
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_sequences = [sequences[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    print(f"{len(sequences)} sequences ({len(train_sequences)} train / {len(val_sequences)} val)")

    F = transition_matrix_torch()

    print("\n--- Training LSTM-1 (airborne, innovation minimization) ---")
    model_a = InnovationLSTM()
    losses_a = train_innovation_lstm(model_a, train_sequences, "xa", "Pa", F)
    print(f"  epoch 1: {losses_a[0]:.3f}  epoch {len(losses_a)}: {losses_a[-1]:.3f}")

    print("--- Training LSTM-2 (ground, innovation minimization) ---")
    model_b = InnovationLSTM()
    losses_b = train_innovation_lstm(model_b, train_sequences, "xb", "Pb", F)
    print(f"  epoch 1: {losses_b[0]:.3f}  epoch {len(losses_b)}: {losses_b[-1]:.3f}")

    print("--- Training LSTM-3 (fusing, RMSE) ---")
    refined_train = refine_dataset(model_a, model_b, train_sequences, F)
    refined_val = refine_dataset(model_a, model_b, val_sequences, F)
    model3 = DynamicFusingLSTM()
    train_losses3, val_losses3 = train_fusing_lstm(model3, refined_train, refined_val)
    for epoch in (1, 10, 30):
        if epoch <= len(train_losses3):
            print(f"  epoch {epoch:2d}  train_rmse={train_losses3[epoch-1]:.3f}  "
                  f"val_rmse={val_losses3[epoch-1]:.3f}")

    torch.save(model_a.state_dict(), "data/approach3_lstm1_airborne.pt")
    torch.save(model_b.state_dict(), "data/approach3_lstm2_ground.pt")
    torch.save(model3.state_dict(), "data/approach3_lstm3_fusing.pt")
    np.savez("data/approach3_loss_curves.npz",
             lstm1_loss=np.array(losses_a), lstm2_loss=np.array(losses_b),
             lstm3_train_rmse=np.array(train_losses3), lstm3_val_rmse=np.array(val_losses3))

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(losses_a); axes[0].set_title("LSTM-1 (airborne) innovation loss")
    axes[1].plot(losses_b); axes[1].set_title("LSTM-2 (ground) innovation loss")
    axes[2].plot(train_losses3, label="train"); axes[2].plot(val_losses3, label="val")
    axes[2].set_title("LSTM-3 fusing RMSE"); axes[2].legend()
    for ax in axes:
        ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig("data/approach3_loss_curves.png")
    print("\nSaved models to data/approach3_lstm{1,2,3}_*.pt, "
          "loss curves to data/approach3_loss_curves.{npz,png}")

    print("\n=== Approach 3 (real OSPA/SIAP, via covariance_intersection.py) ===")
    for k, v in evaluate_approach3(model_a, model_b, model3, F, num_steps=16, seed=2026).items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
