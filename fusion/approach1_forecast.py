"""
Approach 1, steps 2-3 — train the forecasting LSTM and use it at inference
(blueprint 2.3):

    "Step 2: train an LSTM ... to predict future omega values from
    historical omega sequences ... Step 3: at inference, feed the trained
    LSTM the recent omega history to forecast omega for the next several
    time steps, which are then plugged into the CI fusion equation."

Training scheme: one-step-ahead, shifted-target sequence training — a
standard way to train a forecasting RNN. For a window of omega values
[o0, o1, ..., on], the input is [o0, ..., o(n-1)] and the target is
[o1, ..., on] (i.e. "predict the next value from everything up to here",
evaluated at every position in the window at once, since the model outputs
one prediction per input timestep). At inference this generalises directly
to multi-step forecasting: feed the model its own most recent prediction as
the next input and repeat ("autoregressive rollout") to get "the next
several time steps" the blueprint asks for, not just one.
"""
import numpy as np
import torch
import torch.nn as nn

from models.omega_lstm import OmegaForecastLSTM

WINDOW_LEN = 6  # each training window is WINDOW_LEN omega values (X/Y each WINDOW_LEN-1)


def make_windows(sequences, window_len=WINDOW_LEN):
    """Slide a window of length ``window_len`` over every sequence long
    enough to contain one, split each window into (X, Y) shifted by one.

    Returns
    -------
    X, Y : np.ndarray, shape (num_windows, window_len - 1, 1)
    """
    X, Y = [], []
    for seq in sequences:
        for start in range(0, len(seq) - window_len + 1):
            window = seq[start:start + window_len]
            X.append(window[:-1])
            Y.append(window[1:])
    if not X:
        raise ValueError(
            f"No sequence is >= window_len={window_len} steps long — "
            "generate more/longer sequences, or reduce window_len."
        )
    X = np.asarray(X, dtype=np.float32)[..., np.newaxis]
    Y = np.asarray(Y, dtype=np.float32)[..., np.newaxis]
    return X, Y


def train_val_split(X, Y, val_fraction=0.2, seed=0):
    rng = np.random.RandomState(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction)) if n > 1 else 0
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]


def train(model, X_train, Y_train, X_val, Y_val, epochs=20, lr=0.001, batch_size=16):
    """MSE loss, Adam optimizer, ~20 epochs (blueprint's specified setup),
    with a held-out validation split to check for overfitting (blueprint
    Phase 2 deliverable: "training/validation loss curves").

    Returns
    -------
    train_losses, val_losses : list of float, one entry per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train_t = torch.from_numpy(X_train)
    Y_train_t = torch.from_numpy(Y_train)
    X_val_t = torch.from_numpy(X_val)
    Y_val_t = torch.from_numpy(Y_val)

    train_losses, val_losses = [], []
    n = len(X_train_t)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            batch_idx = perm[start:start + batch_size]
            xb, yb = X_train_t[batch_idx], Y_train_t[batch_idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)
        train_losses.append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_losses.append(loss_fn(val_pred, Y_val_t).item())

    return train_losses, val_losses


def forecast_omega(model, omega_history, num_future_steps):
    """Autoregressively forecast ``num_future_steps`` future omega values
    given a 1D array of recent omega history.
    """
    model.eval()
    history = list(omega_history)
    predictions = []
    with torch.no_grad():
        for _ in range(num_future_steps):
            x = torch.tensor(history, dtype=torch.float32).reshape(1, -1, 1)
            next_omega = model(x)[0, -1, 0].item()
            predictions.append(next_omega)
            history.append(next_omega)
    return np.array(predictions)


def evaluate_approach1(model, num_steps=16, seed=None, scenario=None, warmup_steps=5):
    """Real (non-differentiable) evaluation — mirrors
    fusion.approach2_train.evaluate_approach2 / approach3_train.evaluate_approach3
    / approach4_evaluate.evaluate_approach4, added later for Phase 6's
    cross-approach comparison (Approaches 2-4 had one from the start;
    Approach 1 didn't need one until there was something to compare it
    against).

    Inference procedure, matching the blueprint's description ("feed the
    trained LSTM the recent omega history to forecast omega for the next
    several time steps"): the LSTM forecasts, it doesn't compute omega from
    scratch — it needs a "recent history" to extrapolate from. The first
    ``warmup_steps`` steps use the real KL-optimal omega (fusion.kl_objective,
    same as training-label generation) to build that history; every step
    after that uses the model's own autoregressive forecast, with no further
    KL optimization — that's the part actually being evaluated. Fuses with
    the real numpy covariance_intersection either way, and reports OSPA/SIAP
    over the whole resulting track (warmup included, for comparability with
    the other approaches' whole-track metrics) — but note only the
    post-warmup portion reflects the model's own predictions.

    Parameters
    ----------
    scenario : sim.scenarios.ScenarioConfig, optional
        None (default) uses Phase 1's baseline scenario.
    """
    from fusion.covariance_intersection import covariance_intersection
    from fusion.kl_objective import optimize_omega_kl
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

    omega_history = []
    fused_states = []
    for i, (timestamp, airborne_state, ground_state) in enumerate(valid_steps):
        xa, Pa = airborne_state.state_vector, airborne_state.covar
        xb, Pb = ground_state.state_vector, ground_state.covar

        if i < warmup_steps:
            omega = optimize_omega_kl(xa, Pa, xb, Pb)
        else:
            omega = float(forecast_omega(model, omega_history, num_future_steps=1)[0])
        omega_history.append(omega)

        xc, Pc = covariance_intersection(xa, Pa, xb, Pb, omega)
        fused_states.append(GaussianState(xc, Pc, timestamp=timestamp))

    fused_track = Track(fused_states)
    manager = build_metric_manager()
    return compute_metrics(manager, {fused_track}, {truth})


def main():
    from sim.generate_omega_dataset import generate_dataset, save_dataset

    sequences = generate_dataset(num_runs=30, num_steps=16, seed_start=0)
    save_dataset(sequences)

    X, Y = make_windows(sequences)
    X_train, Y_train, X_val, Y_val = train_val_split(X, Y)
    print(f"{len(X)} training windows ({len(X_train)} train / {len(X_val)} val)")

    model = OmegaForecastLSTM()
    train_losses, val_losses = train(model, X_train, Y_train, X_val, Y_val)
    for epoch, (tl, vl) in enumerate(zip(train_losses, val_losses), 1):
        print(f"  epoch {epoch:2d}  train_loss={tl:.6f}  val_loss={vl:.6f}")

    torch.save(model.state_dict(), "data/omega_lstm.pt")
    np.savez("data/approach1_loss_curves.npz",
             train_loss=np.array(train_losses), val_loss=np.array(val_losses))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.legend()
    fig.savefig("data/approach1_loss_curves.png")
    print("Saved model to data/omega_lstm.pt, loss curves to "
          "data/approach1_loss_curves.{npz,png}")

    example_history = sequences[0][:4]
    forecast = forecast_omega(model, example_history, num_future_steps=3)
    print(f"Example forecast from history {np.round(example_history, 3)}: "
          f"{np.round(forecast, 3)}")


if __name__ == "__main__":
    main()
