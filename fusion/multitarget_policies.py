"""
Per-segment omega policies for multi-target evaluation.

Each policy takes one gap-free segment — a list of
(timestamp, airborne_state, ground_state) triples, i.e. exactly the
``valid_steps`` shape the single-target evaluate_approachN functions build —
and returns a list of fused (xc, Pc) pairs, one per step.

Why segments instead of a per-step omega callback: Approaches 1-3 are
LSTMs that need a target's *history*, and every LSTM here is unidirectional,
so running one over a whole segment at once gives the same omega at step t
as feeding it step by step. Association (fusion.track_association) doesn't
depend on omega, so segments can be built first and omega applied after.

The omega logic below is copied from fusion.approach{1,2,3,4}'s
evaluate_approachN functions (not imported, because those functions run
their own single-target simulation inline). If you change how one of those
computes omega, mirror the change here.

Known train/test shift: models are trained on full-length single-target
sequences, but a segment can start mid-run and be short, and LSTM hidden
state restarts at each segment. Approach 1 additionally spends its first
``warmup_steps`` steps on real KL optimisation, so very short segments
barely exercise the model at all.
"""
import numpy as np
import torch

from fusion.covariance_intersection import covariance_intersection


def _fuse(steps, omegas, states=None):
    out = []
    for i, omega in enumerate(omegas):
        if states is None:
            _, a, g = steps[i]
            xa, Pa, xb, Pb = a.state_vector, a.covar, g.state_vector, g.covar
        else:
            xa, Pa, xb, Pb = (s[i] for s in states)
        out.append(covariance_intersection(xa, Pa, xb, Pb, float(omega)))
    return out


def fixed_policy(omega=0.5):
    def policy(steps):
        return _fuse(steps, [omega] * len(steps))
    return policy


def approach1_policy(model, warmup_steps=5):
    from fusion.approach1_forecast import forecast_omega
    from fusion.kl_objective import optimize_omega_kl

    def policy(steps):
        omega_history, omegas = [], []
        for i, (_, a, g) in enumerate(steps):
            if i < warmup_steps:
                omega = optimize_omega_kl(a.state_vector, a.covar, g.state_vector, g.covar)
            else:
                omega = float(forecast_omega(model, omega_history, num_future_steps=1)[0])
            omega_history.append(omega)
            omegas.append(omega)
        return _fuse(steps, omegas)
    return policy


def approach2_policy(model):
    from models.dynamic_fusing_lstm import make_features

    def policy(steps):
        Pa = torch.from_numpy(np.stack([a.covar for _, a, _ in steps]).astype(np.float32))
        Pb = torch.from_numpy(np.stack([g.covar for _, _, g in steps]).astype(np.float32))
        model.eval()
        with torch.no_grad():
            omegas = model(make_features(Pa, Pb).unsqueeze(0)).squeeze(0).numpy()
        return _fuse(steps, omegas)
    return policy


def approach3_policy(model_a, model_b, model3, F):
    from fusion.approach3_innovation import refine_sequence
    from models.dynamic_fusing_lstm import make_features

    def policy(steps):
        xa = torch.from_numpy(np.stack([a.state_vector for _, a, _ in steps]).astype(np.float32))
        Pa = torch.from_numpy(np.stack([a.covar for _, a, _ in steps]).astype(np.float32))
        xb = torch.from_numpy(np.stack([g.state_vector for _, _, g in steps]).astype(np.float32))
        Pb = torch.from_numpy(np.stack([g.covar for _, _, g in steps]).astype(np.float32))
        for m in (model_a, model_b, model3):
            m.eval()
        with torch.no_grad():
            rxa, rPa, _, _ = refine_sequence(model_a, xa, Pa, F)
            rxb, rPb, _, _ = refine_sequence(model_b, xb, Pb, F)
            omegas = model3(make_features(rPa, rPb).unsqueeze(0)).squeeze(0).numpy()
        return _fuse(steps, omegas, states=(rxa.numpy(), rPa.numpy(), rxb.numpy(), rPb.numpy()))
    return policy


def approach4_policy(knowledge_base):
    def policy(steps):
        omegas = [knowledge_base.lookup(a.covar, g.covar)[0] for _, a, g in steps]
        return _fuse(steps, omegas)
    return policy