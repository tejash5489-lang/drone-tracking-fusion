"""Approach 5: causal-Transformer omega selector trained END-TO-END through
Covariance Intersection (loss = fused-state error + consistency + smoothness).
Compares against fixed omega=0.5, a trace heuristic, and an LSTM trained with the same loss.

Run:  python -m fusion.approach5_train                      (synthetic data)
      python -m fusion.approach5_train --data data/my.npz   (your data)

npz keys: x1,x2,x_true (N,T,n) ; P1,P2 (N,T,n,n)
"""
import argparse
import os
import numpy as np
import torch

from models.omega_transformer import CausalOmegaTransformer, LSTMOmega


# ---------------- CI + loss (differentiable) ----------------
def ci_fuse(x1, P1, x2, P2, w):
    n = x1.size(-1)
    eye = torch.eye(n, device=x1.device)
    I1 = torch.linalg.inv(P1 + 1e-6 * eye)
    I2 = torch.linalg.inv(P2 + 1e-6 * eye)
    wm = w[..., None, None]
    P = torch.linalg.inv(wm * I1 + (1 - wm) * I2)
    b = wm * (I1 @ x1.unsqueeze(-1)) + (1 - wm) * (I2 @ x2.unsqueeze(-1))
    return (P @ b).squeeze(-1), P


def nees_of(x, P, xt):
    err = x - xt
    Pinv = torch.linalg.inv(P)
    return (err.unsqueeze(-2) @ Pinv @ err.unsqueeze(-1)).squeeze(-1).squeeze(-1)


def fusion_loss(d, w, lam_cons=0.1, lam_smooth=0.01):
    x, P = ci_fuse(d["x1"], d["P1"], d["x2"], d["P2"], w)
    n = x.size(-1)
    mse = ((x - d["xt"]) ** 2).mean()
    cons = torch.relu(nees_of(x, P, d["xt"]) - n).mean()      # penalize over-confidence
    smooth = (w[:, 1:] - w[:, :-1]).abs().mean()
    return mse + lam_cons * cons + lam_smooth * smooth


@torch.no_grad()
def metrics(d, w):
    x, P = ci_fuse(d["x1"], d["P1"], d["x2"], d["P2"], w)
    rmse = ((x - d["xt"]) ** 2).sum(-1).mean().sqrt().item()
    return rmse, nees_of(x, P, d["xt"]).mean().item()


# ---------------- features ----------------
def make_features(x1, P1, x2, P2):
    d1 = torch.diagonal(P1, dim1=-2, dim2=-1)
    d2 = torch.diagonal(P2, dim1=-2, dim2=-1)
    diff = (x1 - x2) / torch.sqrt(d1 + d2 + 1e-6)
    return torch.cat([torch.log(d1 + 1e-6), torch.log(d2 + 1e-6), diff], -1)


# ---------------- data ----------------
def make_synthetic(N, T, seed):
    """Two sensors, regime-switching noise, noisy reported covariances, shared error (-> CI is needed)."""
    g = torch.Generator().manual_seed(seed)
    n = 4
    base = torch.tensor([1.0, 0.5, 1.0, 0.5])
    xt = torch.cumsum(0.1 * torch.randn(N, T, n, generator=g), 1)

    def scale():
        seg = torch.randint(0, 2, (N, T // 16 + 1), generator=g)
        s = torch.where(seg == 1, torch.tensor(2.0), torch.tensor(0.5))
        return s.repeat_interleave(16, 1)[:, :T]

    s1, s2 = scale(), scale()
    shared = 0.3 * base * torch.randn(N, T, n, generator=g)
    x1 = xt + s1[..., None] * base * torch.randn(N, T, n, generator=g) + shared
    x2 = xt + s2[..., None] * base * torch.randn(N, T, n, generator=g) + shared
    # reported covariance = true scale corrupted per 16-step segment (sensor "mis-reports")
    def reported(s):
        m = torch.exp(0.5 * torch.randn(N, T // 16 + 1, generator=g)).repeat_interleave(16, 1)[:, :T]
        return s * m
    v1 = ((reported(s1)[..., None] * base) ** 2 + (0.3 * base) ** 2)
    v2 = ((reported(s2)[..., None] * base) ** 2 + (0.3 * base) ** 2)
    return dict(x1=x1, x2=x2, xt=xt, P1=torch.diag_embed(v1), P2=torch.diag_embed(v2))


def load_npz(path):
    z = np.load(path)
    return {k: torch.tensor(z[k], dtype=torch.float32)
            for k in ["x1", "x2", "xt" if "xt" in z else "x_true", "P1", "P2"]} | {}


def split(d, frac=0.8):
    N = d["x1"].size(0)
    k = int(N * frac)
    return ({a: v[:k] for a, v in d.items()}, {a: v[k:] for a, v in d.items()})


# ---------------- training ----------------
def train(model, tr, mu, sd, epochs, bs, lr, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    N = tr["x1"].size(0)
    steps = epochs * ((N + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            b = {k: v[idx] for k, v in tr.items()}
            feat = (make_features(b["x1"], b["P1"], b["x2"], b["P2"]) - mu) / sd
            loss = fusion_loss(b, model(feat))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--T", type=int, default=64)
    a = ap.parse_args()

    res = {}
    for s in range(a.seeds):
        d = load_npz(a.data) if a.data else make_synthetic(a.n, a.T, seed=s)
        tr, te = split(d)
        ftr = make_features(tr["x1"], tr["P1"], tr["x2"], tr["P2"])
        mu, sd = ftr.mean((0, 1)), ftr.std((0, 1)) + 1e-6
        fte = (make_features(te["x1"], te["P1"], te["x2"], te["P2"]) - mu) / sd
        in_dim = ftr.size(-1)

        trace1 = torch.diagonal(te["P1"], dim1=-2, dim2=-1).sum(-1)
        trace2 = torch.diagonal(te["P2"], dim1=-2, dim2=-1).sum(-1)
        out = {
            "fixed w=0.5": metrics(te, torch.full_like(trace1, 0.5)),
            "trace heuristic": metrics(te, (trace2 / (trace1 + trace2)).clamp(0.01, 0.99)),
        }
        lstm = train(LSTMOmega(in_dim), tr, mu, sd, a.epochs, a.bs, a.lr, s)
        tf = train(CausalOmegaTransformer(in_dim), tr, mu, sd, a.epochs, a.bs, a.lr, s)
        with torch.no_grad():
            out["LSTM (e2e loss)"] = metrics(te, lstm(fte))
            out["Transformer (e2e loss)"] = metrics(te, tf(fte))
        for k, v in out.items():
            res.setdefault(k, []).append(v)
        print(f"seed {s} done")

        if s == 0:
            os.makedirs("data/models", exist_ok=True)
            torch.save({"model": tf.state_dict(), "mu": mu, "sd": sd, "in_dim": in_dim},
                       "data/models/approach5_transformer.pt")

    print(f"\n{'method':<26}{'RMSE':>18}{'mean NEES (~4 ideal)':>26}")
    for k, v in res.items():
        r = np.array(v)
        print(f"{k:<26}{r[:,0].mean():>10.4f} ± {r[:,0].std():.4f}{r[:,1].mean():>14.2f} ± {r[:,1].std():.2f}")


if __name__ == "__main__":
    main()