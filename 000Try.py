import math

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def make_xy(x, r, L, event_thr=-0.02):
    X = x.unfold(0, L, 1)                         # [N,2,L]
    Y = (r < event_thr).float().unfold(0, L, 1)  # [N,L]
    return X, Y

def market_data_loaders(path="market_data.xlsx", L=52, B=64, split=0.8, event_thr=-0.02):
    df = pd.read_excel(path, sheet_name="US")
    df = df.rename(columns={df.columns[0]: "Date"}).sort_values("Date")

    x = torch.tensor(df[["CF", "ED"]].iloc[1:].values, dtype=torch.float32)   # [T,2]
    r = torch.tensor(df["_MKT"].pct_change().dropna().values, dtype=torch.float32)

    cut = int(split * len(r))
    Xtr, Ytr = make_xy(x[:cut], r[:cut], L, event_thr)
    Xte, Yte = make_xy(x[cut:], r[cut:], L, event_thr)

    return (
        DataLoader(TensorDataset(Xtr, Ytr), batch_size=B, shuffle=True),
        DataLoader(TensorDataset(Xte, Yte), batch_size=B, shuffle=False),
        r
    )

def dataset_sample(loader, idx, device="cpu"):
    X, Y = loader.dataset.tensors
    return X[idx:idx+1].to(device), Y[idx].cpu()

def tau_values(layer):
    return [round(t, 3) for t in layer.tau().detach().cpu().tolist()]

def spike_labels(n_channels):
    half = n_channels // 2
    return ["target"] + [f"CF {i}" for i in range(half)] + [f"ED {i}" for i in range(half)]

class CausalConvBank(nn.Module):
    def __init__(self, n_filters, kernel_size):
        super().__init__()
        self.k = kernel_size
        self.conv = nn.Conv1d(1, n_filters, kernel_size, bias=False)

    def forward(self, x):
        w = self.conv.weight
        w = w / w.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8).view(-1, 1, 1)
        return F.conv1d(F.pad(x, (self.k - 1, 0)), w)  # [B,F,L]

class LIFLayer(nn.Module):
    def __init__(self, n_filters, tau=3.0, threshold=0.25, beta=15.0, dt=1.0):
        super().__init__()
        self.dt = float(dt)
        self.register_buffer("thr", torch.tensor(float(threshold), dtype=torch.float32))
        self.beta = float(beta)
        raw0 = math.log(math.expm1(float(tau)))
        self.raw_tau = nn.Parameter(torch.full((n_filters,), raw0, dtype=torch.float32))

    def tau(self):
        return F.softplus(self.raw_tau) + 1e-4

    def forward(self, I):                                # I: [B,F,L]
        B, F_, L = I.shape
        v = I.new_zeros(B, F_)
        v_hist, z_hist, s_hist = [], [], []

        alpha = torch.exp(-self.dt / self.tau()).to(I.dtype).view(1, F_)
        thr, beta = self.thr.to(I.dtype), self.beta

        for t in range(L):
            v_pre = alpha * v + (1 - alpha) * I[:, :, t]
            z = beta * (v_pre - thr)
            p = torch.sigmoid(z)
            s_hard = (v_pre >= thr).to(I.dtype)
            s = s_hard + p - p.detach()                 # hard forward, surrogate backward
            v = v_pre * (1 - s_hard)
            v_hist.append(v_pre)
            z_hist.append(z)
            s_hist.append(s)

        return torch.stack(v_hist, -1), torch.stack(z_hist, -1), torch.stack(s_hist, -1)

class SNNEventModel(nn.Module):
    def __init__(
        self,
        n_filters=8,
        kernel_size=8,
        tau=3.0,
        threshold=0.25,
        beta=15.0,
        kp=10.0,
        gain_min=0.25,
        gain_max=4.0,
    ):
        super().__init__()


        self.bank_a = CausalConvBank(n_filters, kernel_size)
        self.bank_b = CausalConvBank(n_filters, kernel_size)
        self.lif_a = LIFLayer(n_filters, tau=tau, threshold=threshold, beta=beta)
        self.lif_b = LIFLayer(n_filters, tau=tau, threshold=threshold, beta=beta)

        self.kp = float(kp)
        self.gain_min, self.gain_max = float(gain_min), float(gain_max)
        self.register_buffer("gain", torch.tensor(1.0))

    def forward(self, x):                                # x: [B,2,L]
        Ia = self.gain * self.bank_a(x[:, 0:1])
        Ib = self.gain * self.bank_b(x[:, 1:2])

        va, za, sa = self.lif_a(Ia)
        vb, zb, sb = self.lif_b(Ib)

        v = torch.cat([va, vb], dim=1)                   # [B,16,L]
        z = torch.cat([za, zb], dim=1)                   # [B,16,L]
        s = torch.cat([sa, sb], dim=1)                   # [B,16,L]

        return {"v": v, "z": z, "s": s, "logits": z.amax(dim=1)}  # [B,L]

    @torch.no_grad()
    def p_update_epoch(self, spike_sum, target_sum, spike_sites):
        err = target_sum / spike_sites - spike_sum / spike_sites
        self.gain.add_(self.kp * err).clamp_(self.gain_min, self.gain_max)
        return err

def event_loss(s, y, lambda_spk=1.0, lambda_homeo=0.05, homeo_beta=8.0):
    B, F_, T = s.shape
    L = float(T)
    t = torch.arange(T, device=s.device, dtype=s.dtype)
    D = (t[:, None] - t[None, :]).abs()                  # [T,T]

    p = s.amax(dim=1)                                    # [B,T] any spike over filters
    dt = (D[None] + (L - D)[None] * (1 - p[:, None, :])).min(dim=2).values
    Lt = (y * dt).sum() / y.sum().clamp_min(1.0)

    ds = (D[None, None] + (L - D)[None, None] * (1 - y[:, None, None, :])).min(dim=3).values
    Ls = (s * ds).sum() / s.sum().clamp_min(1.0)

    df = (D[None, None] + (L - D)[None, None] * (1 - s[:, :, None, :])).min(dim=3).values
    matched = (y * (dt < L).to(y.dtype)).unsqueeze(1)
    resp = torch.softmax(-homeo_beta * df, dim=1)
    cnt = (resp * matched).sum(dim=(0, 2))
    q = cnt / cnt.sum().clamp_min(1e-6)
    Lh = ((q - 1.0 / F_) ** 2).mean() * (cnt.sum() > 0).to(s.dtype)

    return Lt + lambda_spk * Ls + lambda_homeo * Lh

@torch.no_grad()
def evaluate(model, loader, device, lambda_spk=1.0, lambda_homeo=0.05, homeo_beta=8.0):
    model.eval()
    total, n, spk, tgt = 0.0, 0, 0.0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb)
        total += event_loss(out["s"], yb, lambda_spk, lambda_homeo, homeo_beta).item() * xb.size(0)
        n += xb.size(0)
        spk += out["s"].sum().item()
        tgt += yb.sum().item()
    return total / n, spk, tgt

def fit(model, train_loader, test_loader, epochs=20, lr=1e-3, device="cpu",
        lambda_spk=1.0, lambda_homeo=0.05, homeo_beta=8.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    hist = {"train": [], "test": [], "gain": []}

    for ep in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        train_spk, train_tgt, train_sites = 0.0, 0.0, 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = event_loss(out["s"], yb, lambda_spk, lambda_homeo, homeo_beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * xb.size(0)
            n += xb.size(0)

            train_spk += out["s"].sum().item()
            train_tgt += yb.sum().item()
            train_sites += out["s"].numel()

        err = model.p_update_epoch(train_spk, train_tgt, train_sites)

        tr_loss = total / n
        te_loss, te_spk, te_tgt = evaluate(model, test_loader, device, lambda_spk, lambda_homeo, homeo_beta)

        hist["train"].append(tr_loss)
        hist["test"].append(te_loss)
        hist["gain"].append(model.gain.item())

        print(
            f"Epoch {ep:03d} | train {tr_loss:.4f} | test {te_loss:.4f} | "
            f"gain {model.gain.item():.3f} | train spikes {int(train_spk)} | train targets {int(train_tgt)} | "
            f"test spikes {int(te_spk)} | test targets {int(te_tgt)} | err {err:.5f}"
        )

    return model, hist

def plot_history(hist):
    plt.figure(figsize=(8,4))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["test"], label="test")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss history")
    plt.legend(); plt.tight_layout(); plt.show()

@torch.no_grad()
def plot_sequence(model, loader, device="cpu", idx=0):
    model.eval()
    x, y = dataset_sample(loader, idx, device)
    out = model(x)
    s = out["s"][0].cpu()

    labels = spike_labels(s.size(0))
    M = torch.cat([y.unsqueeze(0), s], dim=0).numpy()

    plt.figure(figsize=(12, 4.5))
    plt.imshow(M, aspect="auto", interpolation="nearest", cmap="Greys")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("time step in sequence")
    plt.title(f"idx={idx} | gain={model.gain.item():.3f} | targets={int(y.sum())} | spikes={int(s.sum())}")
    plt.tight_layout()
    plt.show()

    print(f"idx {idx:03d} | targets {int(y.sum().item())} | spikes {int(s.sum().item())}")
    print("taus CF:", tau_values(model.lif_a))
    print("taus ED:", tau_values(model.lif_b))
    print("Target positions:", torch.where(y > 0)[0].tolist())
    for i in range(s.size(0)):
        print(f"{labels[i+1]} spikes at:", torch.where(s[i] > 0)[0].tolist())

@torch.no_grad()
def plot_lif_traces(model, loader, device="cpu", idx=0, ncols=2):
    model.eval()
    x, y = dataset_sample(loader, idx, device)
    v = model(x)["v"][0].cpu()
    x = x[0].cpu()
    n, h = v.size(0), v.size(0) // 2
    fig, axs = plt.subplots(math.ceil(n / ncols), ncols, figsize=(12, 2.4 * math.ceil(n / ncols)), sharex=True)
    axs = axs.flatten() if hasattr(axs, "flatten") else [axs]
    tgt, thr = torch.where(y > 0)[0].tolist(), model.lif_a.thr.item()

    for i, ax in enumerate(axs[:n]):
        ts = x[0] if i < h else x[1]
        ax.plot(ts, lw=1)
        for t in tgt: ax.axvline(t, color="r", lw=.8, alpha=.25)
        ax.set_title(f"{'CF' if i < h else 'ED'} {i % h}", fontsize=10)
        ax2 = ax.twinx()
        ax2.plot(v[i], "--", lw=1)
        ax2.axhline(thr, ls=":", lw=1, color="k")
        ax2.set_ylim(0, max(0.3, thr * 1.25, float(v[i].max()) * 1.05))

    for ax in axs[n:]: ax.axis("off")
    plt.tight_layout(); plt.show()

@torch.no_grad()
def sample_spike_stats(model, loader, device="cpu", n=20):
    model.eval()
    s_counts, y_counts = [], []
    for xb, yb in loader:
        out = model(xb.to(device))
        s_counts.append(out["s"].sum(dim=(1, 2)).cpu())
        y_counts.append(yb.sum(dim=1).cpu())
    s_counts, y_counts = torch.cat(s_counts), torch.cat(y_counts)

    print(f"Mean spikes/sample:  {s_counts.float().mean().item():.2f}")
    print(f"Mean targets/sample: {y_counts.float().mean().item():.2f}")
    print(f"Zero-spike samples:  {int((s_counts == 0).sum())}/{len(s_counts)}")
    print(f"Zero-target samples: {int((y_counts == 0).sum())}/{len(y_counts)}")
    print("--- first samples ---")
    for i in range(min(n, len(s_counts))):
        print(f"idx {i:03d} | targets {int(y_counts[i].item())} | spikes {int(s_counts[i].item())}")
    return s_counts, y_counts

def top_target_indices(loader, n=6):
    Y = loader.dataset.tensors[1]
    return torch.topk(Y.sum(dim=1), k=min(n, len(Y))).indices.tolist()

@torch.no_grad()
def plot_sequences(model, loader, device="cpu", idxs=(0,1,2,3)):
    for idx in idxs:
        plot_sequence(model, loader, device=device, idx=int(idx))

device = "cuda" if torch.cuda.is_available() else "cpu"
train_loader, test_loader, returns = market_data_loaders(
    path="market_data.xlsx", L=52, B=64, split=0.8, event_thr=-0.02
)
model = SNNEventModel(
    n_filters=8, kernel_size=8, tau=3.0, threshold=0.25,
    beta=15.0, kp=10.0, gain_min=0.25, gain_max=4.0,
)
model, hist = fit(
    model, train_loader, test_loader,
    epochs=100, lr=1e-3, device=device,
    lambda_spk=1.0, lambda_homeo=0.05, homeo_beta=8.0,
)

plot_history(hist)
print("Learned taus CF:", tau_values(model.lif_a))
print("Learned taus ED:", tau_values(model.lif_b))
sample_spike_stats(model, test_loader, device=device, n=12)
plot_sequences(model, test_loader, device=device, idxs=range(4))
plot_sequences(model, test_loader, device=device, idxs=top_target_indices(test_loader, n=4))
plot_lif_traces(model, test_loader, device=device, idx=top_target_indices(test_loader, n=1)[0], ncols=2)