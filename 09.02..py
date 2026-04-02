"""
Unsupervised "event neurons" on S&P 500 daily data:
TCN (masked modeling pretrain) -> novelty score -> latent prototypes -> LIF spike bank.

Requires:
  pip install torch numpy pandas scikit-learn matplotlib

Data:
  downloads ^SPX daily from Stooq: https://stooq.com/q/d/l/?s=^spx&i=d
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Data
# -----------------------------
def load_spx_from_stooq(csv_url="https://stooq.com/q/d/l/?s=^spx&i=d"):
    df = pd.read_csv(csv_url)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def make_features(df, vol_window=20):
    close = df["Close"].astype(float).values
    logp = np.log(close)
    ret = np.diff(logp, prepend=logp[0]).astype(np.float32)
    vol = pd.Series(ret).rolling(vol_window).std().fillna(0.0).values.astype(np.float32)
    X = np.stack([ret, vol], axis=1).astype(np.float32)  # (T, F=2)
    return X

def windowize(X, W=128, stride=1):
    xs, starts = [], []
    for s in range(0, len(X) - W, stride):
        xs.append(X[s:s+W])
        starts.append(s)
    return np.stack(xs), np.array(starts)

# -----------------------------
# TCN (causal)
# -----------------------------
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=dilation)
    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))

class TCNBlock(nn.Module):
    def __init__(self, ch, k, dilation, dropout=0.1):
        super().__init__()
        self.c1 = CausalConv1d(ch, ch, k, dilation)
        self.c2 = CausalConv1d(ch, ch, k, dilation)
        self.do = nn.Dropout(dropout)
        self.n1 = nn.LayerNorm(ch)
        self.n2 = nn.LayerNorm(ch)
    def forward(self, x):
        y = self.do(F.gelu(self.c1(x)))
        y = self.n1(y.transpose(1,2)).transpose(1,2)
        y = self.do(F.gelu(self.c2(y)))
        y = self.n2(y.transpose(1,2)).transpose(1,2)
        return x + y

class TCNEncoder(nn.Module):
    def __init__(self, in_ch=2, ch=64, levels=6, k=3, dropout=0.1):
        super().__init__()
        self.inproj = nn.Conv1d(in_ch, ch, 1)
        self.blocks = nn.Sequential(*[
            TCNBlock(ch, k, dilation=2**i, dropout=dropout) for i in range(levels)
        ])
        self.outnorm = nn.LayerNorm(ch)
    def forward(self, x):
        # x: (B,T,F)
        x = x.transpose(1,2)          # (B,F,T)
        h = self.inproj(x)            # (B,ch,T)
        h = self.blocks(h)            # (B,ch,T)
        h = self.outnorm(h.transpose(1,2))  # (B,T,ch)
        return h

class MaskedHead(nn.Module):
    def __init__(self, ch=64, out_ch=2):
        super().__init__()
        self.lin = nn.Linear(ch, out_ch)
    def forward(self, h):
        return self.lin(h)  # (B,T,F)

# -----------------------------
# LIF neuron bank (surrogate spiking)
# -----------------------------
class LIFBank(nn.Module):
    """
    Takes per-timestep latent h_t (B,T,C) -> currents (B,T,K) -> LIF -> spikes (B,T,K) + membrane V.
    Includes: optional refractory; we’ll add kWTA in training loop.
    """
    def __init__(self, in_dim, K, alpha=0.95, theta=1.0):
        super().__init__()
        self.K = K
        self.alpha = alpha
        self.theta = theta
        self.readin = nn.Linear(in_dim, K)  # currents
        # learned per-neuron gain/bias can help specialization
        self.gain = nn.Parameter(torch.ones(K))
        self.bias = nn.Parameter(torch.zeros(K))

    @staticmethod
    def spike_fn(v_minus_theta):
        # surrogate: hard spike forward, smooth backward (straight-through-ish)
        s = (v_minus_theta >= 0).float()
        # surrogate grad: fast sigmoid
        sg = torch.sigmoid(10.0 * v_minus_theta)
        return s + (sg - sg.detach())

    def forward(self, h, reset=True):
        # h: (B,T,C)
        I = self.readin(h) * self.gain + self.bias  # (B,T,K)
        B, T, K = I.shape
        V = torch.zeros((B, K), device=h.device)
        spikes = []
        Vs = []
        for t in range(T):
            V = self.alpha * V + I[:, t, :]
            s = self.spike_fn(V - self.theta)
            # reset by subtraction
            V = V - s * self.theta
            spikes.append(s)
            Vs.append(V)
        S = torch.stack(spikes, dim=1)  # (B,T,K)
        Vt = torch.stack(Vs, dim=1)     # (B,T,K)
        return S, Vt, I

# -----------------------------
# Utilities: losses for specialization
# -----------------------------
def decorrelation_loss(S):
    # S: (B,T,K) -> flatten time/batch -> (N,K)
    X = S.reshape(-1, S.shape[-1])
    X = X - X.mean(dim=0, keepdim=True)
    cov = (X.T @ X) / (X.shape[0] + 1e-6)
    off = cov - torch.diag(torch.diag(cov))
    return (off**2).mean()

def firing_rate_loss(S, target=0.002):
    # Encourage rare spiking (~0.2% of timesteps by default)
    rate = S.mean(dim=(0,1))  # (K,)
    return ((rate - target)**2).mean()

def k_winners_mask(I_t, k=1):
    # I_t: (B,K) -> mask (B,K) where only top-k per row are 1
    topk = torch.topk(I_t, k=k, dim=-1).indices
    mask = torch.zeros_like(I_t)
    mask.scatter_(1, topk, 1.0)
    return mask

# -----------------------------
# Main
# -----------------------------
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df = load_spx_from_stooq()
    X = make_features(df)

    # train/test split by time
    split = int(len(X) * 0.8)
    X_train_raw, X_test_raw = X[:split], X[split:]

    # standardize on train
    mu = X_train_raw.mean(axis=0, keepdims=True)
    sd = X_train_raw.std(axis=0, keepdims=True) + 1e-6
    X_train = (X_train_raw - mu) / sd
    X_test  = (X_test_raw  - mu) / sd

    W = 128
    stride = 2
    train_w, train_start = windowize(X_train, W=W, stride=stride)
    test_w,  test_start  = windowize(X_test,  W=W, stride=stride)

    train_tensor = torch.from_numpy(train_w).to(device)
    test_tensor  = torch.from_numpy(test_w).to(device)

    # -------------------------
    # Stage 1: masked modeling pretrain (TCN + head)
    # -------------------------
    enc = TCNEncoder(in_ch=2, ch=64, levels=6, k=3, dropout=0.1).to(device)
    head = MaskedHead(ch=64, out_ch=2).to(device)

    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=2e-3, weight_decay=1e-4)

    batch_size = 256
    mask_prob = 0.25
    iters = 1200  # increase for real run

    enc.train(); head.train()
    for it in range(iters):
        idx = torch.randint(0, train_tensor.shape[0], (batch_size,), device=device)
        x = train_tensor[idx]                         # (B,W,F)
        m = (torch.rand((batch_size, W), device=device) < mask_prob)
        xm = x.clone()
        xm[m] = 0.0
        h = enc(xm)
        xhat = head(h)

        m3 = m.unsqueeze(-1).expand_as(x)
        loss = F.mse_loss(xhat[m3], x[m3])

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
        opt.step()

        if (it+1) % 200 == 0:
            print(f"[pretrain] it={it+1}/{iters} loss={loss.item():.5f}")

    # -------------------------
    # Stage 2a: novelty score u_t from masked recon error
    # -------------------------
    enc.eval(); head.eval()
    train_len = len(X_train)
    u_sum = np.zeros(train_len, dtype=np.float32)
    u_cnt = np.zeros(train_len, dtype=np.float32)

    reps = 200  # increase for smoother u_t
    with torch.no_grad():
        for _ in range(reps):
            idx = torch.randint(0, train_tensor.shape[0], (batch_size,), device=device)
            x = train_tensor[idx]
            starts = train_start[idx.cpu().numpy()]  # (B,)
            m = (torch.rand((batch_size, W), device=device) < mask_prob)
            xm = x.clone()
            xm[m] = 0.0
            h = enc(xm)
            xhat = head(h)

            err = (torch.abs(xhat - x).mean(dim=-1) * m.float()).cpu().numpy()  # (B,W)
            m_np = m.cpu().numpy()

            abs_idx = starts[:, None] + np.arange(W)[None, :]
            flat_idx = abs_idx[m_np].ravel()
            flat_err = err[m_np].ravel()
            np.add.at(u_sum, flat_idx, flat_err)
            np.add.at(u_cnt, flat_idx, 1.0)

    u = u_sum / (u_cnt + 1e-6)
    valid = u_cnt > 0
    thr = np.quantile(u[valid], 0.99)  # "major event" = top 1% novelty (tunable)
    event_mask = (u >= thr) & valid
    event_times = np.where(event_mask)[0]
    print("major-event threshold:", float(thr), "| count:", len(event_times))

    # -------------------------
    # Stage 2b: event descriptors + prototypes (unsupervised categories)
    # -------------------------
    # Descriptor = mean latent over last L steps of each window (window-end time)
    L = 20
    with torch.no_grad():
        H = enc(train_tensor).cpu().numpy()  # (n_win,W,C)
    desc = H[:, -L:, :].mean(axis=1)         # (n_win,C)
    time_list = train_start + (W - 1)        # absolute train time for each descriptor

    # prototypes: cluster descriptor space
    K = 8
    kmeans = KMeans(n_clusters=K, n_init=20, random_state=0).fit(desc)
    proto = torch.from_numpy(kmeans.cluster_centers_).float().to(device)  # (K,C)

    # -------------------------
    # Stage 3: LIF bank learns to spike & specialize (no labels)
    # -------------------------
    lif = LIFBank(in_dim=desc.shape[1], K=K, alpha=0.95, theta=1.0).to(device)
    opt2 = torch.optim.AdamW(lif.parameters(), lr=2e-3, weight_decay=1e-4)

    # training settings
    epochs = 10
    kwta_k = 1
    lam_rate = 5.0
    lam_decorr = 2.0
    lam_align = 1.0

    # We'll train using windows; define per-window target prototype:
    # pick the closest prototype to its descriptor (still unsupervised).
    desc_t = torch.from_numpy(desc).float().to(device)  # (n_win,C)
    with torch.no_grad():
        d2 = torch.cdist(desc_t, proto)                 # (n_win,K)
        y = torch.argmin(d2, dim=1)                     # (n_win,) pseudo "type"

    # We also want spikes to correlate with novelty:
    # treat a window as "novel" if its end time is in top-quantile novelty.
    end_u = u[time_list]
    novel_thr = np.quantile(end_u, 0.90)
    novel_flag = torch.from_numpy((end_u >= novel_thr).astype(np.float32)).to(device)

    enc.eval()  # freeze encoder for stability
    for ep in range(epochs):
        lif.train()
        perm = torch.randperm(train_tensor.shape[0], device=device)
        total = 0.0
        for j in range(0, len(perm), 256):
            b = perm[j:j+256]
            x = train_tensor[b]
            with torch.no_grad():
                h = enc(x)                         # (B,W,C)

            S, Vt, I = lif(h)                       # (B,W,K)
            # kWTA competition applied on currents per timestep
            mask = k_winners_mask(I.reshape(-1, K), k=kwta_k).reshape_as(I)
            S = S * mask

            # Align: when novel, encourage spike mass to select its prototype neuron
            # Use window-level spike count as "assignment"
            spike_count = S.sum(dim=1)              # (B,K)
            # convert to logits and train against prototype id y[b]
            logits = spike_count / (spike_count.std(dim=1, keepdim=True) + 1e-6)

            # only train align on "novel windows" so neurons represent event-types
            nf = novel_flag[b].unsqueeze(1)         # (B,1)
            align_loss = F.cross_entropy(logits, y[b], reduction="none")
            align_loss = (align_loss * nf.squeeze(1)).mean()

            rate_loss = firing_rate_loss(S, target=0.002)
            dec_loss = decorrelation_loss(S)

            loss = lam_align * align_loss + lam_rate * rate_loss + lam_decorr * dec_loss

            opt2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lif.parameters(), 1.0)
            opt2.step()

            total += float(loss.detach().cpu())

        print(f"[LIF] ep={ep+1}/{epochs} loss={total:.3f}  "
              f"(align={float(align_loss.detach().cpu()):.3f}, "
              f"rate={float(rate_loss.detach().cpu()):.3f}, "
              f"decorr={float(dec_loss.detach().cpu()):.3f})")

    # -------------------------
    # Simple visualization on test: novelty + spikes
    # -------------------------
    lif.eval()
    with torch.no_grad():
        # compute latents + spikes for test windows
        Htest = enc(test_tensor)         # (B,W,C)
        S, Vt, I = lif(Htest)            # (B,W,K)
        # take spikes at window-end
        Send = S[:, -1, :].cpu().numpy()  # (n_test,K)

    # build test novelty for reference (quick pass)
    # (in a real run you’d compute u_t on test similarly; here we just plot spikes)
    dates_test_end = df["Date"].iloc[split + test_start + (W-1)].values

    plt.figure(figsize=(12,4))
    plt.title("Test: LIF end-of-window spikes per neuron (binary dots)")
    for k in range(K):
        t_idx = np.where(Send[:,k] > 0.5)[0]
        plt.scatter(dates_test_end[t_idx], np.full_like(t_idx, k), s=10)
    plt.xlabel("Date"); plt.ylabel("Neuron id")
    plt.tight_layout()
    plt.show()

    print("Done. Next: explainability = plot input, I(t), V(t) around spike times.")

if __name__ == "__main__":
    main()
