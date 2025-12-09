import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# 1) Strict backward-looking Hankel (sliding window)
# ============================================================
def hankel_backward(x, T):
    """
    x: [B, L]
    Return windows ending at each time step:
    H[b, t] = x[b, t-T : t]
    Shape: [B, L-T+1, T]
    """
    B,L = x.shape
    xs = []
    for i in range(T, L+1):
        xs.append(x[:, i-T:i])
    return torch.stack(xs, dim=1)


# ============================================================
# 1a) Helper
# ============================================================
def soft_lat(spikes):
    # spikes: [B, L] with surrogate sigmoid spikes
    L = spikes.shape[1]
    time = torch.arange(L, device=spikes.device, dtype=spikes.dtype)
    w = spikes / (spikes.sum(dim=1, keepdim=True) + 1e-6)
    return (w * time).sum(dim=1)


# ============================================================
# 2) Surrogate LIF ODE + latency extraction
# ============================================================
class LIF(nn.Module):
    def __init__(self, tau=20.0, threshold=1.0, k=10.0):
        super().__init__()
        self.tau, self.th, self.k = tau, threshold, k

    def forward(self, I):
        B, L = I.shape
        v = torch.zeros(B, device=I.device, dtype=I.dtype)
        spikes = torch.zeros(B, L, device=I.device, dtype=I.dtype)

        for t in range(L):
            v = v + (-v/self.tau + I[:, t])

            s_soft = torch.sigmoid(self.k * (v - self.th))     # surrogate
            s_hard = (v >= self.th).float()                    # hard spike
            s = s_hard.detach() - s_soft.detach() + s_soft    # STE

            spikes[:, t] = s
            v = v * (1 - s_hard.detach())                      # hard reset

        hard_latency = spikes.argmax(dim=1)
        soft_latency = soft_lat(spikes)

        return spikes, hard_latency, soft_latency



# ============================================================
# 3) Deep dendrites with SOFTMAX surrogate competition
# ============================================================
class DeepDendrites(nn.Module):
    def __init__(self, T, K1=16, K2=8):
        super().__init__()
        self.W1 = nn.Linear(T, K1)
        self.W2 = nn.Linear(K1, K2)

    def forward(self, H):
        # TODO: upgrade (e.g. Conv1d)
        d1 = torch.relu(self.W1(H))
        d2 = torch.relu(self.W2(d1))          # [B, L, K2]

        # --- Hard/Soft Winner (STE) ---
        soft_w = torch.softmax(d2, dim=-1)   # surrogate (backward)
        hard_idx = d2.argmax(dim=-1)         # hard winner (forward)
        hard_w = F.one_hot(hard_idx, d2.shape[-1]).float()

        w = hard_w.detach() - soft_w.detach() + soft_w
        soma_in = (w * d2).sum(dim=-1)       # [B, L]

        vals = (hard_w * d2).sum(dim=-1)     # echter Winner-Wert

        return d2, hard_idx, vals, soma_in



# ============================================================
# 4) Full self-supervised latency predictor
# ============================================================
class LatencyPredictor(nn.Module):
    """
    Two channels (A,B) supervise each other:
    Loss = |latA - latB|.
    """
    def __init__(self, T, K1=16, K2=8):
        super().__init__()
        self.encA = DeepDendrites(T, K1, K2)
        self.encB = DeepDendrites(T, K1, K2)
        self.lifA = LIF()
        self.lifB = LIF()
        self.T = T

    def forward(self, xa, xb):
        Ha = hankel_backward(xa, self.T)
        Hb = hankel_backward(xb, self.T)
        da, ida, vala, Ia = self.encA(Ha)
        db, idb, valb, Ib = self.encB(Hb)
        sa, la_hard, la_soft = self.lifA(Ia)
        sb, lb_hard, lb_soft = self.lifB(Ib)

        return (da, ida, vala, la_hard, la_soft), (db, idb, valb, lb_hard, lb_soft)


# ============================================================
# 5) Latency-matching loss
# ============================================================
def latency_loss(la, lb):
    return torch.mean(torch.abs(la.float() - lb.float()))


# ============================================================
# 6) Visualization utilities
# ============================================================
def plot_winners(time, winner_idx):
    time_np   = time.detach().cpu().numpy()
    winner_np = winner_idx.detach().cpu().numpy()

    plt.figure(figsize=(8,3))
    plt.plot(time_np, winner_np, lw=1)
    plt.title("Winning dendrite over time D")
    plt.xlabel("Real time")
    plt.ylabel("Dendrite index")
    plt.tight_layout()
    plt.show()


def plot_dendrite_io(time, x, dendritic_outputs):
    """
    Per-dendrite visualization:
    Left column: input segment
    Right column: dendritic filtered output
    """
    L, K2 = dendritic_outputs.shape

    # ✅ ALLES matplotlib-sicher machen
    time_np = time.detach().cpu().numpy()
    x_np    = x.detach().cpu().numpy()
    dend_np = dendritic_outputs.detach().cpu().numpy()

    fig, axes = plt.subplots(K2, 2, figsize=(10, 2*K2))
    for k in range(K2):
        axes[k,0].plot(time_np, x_np, color='red')
        axes[k,0].set_title(f"D{k} input")

        axes[k,1].plot(time_np, dend_np[:,k], color='blue')
        axes[k,1].set_title(f"D{k} output")

    plt.tight_layout()
    plt.show()

def plot_dendrites_with_winners(x, da, db, ida, idb, title="Per-dendrite comparison + Winners"):
    """
    x   : [L]        -> Input-Signal
    da  : [L, K2]    -> Dendriten noisy
    db  : [L, K2]    -> Dendriten clean
    ida : [L]        -> Winner-Index noisy
    idb : [L]        -> Winner-Index clean
    """

    L, K2 = da.shape
    samples = torch.arange(L).detach().cpu().numpy()

    x_np   = x.detach().cpu().numpy()
    da_np  = da.detach().cpu().numpy()
    db_np  = db.detach().cpu().numpy()
    ida_np = ida.detach().cpu().numpy()
    idb_np = idb.detach().cpu().numpy()

    # --- Grid + 1 Zusatzplot unten ---
    fig = plt.figure(figsize=(16, 4*K2 + 4))
    gs  = fig.add_gridspec(K2 + 1, 3, height_ratios=[1]*K2 + [0.7])

    # --- 3×K2 Grid ---
    for k in range(K2):

        # Input
        ax = fig.add_subplot(gs[k, 0])
        ax.plot(samples, x_np, color="red")
        ax.set_title(f"D{k} Input")
        ax.set_xlabel("Sample")

        # Dendrite A (noisy)
        ax = fig.add_subplot(gs[k, 1])
        ax.plot(samples, da_np[:, k], color="blue")
        ax.set_title(f"D{k} A (noisy)")
        ax.set_xlabel("Sample")

        # Dendrite B (clean)
        ax = fig.add_subplot(gs[k, 2])
        ax.plot(samples, db_np[:, k], color="purple")
        ax.set_title(f"D{k} B (clean)")
        ax.set_xlabel("Sample")

    # --- Winner Plot ganz unten ---
    ax = fig.add_subplot(gs[-1, :])  # ganze letzte Zeile über alle Spalten

    ax.plot(samples, ida_np, label="Winner noisy (ida)", color="blue")
    ax.plot(samples, idb_np, label="Winner clean (idb)", color="purple", alpha=0.6)

    ax.set_title("Dendrite Winners Over Time")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Winner index")
    ax.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

loss_history = []




# ============================================================
# 7) Training loop (expanded + commented)
# ============================================================
train_steps = 1001
loss_history = []

model = LatencyPredictor(T=20, K1=10, K2=3)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# ---- generate synthetic self-supervised pair ----
t = torch.linspace(0, 4 * 3.14, 400).unsqueeze(0)
clean = torch.sin(t)
noisy = clean + 0.2 * torch.randn_like(clean)

for step in range(train_steps):



    # ---- forward pass ----
    (da, ida, vala, la_hard, la_soft), (db, idb, valb, lb_hard, lb_soft) = model(noisy, clean)

    # ---- compute latency-based self-supervised loss ----
    loss = latency_loss(la_soft, lb_soft)
    loss_history.append(float(loss))

    # ---- optimize ----
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 200 == 0:
        print(step, float(loss))

# ============================================================
# 8) Visualization usage
# ============================================================
# Example after training:
L_vis = vala.shape[1]    # segment length after Hankel
time = t[0, -L_vis:]     # match Hankel-aligned end of time

plot_winners(time, ida[0])
plot_dendrite_io(time, noisy[0, -L_vis:], da[0])
# === Daten vorbereiten ===
x_plot  = noisy[0, -da.shape[1]:]
da_plot = da[0]
db_plot = db[0]
ida_plot = ida[0]
idb_plot = idb[0]

plot_dendrites_with_winners(
    x_plot,
    da_plot,
    db_plot,
    ida_plot,
    idb_plot,
    title="Dendrites + Winner Timeline"
)

#===================loss history===========
plt.figure(figsize=(6,4))
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Latency-Matching Loss Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
