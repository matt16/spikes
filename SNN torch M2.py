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
    def __init__(self, tau=20.0, threshold=1.0):
        super().__init__()
        self.tau = tau
        self.th = threshold

    def forward(self, I):
        """
        I: [B, L]
        Euler ODE: dv = dt*(-v/tau + I)
        Surrogate spike = sigmoid(k*(v-th))
        Reset v ← v*(1 - stopgrad(spike)).
        """
        B,L = I.shape
        v = torch.zeros(B)
        spikes = torch.zeros(B, L)
        k = 10.0
        for t in range(L):
            dv = -v/self.tau + I[:,t]
            v = v + dv
            s = torch.sigmoid(k*(v - self.th))
            spikes[:,t] = s
            v = v * (1 - s.detach())  # reset using stop-grad

        hard_latency = spikes.argmax(dim=1)   # int, no-grad
        soft_latency = soft_lat(spikes)   # float, differentiable
        return spikes, hard_latency, soft_latency


# ============================================================
# 3) Deep dendrites with SOFTMAX surrogate competition
# ============================================================
class DeepDendrites(nn.Module):
    """
    Hard max(d2) is NOT differentiable.
    → replaced with softmax-weighted dendritic sum (surrogate).
    Still returns argmax for interpretability.
    """
    def __init__(self, T, K1=16, K2=8):
        super().__init__()
        self.W1 = nn.Linear(T, K1)
        self.W2 = nn.Linear(K1, K2)

    def forward(self, H):
        """
        H: [B, L, T]
        d1: [B, L, K1]
        d2: [B, L, K2]
        soft_w: [B, L, K2] softmax over dendrites
        soma_input: softmax-weighted sum over dendrites
        """
        d1 = torch.relu(self.W1(H))
        d2 = torch.relu(self.W2(d1))

        # surrogate competition
        soft_w = torch.softmax(d2, dim=-1)  # differentiable
        soma_in = (soft_w * d2).sum(dim=-1) # [B, L]

        # still return discrete winner for diagnostics
        vals, idx = d2.max(dim=-1)
        return d2, idx, vals, soma_in


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
    plt.figure(figsize=(8,3))
    plt.plot(time, winner_idx, lw=1)
    plt.title("Winning dendrite over time")
    plt.xlabel("Real time")
    plt.ylabel("Dendrite index")
    plt.tight_layout()

def plot_dendrite_io(time, x, dendritic_outputs):
    """
    Per-dendrite visualization:
    Left column: input segment
    Right column: dendritic filtered output
    """
    L, K2 = dendritic_outputs.shape
    fig, axes = plt.subplots(K2, 2, figsize=(10, 2*K2))
    for k in range(K2):
        axes[k,0].plot(time, x, color='red')
        axes[k,0].set_title(f"D{k} input")

        axes[k,1].plot(time, dendritic_outputs[:,k], color='blue')
        axes[k,1].set_title(f"D{k} output")

    plt.tight_layout()


# ============================================================
# 7) Training loop (expanded + commented)
# ============================================================
model = LatencyPredictor(T=20, K1=32, K2=12)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(2000):

    # ---- generate synthetic self-supervised pair ----
    t = torch.linspace(0, 4*3.14, 400).unsqueeze(0)
    clean = torch.sin(t)
    noisy = clean + 0.2*torch.randn_like(clean)

    # ---- forward pass ----
    (da, ida, vala, la_hard, la_soft), (db, idb, valb, lb_hard, lb_soft) = model(noisy, clean)

    # ---- compute latency-based self-supervised loss ----
    loss = latency_loss(la_soft, lb_soft)

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

