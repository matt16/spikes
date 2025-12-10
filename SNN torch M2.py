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
        Returns spike raster and latency of first spike.
        """
        B,L = I.shape
        v = torch.zeros(B)
        spikes = torch.zeros(B, L)
        hard_spikes = torch.zeros(B, L)  # binary spike raster
        k = 10.0
        for t in range(L):
            dv = -v/self.tau + I[:,t]
            v = v + dv
            s = torch.sigmoid(k*(v - self.th))  # soft surrogate spike
            spikes[:,t] = s
            
            # Hard spike: threshold the surrogate (detached for gradient flow)
            hard_spike = (v.detach() > self.th).float()
            hard_spikes[:,t] = hard_spike
            
            v = v * (1 - s.detach())  # reset using stop-grad from surrogate

        # Find first spike time (hard latency)
        first_spikes = (hard_spikes > 0.5).int()
        first_spike_times = torch.full((B,), L, dtype=torch.long)
        for b in range(B):
            spike_times = torch.where(first_spikes[b] > 0)[0]
            if len(spike_times) > 0:
                first_spike_times[b] = spike_times[0]
        
        hard_latency = first_spike_times
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

        return (da, ida, vala, sa, la_hard, la_soft), (db, idb, valb, sb, lb_hard, lb_soft)


# ============================================================
# 5) Latency-matching loss
# ============================================================
def latency_loss(la, lb):
    return torch.mean(torch.abs(la.float() - lb.float()))


# ============================================================
# 6) Visualization utilities
# ============================================================
def plot_spike_rasters_with_winners(time, spikes_a, winner_idx_a, spikes_b, winner_idx_b, threshold=0.5):
    """
    Plot spike rasters from both channels side-by-side with winning dendrite index overlaid.
    Only shows winner index when an actual spike occurs (spike > threshold).
    
    spikes_a, spikes_b: [L] spike rasters from channels A and B
    winner_idx_a, winner_idx_b: [L] winning dendrite index at each time
    threshold: spike threshold for determining if a spike occurred
    """
    L = spikes_a.shape[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Channel A (noisy)
    ax = axes[0]
    ax.plot(time, spikes_a, color='red', lw=2, alpha=0.7, label='Spikes')
    ax.set_ylabel("Spike strength", color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_title("Channel A (Noisy)")
    ax.grid(True, alpha=0.3)
    
    # Overlay winner index only when spikes occur
    ax2 = ax.twinx()
    spike_mask_a = spikes_a > threshold
    winner_on_spike_a = torch.where(spike_mask_a, winner_idx_a.float(), torch.nan)
    ax2.scatter(time[spike_mask_a], winner_on_spike_a[spike_mask_a], 
               color='blue', s=50, alpha=0.8, label='Winner')
    ax2.set_ylabel("Winning dendrite", color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(-0.5, winner_idx_a.max().item() + 1)
    
    # Channel B (clean)
    ax = axes[1]
    ax.plot(time, spikes_b, color='orange', lw=2, alpha=0.7, label='Spikes')
    ax.set_ylabel("Spike strength", color='orange')
    ax.tick_params(axis='y', labelcolor='orange')
    ax.set_title("Channel B (Clean)")
    ax.set_xlabel("Real time")
    ax.grid(True, alpha=0.3)
    
    # Overlay winner index only when spikes occur
    ax2 = ax.twinx()
    spike_mask_b = spikes_b > threshold
    winner_on_spike_b = torch.where(spike_mask_b, winner_idx_b.float(), torch.nan)
    ax2.scatter(time[spike_mask_b], winner_on_spike_b[spike_mask_b], 
               color='green', s=50, alpha=0.8, label='Winner')
    ax2.set_ylabel("Winning dendrite", color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim(-0.5, winner_idx_b.max().item() + 1)
    
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
# 7) Generate synthetic self-supervised pair
# ============================================================
t = torch.linspace(0, 4*3.14, 400).unsqueeze(0)
clean = torch.sin(t)
noisy = clean + 0.2*torch.randn_like(clean)

# ============================================================
# 8) Training loop (expanded + commented)
# ============================================================
model = LatencyPredictor(T=20, K1=32, K2=12)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []

for step in range(2000):
    # ---- forward pass ----
    (da, ida, vala, sa, la_hard, la_soft), (db, idb, valb, sb, lb_hard, lb_soft) = model(noisy, clean)

    # ---- compute latency-based self-supervised loss ----
    loss = latency_loss(la_soft, lb_soft)

    # ---- optimize ----
    opt.zero_grad()
    loss.backward()
    opt.step()

    losses.append(float(loss))

    if step % 200 == 0:
        print(step, float(loss))

# ============================================================
# 9) Visualization usage
# ============================================================
# Training progress plot
plt.figure(figsize=(10, 4))
plt.plot(losses, lw=1)
plt.xlabel("Training step")
plt.ylabel("Loss (latency mismatch)")
plt.title("Training Progress")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Example after training:
T_window = 20  # matches T in LatencyPredictor
L_vis = da.shape[1]  # segment length after Hankel
time = t[0, T_window-1:T_window-1+L_vis]  # match Hankel-aligned time window

# Plot spike rasters from both channels with winning dendrite overlaid
plot_spike_rasters_with_winners(time.detach(), sa[0].detach(), ida[0].detach(),
                                sb[0].detach(), idb[0].detach())
plot_dendrite_io(time.detach(), noisy[0, T_window-1:T_window-1+L_vis].detach(), da[0].detach())

