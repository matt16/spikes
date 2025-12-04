
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Strict backward-looking Hankel (sliding window)
# ------------------------------------------------------------
def hankel_backward(x, T):
    """
    x: [B, L] time series
    returns H: [B, L-T+1, T] windows ending at each time step
    """
    B,L = x.shape
    xs = []
    for i in range(T, L+1):
        xs.append(x[:, i-T:i])
    return torch.stack(xs, dim=1)

# ------------------------------------------------------------
# 2. Surrogate LIF ODE + latency extraction
# ------------------------------------------------------------
class LIF(nn.Module):
    def __init__(self, tau=20.0, threshold=1.0):
        super().__init__()
        self.tau = tau
        self.th = threshold

    def forward(self, I):
        """
        I: [B, L] dendritic max-current over time
        Uses simple Euler ODE: dv/dt = -v/tau + I
        Surrogate spike: s = sigmoid(k*(v-th))
        Latency = first time index where s > 0.5
        """
        B,L = I.shape
        v = torch.zeros(B)
        spikes = torch.zeros(B, L)
        k = 10.0
        for t in range(L):
            dv = (-v/self.tau + I[:,t])
            v = v + dv
            s = torch.sigmoid(k*(v - self.th))   # surrogate spike
            spikes[:,t] = s
            v = v * (1 - s.detach())             # reset
        # latency per batch element
        latency = spikes.argmax(dim=1)
        return spikes, latency

# ------------------------------------------------------------
# 3. Deep dendritic block
# ------------------------------------------------------------
class DeepDendrites(nn.Module):
    def __init__(self, T, K1=16, K2=8):
        super().__init__()
        self.T = T
        self.K1 = K1
        self.K2 = K2
        self.W1 = nn.Linear(T, K1)
        self.W2 = nn.Linear(K1, K2)

    def forward(self, H):
        """
        H: [B, L, T] windows
        returns:
          dendritic_output: [B, L, K2]
          winner_idx:       [B, L]
          winner_vals:      [B, L]
        """
        d1 = torch.relu(self.W1(H))          # layer 1 dendrites
        d2 = torch.relu(self.W2(d1))         # layer 2 dendrites
        vals, idx = d2.max(dim=-1)           # dendritic competition
        return d2, idx, vals

# ------------------------------------------------------------
# 4. Full self-supervised predictor
# ------------------------------------------------------------
class LatencyPredictor(nn.Module):
    def __init__(self, T, K1=16, K2=8):
        super().__init__()
        self.T = T
        self.encA = DeepDendrites(T, K1, K2)
        self.encB = DeepDendrites(T, K1, K2)
        self.lifA = LIF()
        self.lifB = LIF()

    def forward(self, xa, xb):
        # sliding windows
        Ha = hankel_backward(xa, self.T)   # [B, L-T+1, T]
        Hb = hankel_backward(xb, self.T)

        # deep dendrites
        da, ida, vala = self.encA(Ha)
        db, idb, valb = self.encB(Hb)

        # soma input = max dendrite per step
        Ia = vala
        Ib = valb

        # LIF
        sa, la = self.lifA(Ia)
        sb, lb = self.lifB(Ib)

        return (da, ida, vala, la), (db, idb, valb, lb)

# ------------------------------------------------------------
# 5. Loss = |latencyA - latencyB|
# ------------------------------------------------------------
def latency_loss(la, lb):
    return torch.mean(torch.abs(la - lb))

# ------------------------------------------------------------
# 6. Visualization utilities
# ------------------------------------------------------------
def plot_winners(time, winner_idx):
    plt.figure(figsize=(8,3))
    plt.plot(time, winner_idx, lw=1)
    plt.title("Winning dendrite over time")
    plt.xlabel("Real time")
    plt.ylabel("Dendrite index")
    plt.tight_layout()

def plot_dendrite_io(time, x, dendritic_outputs):
    """
    x: [L] input signal
    dendritic_outputs: [L, K2]
    Show input (red) and dendritic output (blue)
    Two-column grid, 1 row per dendrite.
    """
    L, K2 = dendritic_outputs.shape
    fig, axes = plt.subplots(K2, 2, figsize=(10, 2*K2))
    for k in range(K2):
        axes[k,0].plot(time, x, color='red')
        axes[k,0].set_title(f"D{k} input")

        axes[k,1].plot(time, dendritic_outputs[:,k], color='blue')
        axes[k,1].set_title(f"D{k} output")

    plt.tight_layout()

# ------------------------------------------------------------
# 7. Training sketch (sine + noise → predict clean sine)
# ------------------------------------------------------------

model = LatencyPredictor(T=20)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(200):
    t = torch.linspace(0, 2*3.14, 200).unsqueeze(0)
    clean = torch.sin(t)
    noisy = clean + 0.2*torch.randn_like(clean)

    (da, ida, vala, la), (db, idb, valb, lb) = model(noisy, clean)
    loss = latency_loss(la, lb)

    opt.zero_grad()
    loss.backward()
    opt.step()

# ------------------------------------------------------------
# Usage sketch for visualization
# ------------------------------------------------------------
# After forward pass:
# time axis matching Hankel output length

time = t[0, -vala.shape[1]:]
plot_winners(time, ida[0])
plot_dendrite_io(time, noisy[0, -vala.shape[1]:], da[0])
