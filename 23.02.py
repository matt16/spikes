from gc import freeze

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader, TensorDataset


# -------------------- Surrogate --------------------
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, theta=1.0):
        ctx.save_for_backward(u)
        ctx.theta = theta
        return (u > theta).float()

    @staticmethod
    def backward(ctx, grad_output):
        (u,) = ctx.saved_tensors
        theta = ctx.theta
        grad = (1.0 - (u - theta).abs()).clamp(min=0.0)
        return grad_output * grad, None


spike_fn = SurrogateSpike.apply


# -------------------- Vectorized LIF --------------------
class LIFLayer(nn.Module):
    def __init__(self, n_filters, tau_mem=0.9, tau_trace=0.95, theta=1.0):
        super().__init__()
        self.tau_mem = tau_mem
        self.tau_trace = tau_trace
        self.theta = theta
        self.n_filters = n_filters
        self.u = None
        self.trace = None

    def reset(self, batch_size, device):
        self.u = torch.zeros(batch_size, self.n_filters, device=device)
        self.trace = torch.zeros(batch_size, self.n_filters, device=device)

    def forward(self, input_current):
        self.trace = self.tau_trace * self.trace + input_current
        self.u = self.tau_mem * self.u + self.trace
        s = spike_fn(self.u, self.theta)
        self.u = self.u * (1.0 - s)
        return s


# -------------------- Model --------------------
class FastSNN(nn.Module):
    def __init__(self, n_filters, window_size, n_classes):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_filters, window_size))
        self.lif = LIFLayer(n_filters)
        self.decoder = nn.Linear(n_filters, n_classes)

    def reset(self, batch_size, device):
        self.lif.reset(batch_size, device)

    def forward(self, x_windows):
        # x_windows: [T, B, W]
        spike_counts = 0
        for t in range(x_windows.shape[0]):
            # Matrix multiply instead of for-loop
            proj = x_windows[t] @ self.weights.T  # [B, K]
            s = self.lif(proj)
            spike_counts = spike_counts + s
        return self.decoder(spike_counts)


# -------------------- Hankel Builder --------------------
def make_hankel_sequences(x, W, T):
    windows = x.unfold(0, W, 1)
    seq = windows.unfold(0, T, 1)
    return seq.permute(0, 2, 1).contiguous()


# -------------------- Data --------------------
T = 8
B = 64
W = 100
L = 8000

t = torch.linspace(0, 10 * torch.pi, L)
s0 = torch.sin(20 * t)
s1 = torch.sin(100 * t)

X0 = make_hankel_sequences(s0, W, T)
X1 = make_hankel_sequences(s1, W, T)

y0 = torch.zeros(X0.size(0), dtype=torch.long)
y1 = torch.ones(X1.size(0), dtype=torch.long)

X = torch.cat([X0, X1])
y = torch.cat([y0, y1])

loader = DataLoader(TensorDataset(X, y), batch_size=B, shuffle=True)


# -------------------- Training --------------------
model = FastSNN(n_filters=16, window_size=W, n_classes=2)
opt = torch.optim.Adam([
    {"params": [model.weights], "lr": 1e-3},                 # Filter
    {"params": model.decoder.parameters(), "lr": 1e-3},      # Decoder langsamer
], weight_decay=0.0)
loss_fn = nn.CrossEntropyLoss()
freeze_epochs = 0
history = []

for epoch in range(20):
    freeze = epoch < freeze_epochs
    model.decoder.weight.requires_grad_(not freeze)
    model.decoder.bias.requires_grad_(not freeze)
    for x_batch, y_batch in loader:
        x_batch = x_batch.permute(1, 0, 2)

        model.reset(x_batch.shape[1], x_batch.device)

        logits = model(x_batch)
        loss = loss_fn(logits, y_batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

        history.append(loss.item())


# -------------------- Plot --------------------
#--------------Loss History -------
plt.figure()
plt.plot(history)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Loss History")
plt.show()
#-------------Filter-------
# --- after training ---
weights = model.weights.detach().cpu()   # shape [K, W]
K, W = weights.shape

# Grid-Layout (nahezu quadratisch)
ncols = math.ceil(math.sqrt(K))
nrows = math.ceil(K / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 2.6*nrows), sharex=True)
axes = axes.flatten()

x = range(W)

for k in range(K):
    axes[k].plot(x, weights[k].numpy())
    axes[k].set_title(f"Filter {k}")
    axes[k].set_xlabel("Window index")
    axes[k].set_ylabel("Weight")

# leere Achsen ausblenden (falls Grid größer als K)
for i in range(K, len(axes)):
    axes[i].axis("off")

fig.suptitle("Learned filter weights (K filters over W window positions)", y=1.02)
fig.tight_layout()
plt.show()