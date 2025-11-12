"""
Channel-latency Spiking Seq→Value Predictor with STDP and latency loss (PyTorch)

This revision implements your requested change:
- Supervised objective is **latency loss** (absolute difference between predicted and true spike times) instead of MSE on scalar outputs.
- The model predicts **latencies** directly (one scalar per channel). The true latencies are computed by applying the same encoder+LIF latency encoder to the data — so encoder and decoder use the exact same latency code.
- STDP remains and uses batch-averaged latencies to potentiate source->target gates when source spikes earlier than target.

Key changes:
- `ChannelLatencySeq2Value.forward` now returns `(pred_latency, true_latency, act)` where `true_latency` is computed from the input drive and `pred_latency` is produced by the post-MLP (softplus + clamp to time range).
- Training loop `train_seq2value_stdp` uses **L1 latency difference loss**: `loss = mean(|pred_latency - true_latency|) + l1_on_gates`.
- Data loader now expects **only the input sequence tensor** (targets are derived as latencies from the data itself).
- Plotting updated to show predicted vs true latencies per channel.

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ------------------------
# Latency LIF encoder (stateless)
# ------------------------
class LIFLatencyEncoder:
    def __init__(self, tau_mem=5.0, threshold=1.0):
        self.tau_mem = tau_mem
        self.alpha = math.exp(-1.0 / tau_mem)
        self.threshold = threshold

    def encode(self, drive):
        # drive: (batch, channels, time)
        B, C, T = drive.shape
        device = drive.device
        V = torch.zeros(B, C, device=device)
        fired = torch.zeros(B, C, dtype=torch.bool, device=device)
        latency = torch.full((B, C), float(T), device=device)

        for t in range(T):
            V = self.alpha * V + (1.0 - self.alpha) * drive[:, :, t]
            just_fired = (V >= self.threshold) & (~fired)
            if just_fired.any():
                latency[just_fired] = float(t)
                fired = fired | just_fired
        # latency is integer-valued in [0,T], T means 'no spike'
        return latency  # (B, C)

# ------------------------
# Multi-path depthwise encoder
# ------------------------
class DepthwiseMultiPathEncoder(nn.Module):
    def __init__(self, channels, kernel_specs=[(3,6),(5,6),(9,6)]):
        """kernel_specs: list of (kernel_size, out_per_kernel)
        Produces per-channel aggregated drive: (batch, channels, time)
        """
        super().__init__()
        self.channels = channels
        self.convs = nn.ModuleList()
        self.total_per_channel = 0
        for k, out_per in kernel_specs:
            padding = (k - 1) // 2
            conv = nn.Conv1d(in_channels=channels,
                             out_channels=channels * out_per,
                             kernel_size=k,
                             padding=padding,
                             groups=channels)
            self.convs.append(conv)
            self.total_per_channel += out_per
        # projection to reduce out_per_channel -> 1 per channel (implemented as grouped conv)
        self.reduce = nn.Conv1d(in_channels=channels * self.total_per_channel,
                                out_channels=channels,
                                kernel_size=1,
                                groups=channels)

    def forward(self, x):
        # x: (batch, channels, time)
        outs = [conv(x) for conv in self.convs]
        cat = torch.cat(outs, dim=1)  # (batch, channels*total_per, time)
        reduced = self.reduce(cat)    # (batch, channels, time)
        return reduced

# ------------------------
# Channel-latency Spiking Seq->Value model with STDP
# ------------------------
class ChannelLatencySeq2Value(nn.Module):
    def __init__(self, channels, kernel_specs=[(3,6),(5,6),(9,6)],
                 lif_tau=5.0, lif_threshold=1.0, decoder_channels=128):
        super().__init__()
        self.channels = channels
        self.encoder = DepthwiseMultiPathEncoder(channels, kernel_specs=kernel_specs)
        self.lif = LIFLatencyEncoder(tau_mem=lif_tau, threshold=lif_threshold)
        # learnable scale to map latency->activation
        self.latency_scale = nn.Parameter(torch.tensor(5.0))

        # output gates: channels x channels (source -> target); zero diagonal enforced regularly
        self.output_gates = nn.Parameter(torch.zeros(channels, channels))
        # bias per target
        self.bias = nn.Parameter(torch.zeros(channels))

        # MLP that maps mixed activations -> predicted latency (one scalar per channel)
        # We'll output positive numbers via softplus and clamp them to [0, T]
        self.post_mlp = nn.Sequential(
            nn.Linear(channels, decoder_channels),
            nn.ReLU(),
            nn.Linear(decoder_channels, channels)
        )
        nn.init.xavier_uniform_(self.output_gates)
        with torch.no_grad():
            self.zero_output_diagonal_()

    def zero_output_diagonal_(self):
        with torch.no_grad():
            diag_idx = torch.eye(self.channels, device=self.output_gates.device).bool()
            self.output_gates[diag_idx] = 0.0

    def forward(self, x):
        # x: (batch, channels, time)
        B, C, T = x.shape
        drive = self.encoder(x)  # (batch, channels, time)
        true_latency = self.lif.encode(drive)  # (batch, channels)
        # convert true_latency to activation (earlier -> larger)
        scale = torch.clamp(self.latency_scale, min=1e-3)
        act = torch.exp(-true_latency / scale)  # (batch, channels)

        # mix via gates: mixed_j = bias_j + sum_i gates[j,i] * act_i
        mixed = torch.matmul(act, self.output_gates.t()) + self.bias.unsqueeze(0)

        # predict latency directly from mixed signal
        raw_pred = self.post_mlp(mixed)  # (batch, channels) unconstrained
        # make positive and in-range [0,T]
        pred_latency = F.softplus(raw_pred)
        pred_latency = torch.clamp(pred_latency, min=0.0, max=float(T))

        return pred_latency, true_latency, act

# ------------------------
# STDP update function (batch-averaged)
# ------------------------

def stdp_update_from_latencies(model, latencies, stdp_lr=1e-3, A_plus=0.01, A_minus=0.012, tau_plus=2.0, tau_minus=2.0, clip=1):
    # latencies: (batch, channels)
    mean_lat = latencies.mean(dim=0)  # (channels,)
    C = latencies.shape[1]
    device = mean_lat.device
    t_post = mean_lat.view(C, 1)
    t_pre = mean_lat.view(1, C)
    delta = t_post - t_pre
    pos_mask = (delta > 0).float()
    neg_mask = (delta <= 0).float()
    pot = A_plus * torch.exp(-delta / tau_plus) * pos_mask
    dep = -A_minus * torch.exp(delta / tau_minus) * neg_mask
    delta_w = pot + dep
    delta_w = delta_w * (1.0 - torch.eye(C, device=device))
    with torch.no_grad():
        model.output_gates += stdp_lr * delta_w
        model.output_gates.clamp_(min=-clip, max=clip)
        diag_idx = torch.eye(C, device=device).bool()
        model.output_gates[diag_idx] = 0.0

# ------------------------
# Utilities: l1, adjacency, plotting
# ------------------------

def l1_penalty(model, weight=1e-4):
    return weight * torch.sum(torch.abs(model.output_gates))


def extract_channel_adjacency(model, threshold=1e-3):
    gate = model.output_gates.detach().cpu().numpy().copy()
    np.fill_diagonal(gate, 0.0)
    adj = (np.abs(gate) >= threshold).astype(float)
    return adj, gate


def plot_loss(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history['train_loss'], label='Train loss')
    if history['val_loss'][0] is not None:
        plt.plot(history['val_loss'], label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Evolution')
    plt.show()


def plot_latencies_subplots(pred_latencies, true_latencies, title="Predicted vs True Latencies"):
    pred = pred_latencies.detach().cpu().numpy()
    true = true_latencies.detach().cpu().numpy()
    num_channels = pred.shape[1]
    fig, axes = plt.subplots(1, num_channels, figsize=(4 * num_channels, 4), sharey=True)
    if num_channels == 1:
        axes = [axes]
    for i in range(num_channels):
        ax = axes[i]
        ax.scatter(true[:, i], pred[:, i], alpha=0.6, edgecolor='k')
        ax.plot([true[:, i].min(), true[:, i].max()],
                [true[:, i].min(), true[:, i].max()], 'r--', lw=1.5)
        ax.set_title(f'Channel {i + 1}')
        ax.set_xlabel('True latency')
        if i == 0:
            ax.set_ylabel('Predicted latency')
        ax.grid(True, alpha=0.3)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_latencies(pred_lat, true_lat):
    # pred_lat, true_lat: (N, channels)
    N, C = true_lat.shape
    plt.figure(figsize=(8, 4))
    for c in range(C):
        plt.scatter(range(N), true_lat[:, c].cpu().numpy(), label=f'true_ch{c}', alpha=0.6)
        plt.scatter(range(N), pred_lat[:, c].cpu().numpy(), marker='x', label=f'pred_ch{c}', alpha=0.6)
    plt.legend()
    plt.title('Predicted vs True Latencies (per channel)')
    plt.xlabel('Example index')
    plt.ylabel('Latency (timesteps)')
    plt.show()


def plot_adjacency(adj, title='Inferred adjacency'):
    plt.figure(figsize=(5, 4))
    plt.imshow(adj, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Gate presence')
    plt.title(title)
    plt.xlabel('Source channel')
    plt.ylabel('Target channel')
    plt.show()

# ------------------------
# Training loop: supervised latency loss + STDP
# ------------------------

def train_seq2value_stdp(model, data_tensor, val_tensor=None, epochs=200, batch_size=64, lr=1e-3,
                         l1_weight=1e-4, proximal_lambda=0.0, stdp_lr=1e-3,
                         adjacency_threshold=0.1, device=None, autoreg_masking=True,
                         order_consistency_alpha=0.0,
                         # 🔽 NEU:
                         print_adj_per_batch=False,
                         adj_print_threshold=0.1):
    import numpy as np
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device)
    model.zero_output_diagonal_()

    if isinstance(data_tensor, (tuple, list)):
        x_tensor = data_tensor[0]
    else:
        x_tensor = data_tensor

    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # optimizer excludes output_gates (updated by STDP)
    opt = torch.optim.Adam([p for p in model.parameters() if p is not model.output_gates], lr=lr)

    channels = model.channels
    target_mask = torch.ones(channels, device=device)

    history = {'train_loss': [], 'val_loss': [], 'adjacency': []}

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for bidx, (xb,) in enumerate(loader):
            xb = xb.to(device)

            # ---------- 🔎 DEBUG-PRINT *vor* Updates: Adjazenz dieser Iteration ----------
            if print_adj_per_batch:
                with torch.no_grad():
                    # Diagonale sauber halten, dann kopieren
                    model.zero_output_diagonal_()
                    W = model.output_gates.detach().cpu().numpy().copy()
                    np.fill_diagonal(W, 0.0)
                    print("Raw output_gates:\n", np.round(W, 3))


            # ---------- normaler Trainingsschritt ----------
            opt.zero_grad()
            pred_lat, true_lat, act = model(xb)

            latency_loss = torch.mean(torch.abs(pred_lat - true_lat))

            order_loss = 0.0
            if order_consistency_alpha > 0.0:
                pred_diff = pred_lat.unsqueeze(1) - pred_lat.unsqueeze(2)
                true_diff = true_lat.unsqueeze(1) - true_lat.unsqueeze(2)
                order_loss = torch.mean(torch.relu(-pred_diff * torch.sign(true_diff)))

            loss = latency_loss + order_consistency_alpha * order_loss + l1_penalty(model, weight=l1_weight)
            loss.backward()
            opt.step()

            # Proximal-Schrumpfung auf Gates (optional)
            if proximal_lambda > 0.0:
                with torch.no_grad():
                    param = model.output_gates
                    param.copy_(torch.sign(param) * torch.clamp(torch.abs(param) - proximal_lambda, min=0.0))
                    diag_idx = torch.eye(channels, device=param.device).bool()
                    param[diag_idx] = 0.0

            # STDP-Update (außerhalb des Gradientenflusses)
            stdp_update_from_latencies(model, true_lat, stdp_lr=stdp_lr)

            total_loss += loss.item() * xb.shape[0]
            n += xb.shape[0]

        avg_loss = total_loss / n
        history['train_loss'].append(avg_loss)

        if val_tensor is not None:
            model.eval()
            with torch.no_grad():
                if isinstance(val_tensor, (tuple, list)):
                    x_val = val_tensor[0]
                else:
                    x_val = val_tensor
                x_val = x_val.to(device)
                p_lat, t_lat, _ = model(x_val)
                val_loss = torch.mean(torch.abs(p_lat - t_lat)).item()
                history['val_loss'].append(val_loss)
        else:
            history['val_loss'].append(None)

        adj, gate_mat = extract_channel_adjacency(model, threshold=adjacency_threshold)
        history['adjacency'].append(adj)

        if autoreg_masking:
            outgoing_counts = adj.sum(axis=1)
            new_mask = torch.ones(channels, device=device)
            for i in range(channels):
                if outgoing_counts[i] >= (channels - 1):
                    new_mask[i] = 0.0
            target_mask = new_mask

        if (ep + 1) % max(1, epochs // 10) == 0 or ep < 5:
            print(f"Epoch {ep+1}/{epochs}  train_loss={avg_loss:.6f}  val_loss={history['val_loss'][-1]}")
    return model, history

# ------------------------
# Example usage
# ------------------------
if __name__ == '__main__':
    torch.manual_seed(0)
    N, C, T = 512, 3, 64
    t = torch.arange(T).float()
    X = torch.sin(t.unsqueeze(0) * 0.2).unsqueeze(0).repeat(N, 1, 1) + 0.01 * torch.randn(N, 1, T)
    Y = 0.5 * F.pad(X[:, 0, :-1], (1, 0)) + 0.35 * torch.randn(N, T)  # more noisy so longer latency
    Y = Y.unsqueeze(1)
    Z = 0.3 * F.pad(X[:, 0, :-1], (1, 0)) + 0.4 * F.pad(Y[:, 0, :-1], (1, 0)) + 0.25 * torch.randn(N, T)
    Z = Z.unsqueeze(1)
    data = torch.cat([X, Y, Z], dim=1)

    inp = data  # we derive target latencies from the same multichannel sequence

    model = ChannelLatencySeq2Value(channels=C, kernel_specs=[(3,6),(5,6),(9,6)], lif_tau=5.0, lif_threshold=0.5)

    model, history = train_seq2value_stdp(model, inp, epochs=50, batch_size=128, lr=1e-3,
                                         l1_weight=1e-5, proximal_lambda=1e-4, stdp_lr=1e-2,print_adj_per_batch=True,
                                         adjacency_threshold=0.05)

    adj, gate = extract_channel_adjacency(model, threshold=0.05)
    print('Learned adjacency (thresholded):')
    print(adj)

    # visualize latencies and adjacency
    x_val = inp[:64]
    pred_lat, true_lat, _ = model(x_val)
    plot_loss(history)
    plot_latencies_subplots(pred_lat.detach(), true_lat.detach())
    plot_adjacency(adj, title='Inferred gates (thresholded)')