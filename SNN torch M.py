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

    def encode(self, drive: torch.Tensor) -> torch.Tensor:
        """
        drive: (B, C, T)
        returns: latency (B, C) where value T means 'no spike'
        """
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
# Multi-path depthwise encoder (returns per-dendrite activations)
# ------------------------
class DepthwiseMultiPathEncoder(nn.Module):
    def __init__(self, channels, kernel_specs=[(3, 6), (5, 6), (9, 6)]):
        """
        kernel_specs: list of (kernel_size, out_per_kernel)
        Produces:
            - summed: (B, C, T)
            - dendritic_drive: (B, C, D, T) where D = sum(out_per_kernel)
        """
        super().__init__()
        self.channels = channels
        self.kernel_specs = kernel_specs
        self.convs = nn.ModuleList()
        self.total_per_channel = sum(out_per for k, out_per in kernel_specs)
        # Build grouped convs (depthwise per-channel with multiple outputs per channel)
        for k, out_per in kernel_specs:
            padding = (k - 1) // 2
            conv = nn.Conv1d(in_channels=channels, out_channels=channels * out_per, kernel_size=k, padding=padding, groups=channels)
            self.convs.append(conv)
        # Reduce (grouped 1x1 conv) to combine per-channel dendrites into one soma drive
        self.reduce = nn.Conv1d(in_channels=channels * self.total_per_channel, out_channels=channels, kernel_size=1, groups=channels)

    def forward(self, x: torch.Tensor):
        # x: (B, C, T)
        dendritic_outs = []  # list of (B, C, out_per, T)
        for (k, out_per), conv in zip(self.kernel_specs, self.convs):
            out = conv(x)  # (B, C*out_per, T)
            B, _, T = out.shape
            out = out.view(B, self.channels, out_per, T)  # (B, C, out_per, T)
            dendritic_outs.append(out)
        # Concatenate along per-channel dendrite axis -> (B, C, D, T)
        dendritic_drive = torch.cat(dendritic_outs, dim=2)  # D = total_per_channel
        # Build summed drive for LIF: reshape and reduce via grouped conv
        B, C, D, T = dendritic_drive.shape
        summed = self.reduce(dendritic_drive.view(B, C * D, T))  # (B, C, T)
        return summed, dendritic_drive

    def get_all_kernels(self):
        """
        Return kernels as a tensor of shape (C, D, K) where:
          C = channels, D = total_per_channel, K = kernel_size (varies across groups).
        Note: kernel_sizes may differ across kernel_specs; this routine returns a list of arrays per-kernel-size.
        For convenience we return a list of tensors, each shaped (C, out_per, K) in the same order as kernel_specs.
        """
        kernels_per_spec = []
        for (k, out_per), conv in zip(self.kernel_specs, self.convs):
            # conv.weight shape: (C*out_per, 1, k)
            w = conv.weight.detach().cpu()  # (C*out_per, 1, k)
            C = self.channels
            w = w.view(C, out_per, k)       # (C, out_per, K)
            kernels_per_spec.append(w)
        # Return list of (C, out_per, K) in kernel_specs order
        return kernels_per_spec

# ------------------------
# Channel-latency Spiking Seq->Value model with STDP
# ------------------------
class ChannelLatencySeq2Value(nn.Module):
    def __init__(self, channels, kernel_specs=[(3,6),(5,6),(9,6)], lif_tau=5.0, lif_threshold=1.0, decoder_channels=128):
        super().__init__()
        self.channels = channels
        self.encoder = DepthwiseMultiPathEncoder(channels, kernel_specs=kernel_specs)
        self.lif = LIFLatencyEncoder(tau_mem=lif_tau, threshold=lif_threshold)
        # will hold concatenated (C*D,) relevance vectors per spike (reset per forward)
        self.relevance_records = []
        # learnable scale to map latency->activation
        self.latency_scale = nn.Parameter(torch.tensor(5.0))
        # output gates: channels x channels (source -> target); zero diagonal enforced regularly
        self.output_gates = nn.Parameter(torch.zeros(channels, channels))
        # bias per target (unused currently but left for extension)
        self.bias = nn.Parameter(torch.zeros(channels))
        # MLP that maps mixed activations -> predicted latency (one scalar per channel)
        self.post_mlp = nn.Sequential(nn.Linear(channels, decoder_channels), nn.ReLU(), nn.Linear(decoder_channels, channels))
        nn.init.xavier_uniform_(self.output_gates)
        with torch.no_grad():
            self.zero_output_diagonal_()

    def zero_output_diagonal_(self):
        with torch.no_grad():
            diag_idx = torch.eye(self.channels, device=self.output_gates.device).bool()
            self.output_gates[diag_idx] = 0.0

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, T)
        returns:
            pred_latency: (B, C)
            true_latency: (B, C)
            act: (B, C)
            dendritic_drive: (B, C, D, T)
        Side effect:
            self.relevance_records <- list of tensors shape (C*D,) for each spike observed in the batch
        """
        B, C, T = x.shape
        drive, dendritic_drive = self.encoder(x)          # drive: (B,C,T), dendritic_drive: (B,C,D,T)
        true_latency = self.lif.encode(drive)             # (B, C)
        # reset records for this forward pass
        self.relevance_records = []
        # record dendritic snapshot at spike times (flattened across channels x dendrites)
        _, _, D, _ = dendritic_drive.shape
        for b in range(B):
            for c in range(C):
                t_spike = int(true_latency[b, c].item())
                if t_spike >= T:
                    # no spike for this (b,c)
                    continue
                # snapshot across all channels and dendrites at that time: shape (C, D)
                snap = dendritic_drive[b, :, :, t_spike]  # (C, D)
                # normalize to a distribution (use absolute to avoid sign cancellations)
                snap_abs = snap.abs()
                denom = snap_abs.sum() + 1e-12
                snap_norm = (snap_abs / denom).reshape(-1)  # flattened (C*D,)
                self.relevance_records.append(snap_norm.cpu())
        # convert true_latency to activation (earlier -> larger)
        scale = torch.clamp(self.latency_scale, min=1e-3)
        act = torch.exp(-true_latency / scale)  # (B, channels)
        # mix via gates: mixed_j = bias_j + sum_i gates[j,i] * act_i
        mixed = torch.matmul(act, self.output_gates.t()) + self.bias.unsqueeze(0)
        # predict latency directly from mixed signal
        raw_pred = self.post_mlp(mixed)  # (batch, channels) unconstrained
        # make positive and in-range [0,T]
        pred_latency = F.softplus(raw_pred)
        pred_latency = torch.clamp(pred_latency, min=0.0, max=float(T))
        return pred_latency, true_latency, act, dendritic_drive

# ------------------------
# STDP update function (batch-averaged)
# ------------------------
def stdp_update_from_latencies(model: ChannelLatencySeq2Value, latencies: torch.Tensor, stdp_lr=1e-3, A_plus=0.01, A_minus=0.012, tau_plus=2.0, tau_minus=2.0, clip=1.0):
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
    delta_w = delta_w * (1.0 - torch.eye(C, device=device))  # zero diag
    with torch.no_grad():
        model.output_gates += stdp_lr * delta_w
        model.output_gates.clamp_(min=-clip, max=clip)
        model.zero_output_diagonal_()

# ------------------------
# Utilities
# ------------------------
def extract_channel_adjacency(model, threshold=1e-3):
    gate = model.output_gates.detach().cpu().numpy().copy()
    np.fill_diagonal(gate, 0.0)
    adj = (np.abs(gate) >= threshold).astype(float)
    return adj, gate

def plot_loss(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history['train_loss'], label='Train loss')
    if any(v is not None for v in history['val_loss']):
        plt.plot([v for v in history['val_loss'] if v is not None], label='Val loss')
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
        ax.plot([true[:, i].min(), true[:, i].max()], [true[:, i].min(), true[:, i].max()], 'r--', lw=1.5)
        ax.set_title(f'Channel {i + 1}')
        ax.set_xlabel('True latency')
        if i == 0:
            ax.set_ylabel('Predicted latency')
        ax.grid(True, alpha=0.3)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_adjacency(adj, title='Inferred adjacency'):
    plt.figure(figsize=(5, 4))
    plt.imshow(adj, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Gate presence')
    plt.title(title)
    plt.xlabel('Source channel')
    plt.ylabel('Target channel')
    plt.show()

def plot_relevant_kernels(model: ChannelLatencySeq2Value, encoder_kernels, spike_batch_label="Spike Batch"):
    """
    Plots:
      - heatmap of relevance records (concatenated C*D dendrites on x axis)
      - bar plots of kernel weights for each dendrite in the same concatenation order
    encoder_kernels can be:
      - a list (model.encoder.convs)
      - or a list returned by encoder.get_all_kernels() (list of (C, out_per, K))
    """
    if not model.relevance_records:
        print("No relevance records found. Run a forward pass first!")
        return

    # ---- stack records: list of (C*D,) tensors -> (S, C*D)
    R = torch.stack(model.relevance_records)  # (S, C*D)
    R_np = R.numpy()  # (S, C*D)

    # Heatmap: x-axis = concatenated dendrite index (C*D), y-axis = spike index
    plt.figure(figsize=(12, 6))
    plt.imshow(R_np.T, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Normalized dendritic drive')
    plt.xlabel("Spike index")
    plt.ylabel("Concatenated Channels × Dendrites")
    plt.title(f"Dendritic Relevance Heatmap ({spike_batch_label})")
    plt.tight_layout()
    plt.show()

    # ---- build kernel matrix in the same concatenation order (C*D, K)
    # encoder_kernels can be a list of conv modules or list of (C, out_per, K) tensors
    if isinstance(encoder_kernels, (list, tuple)) and len(encoder_kernels) > 0 and isinstance(encoder_kernels[0], nn.Conv1d):
        # encoder_kernels is the list of conv modules (model.encoder.convs)
        kernels_list = []
        for (k, out_per), conv in zip(model.encoder.kernel_specs, model.encoder.convs):
            w = conv.weight.detach().cpu()           # (C*out_per, 1, k)
            C = model.encoder.channels
            w = w.view(C, out_per, -1)               # (C, out_per, K)
            kernels_list.append(w)                   # keep per-spec
        # concatenate along dendrite axis -> (C, D, K)
        kernels_CD = torch.cat(kernels_list, dim=1)    # (C, D, K)
    else:
        # assume encoder_kernels is list of (C, out_per, K) tensors (from get_all_kernels)
        # or a single numpy/tensor shaped (C, D, K)
        if isinstance(encoder_kernels, list):
            kernels_CD = torch.cat(encoder_kernels, dim=1)  # list of (C, out_per, K)
        else:
            # tensor-like
            kernels_CD = torch.tensor(encoder_kernels) if not isinstance(encoder_kernels, torch.Tensor) else encoder_kernels

    C, D, K = kernels_CD.shape
    kernels_flat = kernels_CD.reshape(C * D, K).numpy()  # (C*D, K)

    # ---- bar plot for each dendrite (arranged in grid if many dendrites)
    num_dendrites = kernels_flat.shape[0]
    cols = 4
    rows = int(np.ceil(num_dendrites / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows))
    axes = np.array(axes).reshape(-1)
    for i in range(len(axes)):
        ax = axes[i]
        if i < num_dendrites:
            ax.bar(np.arange(K), kernels_flat[i])
            ax.set_title(f"Dendrite {i}")
            ax.set_xticks([])
        else:
            ax.axis('off')
    plt.suptitle("Dendrite Kernel Weights (concatenated C×D order)")
    plt.tight_layout()
    plt.show()

# ------------------------
# MAIN: simplified training loop with extended losses
# ------------------------
if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # toy data
    N, C, T = 512, 3, 64
    t = torch.arange(T).float()
    X = torch.sin(t.unsqueeze(0) * 0.2).unsqueeze(0).repeat(N, 1, 1) + 0.01 * torch.randn(N, 1, T)
    Y = 0.5 * F.pad(X[:, 0, :-1], (1, 0)) + 0.35 * torch.randn(N, T)  # more noisy so longer latency
    Y = Y.unsqueeze(1)
    Z = 0.3 * F.pad(X[:, 0, :-1], (1, 0)) + 0.4 * F.pad(Y[:, 0, :-1], (1, 0)) + 0.25 * torch.randn(N, T)
    Z = Z.unsqueeze(1)
    data = torch.cat([X, Y, Z], dim=1)
    inp = data  # supervised target is derived from the same multichannel sequence

    # model
    model = ChannelLatencySeq2Value(channels=C, kernel_specs=[(3,6),(5,6),(9,6)], lif_tau=5.0, lif_threshold=0.03)
    model.to(device)
    model.zero_output_diagonal_()

    # training hyperparams
    epochs = 500
    batch_size = 128
    lr = 1e-3
    l1_weight = 1e-5         # sparsity coefficient (lambda)
    beta_cycle = 1e-2        # cycle suppression coefficient (beta)
    gamma_dom = 0.0          # dominance coefficient (optional)
    epsilon_dom = 1.0        # small threshold for dominance term (unused if gamma_dom==0)
    proximal_lambda = 1e-4
    stdp_lr = 1e-2
    adjacency_threshold = 0.05
    order_consistency_alpha = 0.0

    # data loader
    dataset = TensorDataset(inp)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # optimizer excludes output_gates (updated by STDP)
    opt = torch.optim.Adam([p for p in model.parameters() if p is not model.output_gates], lr=lr)
    history = {'train_loss': [], 'val_loss': [], 'adjacency': []}

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        n_examples = 0
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            pred_lat, true_lat, act, dendrites = model(xb)
            # supervised latency reconstruction (L1)
            latency_loss = torch.mean(torch.abs(pred_lat - true_lat))
            # optional ordering consistency loss (keeps predicted order consistent with true)
            order_loss = 0.0
            if order_consistency_alpha > 0.0:
                pred_diff = pred_lat.unsqueeze(1) - pred_lat.unsqueeze(2)
                true_diff = true_lat.unsqueeze(1) - true_lat.unsqueeze(2)
                order_loss = torch.mean(torch.relu(-pred_diff * torch.sign(true_diff)))

            # sparsity (L1) on gates
            l1 = l1_weight * torch.sum(torch.abs(model.output_gates))

            # cycle suppression: penalize mutual edges g_ij * g_ji
            G = model.output_gates
            Cc = G.shape[0]
            eye = torch.eye(Cc, device=G.device)
            mask = 1.0 - eye
            # sum_{i<j} |g_ij * g_ji| = 0.5 * sum_{i!=j} |g_ij * g_ji|
            cycle_loss = beta_cycle * 0.5 * torch.sum(torch.abs((G * G.t()) * mask))

            # optional dominance penalty: penalize many small parents per target (encourage one strong parent)
            dom_loss = 0.0
            if gamma_dom > 0.0:
                # for each target j, measure L1 over incoming edges and penalize if it exceeds epsilon_dom
                incoming = torch.sum(torch.abs(G), dim=1)  # shape (C,) incoming sum for each target
                dom_loss = gamma_dom * torch.sum(torch.relu(incoming - epsilon_dom))

            loss = latency_loss + order_consistency_alpha * order_loss + l1 + cycle_loss + dom_loss
            loss.backward()
            opt.step()

            # proximal shrinkage on gates (soft-thresholding) to encourage exact sparsity
            if proximal_lambda > 0.0:
                with torch.no_grad():
                    param = model.output_gates
                    param.copy_(torch.sign(param) * torch.clamp(torch.abs(param) - proximal_lambda, min=0.0))
                    model.zero_output_diagonal_()

            # STDP update (outside autograd)
            stdp_update_from_latencies(model, true_lat, stdp_lr=stdp_lr)

            total_loss += loss.item() * xb.shape[0]
            n_examples += xb.shape[0]

        avg_loss = total_loss / n_examples
        history['train_loss'].append(avg_loss)

        # simple val using the whole dataset (same data here)
        model.eval()
        with torch.no_grad():
            x_val = inp.to(device)
            p_lat, t_lat, _, _ = model(x_val)
            val_loss = torch.mean(torch.abs(p_lat - t_lat)).item()
            history['val_loss'].append(val_loss)
        adj, gate_mat = extract_channel_adjacency(model, threshold=adjacency_threshold)
        history['adjacency'].append(adj)
        if (ep + 1) % max(1, epochs // 10) == 0 or ep < 5:
            print(f"Epoch {ep+1}/{epochs}  train_loss={avg_loss:.6f}  val_loss={val_loss:.6f}")

    # ------------------------
    # Post-training: adjacency and dendritic attribution example
    # ------------------------
    adj, gate = extract_channel_adjacency(model, threshold=adjacency_threshold)
    print('Learned adjacency (thresholded):')
    print(adj)
    plot_loss(history)
    plot_adjacency(adj, title='Inferred gates (thresholded)')
    # compute dendritic causal strength for each discovered edge using a small batch
    x_val = inp[:64].to(device)
    with torch.no_grad():
        p_lat, t_lat, act, dendrites = model(x_val)
    # Show relevance heatmap and kernel plots (concatenated C*D)
    plot_relevant_kernels(model, encoder_kernels=model.encoder.convs, spike_batch_label="Validation Batch")
    # optional: plot predicted vs true latencies for first 64 examples
    plot_latencies_subplots(p_lat.detach(), t_lat.detach())
