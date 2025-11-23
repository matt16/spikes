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
    def __init__(self, channels, kernel_specs=[(3,2),(5,2)], lif_tau=5.0, lif_threshold=1.0, decoder_channels=128):
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
        mixed = torch.matmul(act, self.output_gates.t()) #+ self.bias.unsqueeze(0)
        # predict latency directly from mixed signal
        raw_pred = self.post_mlp(mixed)  # (batch, channels) unconstrained
        # make positive and in-range [0,T]
        pred_latency = F.softplus(raw_pred)
        pred_latency = torch.clamp(pred_latency, min=0.0, max=float(T))
        return pred_latency, true_latency, act, dendritic_drive

# ------------------------
# STDP update function (batch-averaged)
# ------------------------
def stdp_update_from_latencies(
    model: ChannelLatencySeq2Value,
    latencies: torch.Tensor,
    stdp_lr=1e-3,
    A_plus=0.01,
    A_minus=0.012,
    tau_plus=2.0,
    tau_minus=2.0,
    clip=1.0,
):
    #print("\n[STDP BEFORE] output_gates before STDP update:")
    #print(model.output_gates.detach().cpu().numpy())

    # latencies: (batch, channels)
    mean_lat = latencies.mean(dim=0)  # (channels,)
    C = latencies.shape[1]
    device = mean_lat.device

    # compute STDP pairwise timing differences
    t_post = mean_lat.view(C, 1)
    t_pre  = mean_lat.view(1, C)
    delta = t_post - t_pre

    pos_mask = (delta > 0).float()
    neg_mask = (delta <= 0).float()

    pot = A_plus  * torch.exp(-delta / tau_plus)  * pos_mask
    dep = -A_minus * torch.exp( delta / tau_minus) * neg_mask

    delta_w = pot + dep
    delta_w = delta_w * (1.0 - torch.eye(C, device=device))  # zero diag

    # -----------------------------
    # compute STDP metric (before update)
    # -----------------------------
    stdp_metric = (stdp_lr * delta_w).abs().mean().item()

    # -----------------------------
    # apply update
    # -----------------------------
    with torch.no_grad():
        model.output_gates += stdp_lr * delta_w
        model.output_gates.clamp_(min=-clip, max=clip)
        model.zero_output_diagonal_()

    return stdp_metric

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
        ax.plot(
            [true[:, i].min(), true[:, i].max()],
            [true[:, i].min(), true[:, i].max()],
            'r--', lw=1.5
        )
        ax.set_title(f'Channel {i + 1}')
        ax.set_xlabel('True latency')
        if i == 0:
            ax.set_ylabel('Predicted latency')
        ax.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    # ------------------------------
    # NEW: Print true vs predicted for channel 0
    # ------------------------------
    print("\n=== Channel 0: True vs Predicted Latencies ===")
    print("True values (list):")
    print(true[:, 0].tolist())

    print("\nPredicted values (list):")
    print(pred[:, 0].tolist())


#plot different loss components
def plot_loss_components(history):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    components = [
        ('train_latency_loss',   'Latency loss'),
        ('train_order_loss',     'Order loss'),
        ('train_sparsity_loss',  'Sparsity (L1)'),
        ('train_dom_loss',       'Dominance loss'),
        ('train_cycle_loss',     'Cycle loss'),
        ('train_stdp_update',    'STDP update'),
    ]

    for ax, (key, title) in zip(axes, components):
        if key not in history or len(history[key]) == 0:
            ax.set_visible(False)
            continue

        ax.plot(history[key], label=title)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle('Loss Components Evolution')
    plt.tight_layout()
    plt.show()


def plot_output_gate_trajectories_batches(gate_history):

    gate_hist = np.stack(gate_history, axis=0)   # (B, C, C)
    B, C, _ = gate_hist.shape
    steps = np.arange(B)

    # Alle Off-Diagonal-Paare
    indices = [(i, j) for i in range(C) for j in range(C) if i != j]
    n_plots = len(indices)

    cols = int(np.ceil(np.sqrt(n_plots)))
    rows = int(np.ceil(n_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), sharex=True)
    axes = np.array(axes).reshape(rows, cols)

    for k, (i, j) in enumerate(indices):
        r = k // cols
        c = k % cols
        ax = axes[r, c]

        ax.plot(steps, gate_hist[:, i, j])
        ax.set_title(f"gate[{i}→{j}]")
        ax.grid(True, alpha=0.3)

        if r == rows - 1:
            ax.set_xlabel("Batch Step")
        if c == 0:
            ax.set_ylabel("Weight")

    # Unbenutzte Achsen ausblenden
    for k in range(n_plots, rows*cols):
        r = k // cols
        c = k % cols
        axes[r, c].set_visible(False)

    plt.suptitle("Output-gate weights over training (per batch)")
    plt.tight_layout()
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

    # ---- build list of per-dendrite kernel vectors (each can have its own length K_i)
    dendrite_kernels = []

    # Fall 1: wir bekommen die Conv-Layer (z.B. model.encoder.convs)
    if isinstance(encoder_kernels, (list, tuple, nn.ModuleList)) and len(encoder_kernels) > 0 and isinstance(
            encoder_kernels[0], nn.Conv1d):
        for (k_size, out_per), conv in zip(model.encoder.kernel_specs, encoder_kernels):
            # conv.weight: (C*out_per, 1, k_size)
            w = conv.weight.detach().cpu()
            C = model.encoder.channels
            w = w.view(C, out_per, k_size)  # (C, out_per, k_size)
            for c in range(C):
                for d in range(out_per):
                    # 1D-Array für diese Dendrite
                    dendrite_kernels.append(w[c, d].numpy())  # shape: (k_size,)

    # Fall 2: wir bekommen schon Tensoren aus get_all_kernels()
    elif isinstance(encoder_kernels, list) and len(encoder_kernels) > 0 and isinstance(encoder_kernels[0],
                                                                                       torch.Tensor):
        for w in encoder_kernels:
            w = w.detach().cpu()  # (C, out_per, K)
            C, out_per, k_size = w.shape
            for c in range(C):
                for d in range(out_per):
                    dendrite_kernels.append(w[c, d].numpy())

    elif isinstance(encoder_kernels, torch.Tensor):
        # erwarten Form (C, D, K)
        w = encoder_kernels.detach().cpu()
        if w.ndim != 3:
            raise ValueError(f"Expected encoder_kernels as (C, D, K) tensor, got shape {w.shape}")
        C, D, k_size = w.shape
        for c in range(C):
            for d in range(D):
                dendrite_kernels.append(w[c, d].numpy())
    else:
        raise TypeError(f"Unsupported type for encoder_kernels: {type(encoder_kernels)}")



    plt.show()


def plot_encoder_kernels_as_tables(model: ChannelLatencySeq2Value, decimals=4):
    """
    Plottet die finalen Encoder-Kernelgewichte als Tabellen.

    Layout:
      - Zeilen: Channels
      - Spalten: kernel_specs
    In jedem Subplot:
      - Eine Tabelle (out_per x kernel_size) NUR mit Zahlen.
        -> KEINE rowLabels, KEINE colLabels.
    """


    encoder = model.encoder
    C = encoder.channels
    specs = encoder.kernel_specs
    convs = encoder.convs
    num_specs = len(specs)

    # Figure vorbereiten
    fig, axes = plt.subplots(C, num_specs, figsize=(4 * num_specs, 2.5 * C))

    # Ensure axes is always 2D
    if C == 1 and num_specs == 1:
        axes = np.array([[axes]])
    elif C == 1:
        axes = np.expand_dims(axes, axis=0)
    elif num_specs == 1:
        axes = np.expand_dims(axes, axis=1)
    else:
        axes = np.array(axes)

    for spec_idx, ((k, out_per), conv) in enumerate(zip(specs, convs)):
        # Extract weights (C, out_per, k)
        w = conv.weight.detach().cpu().numpy().reshape(C, out_per, k)

        for c in range(C):
            ax = axes[c, spec_idx]
            ax.axis("off")

            kernels = w[c]  # (out_per, k)
            data = np.round(kernels, decimals=decimals)

            # Tabelle OHNE Labels erstellen
            table = ax.table(
                cellText=data,
                loc="center"
            )
            table.scale(1.2, 1.2)

            ax.set_title(f"Channel {c}, k={k}, {out_per} Filter", fontsize=9)

    plt.suptitle("Finale Encoder-Kernel pro Channel und Kernelgröße", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_input_channels(inp, sample_idx=0):
    """
    Plottet die drei Input-Kanäle X, Y, Z in drei Subplots untereinander.

    Parameter:
    - inp: Tensor der Form (N, 3, T)
    - sample_idx: welches Sample geplottet werden soll (Standard: 0)
    """
    import matplotlib.pyplot as plt

    data = inp[sample_idx].detach().cpu()  # (3, T)
    names = ["X (Channel 0)", "Y (Channel 1)", "Z (Channel 2)"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i in range(3):
        axes[i].plot(data[i])
        axes[i].set_title(names[i])
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time")
    plt.suptitle(f"Input Data (Sample {sample_idx})")
    plt.tight_layout()
    plt.show()

def plot_encoder_kernels_as_curves(model: ChannelLatencySeq2Value):
    """
    Plottet die finalen Encoder-Kernelgewichte als Kurven.

    Layout:
      - Zeilen: Channels
      - Spalten: kernel_specs
    In jedem Subplot:
      - x-Achse: Kernel-Index (0..k-1)
      - y-Achse: Gewicht
      - eine Kurve pro Filter (out_per)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    encoder = model.encoder
    C = encoder.channels
    specs = encoder.kernel_specs
    convs = encoder.convs
    num_specs = len(specs)

    fig, axes = plt.subplots(C, num_specs, figsize=(4 * num_specs, 3 * C), squeeze=False)

    for spec_idx, ((k, out_per), conv) in enumerate(zip(specs, convs)):
        # conv.weight: (C*out_per, 1, k) -> (C, out_per, k)
        w = conv.weight.detach().cpu().numpy().reshape(C, out_per, k)
        x = np.arange(k)

        for c in range(C):
            ax = axes[c, spec_idx]
            ax.set_title(f"Channel {c}, k={k}", fontsize=9)

            for f_idx in range(out_per):
                kernel_vals = w[c, f_idx]  # (k,)
                ax.plot(x, kernel_vals, marker="o", linewidth=1, alpha=0.9, label=f"f{f_idx}")

            ax.set_xlabel("Kernel index")
            ax.set_ylabel("Weight")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

    plt.suptitle("Encoder-Kernel als Kurven pro Channel und Kernelgröße", fontsize=14)
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
    X = torch.cos(t.unsqueeze(0) * 0.5).unsqueeze(0).repeat(N, 1, 1)*torch.sin(t.unsqueeze(0) * 0.2).unsqueeze(0).repeat(N, 1, 1) + 0.01 * torch.randn(N, 1, T)
    Y = 0.5 * F.pad(X[:, 0, :-1], (1, 0)) + 0.35 * torch.randn(N, T)  # more noisy so longer latency
    Y = Y.unsqueeze(1)
    Z = 0.3 * F.pad(X[:, 0, :-1], (1, 0)) + 0.4 * F.pad(Y[:, 0, :-1], (1, 0)) + 0.25 * torch.randn(N, T)
    Z = Z.unsqueeze(1)
    data = torch.cat([X, Y, Z], dim=1)
    inp = data  # supervised target is derived from the same multichannel sequence

    # model
    model = ChannelLatencySeq2Value(channels=C, kernel_specs=[(3,6),(5,6),(9,6)], lif_tau=5.0, lif_threshold=0.01)
    model.to(device)
    model.zero_output_diagonal_()

    # training hyperparams
    epochs = 100
    batch_size = 128
    lr = 0.003
    l1_weight = 0 #1e-5         # sparsity coefficient (lambda)
    beta_cycle = 0.3     # cycle suppression coefficient (beta)
    gamma_dom = 0.0          # dominance coefficient (optional)
    epsilon_dom = 1.0        # small threshold for dominance term (unused if gamma_dom==0)
    proximal_lambda = 0
    stdp_lr = 0
    adjacency_threshold = 0.05
    order_consistency_alpha = 0.0

    # data loader
    dataset = TensorDataset(inp)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # optimizer excludes output_gates (updated by STDP)
    opt = torch.optim.Adam([p for p in model.parameters()], lr=lr)
    #opt = torch.optim.Adam([p for p in model.parameters() if p is not model.output_gates], lr=lr)
    gate_history = []
    history = {
        'train_loss': [],
        'val_loss': [],  # falls du das schon nutzt

        'train_latency_loss': [],
        'train_order_loss': [],
        'train_sparsity_loss': [],  # L1
        'train_cycle_loss': [],
        'train_dom_loss': [],
        'train_stdp_update': [],  # STDP-Metrik
    }

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        total_latency_loss = 0.0
        total_order_loss = 0.0
        total_l1 = 0.0
        total_cycle_loss = 0.0
        total_dom_loss = 0.0
        total_stdp_update = 0.0

        n_examples = 0

        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()

            pred_lat, true_lat, act, dendrites = model(xb)

            # 1) supervised latency reconstruction (L1)
            latency_loss = torch.mean(torch.abs(pred_lat - true_lat))

            # 2) optional ordering consistency loss
            order_loss = 0.0
            if order_consistency_alpha > 0.0:
                pred_diff = pred_lat.unsqueeze(1) - pred_lat.unsqueeze(2)
                true_diff = true_lat.unsqueeze(1) - true_lat.unsqueeze(2)
                order_loss = torch.mean(torch.relu(-pred_diff * torch.sign(true_diff)))
            # du kannst hier auch "order_loss_eff = order_consistency_alpha * order_loss" loggen,
            # wenn du den tatsächlichen Beitrag im Gesamt-Loss sehen willst.

            # 3) sparsity (L1) on gates
            l1 = l1_weight * torch.sum(torch.abs(model.output_gates))

            # 4) cycle suppression
            G = model.output_gates
            Cc = G.shape[0]
            eye = torch.eye(Cc, device=G.device)
            mask = 1.0 - eye
            cycle_loss = beta_cycle * 0.5 * torch.sum(torch.abs((G * G.t()) * mask))

            # 5) dominance penalty
            dom_loss = 0.0
            if gamma_dom > 0.0:
                incoming = torch.sum(torch.abs(G), dim=1)
                dom_loss = gamma_dom * torch.sum(torch.relu(incoming - epsilon_dom))

            # total-Loss
            loss = latency_loss + order_consistency_alpha * order_loss + l1 + cycle_loss + dom_loss
            loss.backward()
            opt.step()

            # proximal shrinkage auf Gates (wie gehabt)
            if proximal_lambda > 0.0:
                with torch.no_grad():
                    param = model.output_gates
                    param.copy_(torch.sign(param) * torch.clamp(torch.abs(param) - proximal_lambda, min=0.0))
                    model.zero_output_diagonal_()

            # 6) STDP update (hier nehmen wir an, dass die Funktion eine Skalar-Metrik zurückgibt)
            stdp_metric = stdp_update_from_latencies(model, true_lat, stdp_lr=stdp_lr)
            # falls deine Funktion aktuell nichts zurückgibt, einfach dort z.B. die Summe der
            # absoluten Gewichtsänderungen berechnen und zurückgeben.
            if stdp_metric is None:
                stdp_metric = 0.0
            gate_history.append(model.output_gates.detach().cpu().numpy().copy())

            batch_size = xb.shape[0]
            total_loss += loss.item() * batch_size
            total_latency_loss += latency_loss.item() * batch_size
            # hier logge ich den "rohen" order_loss, nicht alpha*order_loss:
            total_order_loss += (order_loss if isinstance(order_loss, float) else order_loss.item()) * batch_size
            total_l1 += l1.item() * batch_size
            total_cycle_loss += cycle_loss.item() * batch_size
            total_dom_loss += (dom_loss if isinstance(dom_loss, float) else dom_loss.item()) * batch_size
            total_stdp_update += float(stdp_metric) * batch_size

            n_examples += batch_size

        # Epochen-Mittelwerte
        avg_loss = total_loss / n_examples
        avg_latency_loss = total_latency_loss / n_examples
        avg_order_loss = total_order_loss / n_examples
        avg_l1 = total_l1 / n_examples
        avg_cycle_loss = total_cycle_loss / n_examples
        avg_dom_loss = total_dom_loss / n_examples
        avg_stdp_update = total_stdp_update / n_examples

        history['train_loss'].append(avg_loss)
        history['train_latency_loss'].append(avg_latency_loss)
        history['train_order_loss'].append(avg_order_loss)
        history['train_sparsity_loss'].append(avg_l1)
        history['train_cycle_loss'].append(avg_cycle_loss)
        history['train_dom_loss'].append(avg_dom_loss)
        history['train_stdp_update'].append(avg_stdp_update)
        # simple val using the whole dataset (same data here)
        model.eval()
        with torch.no_grad():
            x_val = inp.to(device)
            p_lat, t_lat, _, _ = model(x_val)
            val_loss = torch.mean(torch.abs(p_lat - t_lat)).item()
            history['val_loss'].append(val_loss)

        if (ep + 1) % max(1, epochs // 10) == 0 or ep < 5:
            print(
                f"Epoch {ep + 1}/{epochs}  "
                f"train_loss={avg_loss:.6f}  "
                f"latency={avg_latency_loss:.6f}  "
                f"order={avg_order_loss:.6f}  "
                f"l1={avg_l1:.6f}  "
                f"cycle={avg_cycle_loss:.6f}  "
                f"dom={avg_dom_loss:.6f}  "
                f"stdp={avg_stdp_update:.6f}"
                + (f"  val_loss={val_loss:.6f}" if 'val_loss' in locals() or 'val_loss' in history else "")
            )

    # ------------------------
    # Post-training: adjacency and dendritic attribution example
    # ------------------------
    adj, gate = extract_channel_adjacency(model, threshold=adjacency_threshold)
    print('Learned adjacency (thresholded):')
    print(adj)
    #plot_loss(history)
    #plot_latencies_subplots(pred_lat.detach(), true_lat.detach())
    #plot_adjacency(adj, title='Inferred gates (thresholded)')
    plot_output_gate_trajectories_batches(gate_history)

    # compute dendritic causal strength for each discovered edge using a small batch
    x_val = inp[:64].to(device)
    with torch.no_grad():
        p_lat, t_lat, act, dendrites = model(x_val)
    # Show relevance heatmap and kernel plots (concatenated C*D)
    '''plot_relevant_kernels(
        model,
        encoder_kernels=list(model.encoder.convs),  # ModuleList -> echte Python-Liste
        spike_batch_label="Validation Batch"
    )'''
    plot_loss_components(history)
    # optional: plot predicted vs true latencies for first 64 examples
    plot_latencies_subplots(p_lat.detach(), t_lat.detach())
    #plot_encoder_kernels_as_tables(model)
    plot_encoder_kernels_as_curves(model)
    plot_input_channels(inp)


