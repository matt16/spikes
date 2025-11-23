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
            conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels * out_per,
                kernel_size=k,
                padding=padding,
                groups=channels
            )
            self.convs.append(conv)
        # Reduce (grouped 1x1 conv) to combine per-channel dendrites into one soma drive
        self.reduce = nn.Conv1d(
            in_channels=channels * self.total_per_channel,
            out_channels=channels,
            kernel_size=1,
            groups=channels
        )

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
# Channel-latency Spiking Seq->Seq model (Reconstruction from filter database)
# ------------------------
class ChannelLatencySeq2Seq(nn.Module):
    def __init__(self, channels, kernel_specs=[(3, 6), (5, 6), (9, 6)],
                 lif_tau=5.0, lif_threshold=0.01):
        super().__init__()
        self.channels = channels
        self.encoder = DepthwiseMultiPathEncoder(channels, kernel_specs=kernel_specs)
        self.lif = LIFLatencyEncoder(tau_mem=lif_tau, threshold=lif_threshold)

        # learnable scale to map latency->importance (earlier -> larger)
        self.latency_scale = nn.Parameter(torch.tensor(5.0))

        # output gates: channels x channels (source -> target); zero diagonal enforced
        self.output_gates = nn.Parameter(torch.zeros(channels, channels))

        # filter-spezifische Gewichte pro (target, source, filter)
        D = self.encoder.total_per_channel
        self.filter_weights = nn.Parameter(
            0.01 * torch.randn(channels, channels, D)
        )

        nn.init.xavier_uniform_(self.output_gates)
        with torch.no_grad():
            self.zero_output_diagonal_()

    def zero_output_diagonal_(self):
        with torch.no_grad():
            diag_idx = torch.eye(self.channels, device=self.output_gates.device).bool()
            self.output_gates[diag_idx] = 0.0

    def _get_padded_kernels(self, T: int) -> torch.Tensor:
        """
        Baut aus den Conv-Kernen eine "Filterdatenbank" in Form (C, D, T).
        Jeder Kernel wird auf Länge T nach rechts mit Nullen gepadded.
        """
        C = self.channels
        D_total = self.encoder.total_per_channel
        device = self.output_gates.device

        kernels_padded = torch.zeros(C, D_total, T, device=device)
        d_offset = 0
        for (k, out_per), conv in zip(self.encoder.kernel_specs, self.encoder.convs):
            # conv.weight: (C*out_per, 1, k) -> (C, out_per, k)
            w = conv.weight.view(C, out_per, k)  # keine .detach(), Gradienten sollen durchgehen
            # für jeden Channel separat in die D-Achse einsortieren
            # und in der Zeitachse auf 0..k-1 legen, Rest = 0
            kernels_padded[:, d_offset:d_offset + out_per, :k] = w
            d_offset += out_per

        return kernels_padded  # (C, D_total, T)

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, T)
        returns:
            recon:       (B, C, T)  rekonstruierte Zeitreihen pro Kanal
            true_seq:    (B, C, T)  Original-Zeitreihen (einfach x)
            true_lat:    (B, C)     LIF-Latenzen
            act:         (B, C)     Latenz-basierte Importance
        """
        B, C, T = x.shape
        drive, dendritic_drive = self.encoder(x)      # drive: (B,C,T)
        true_lat = self.lif.encode(drive)             # (B, C)

        # Latenz -> Importance: earlier spike (kleine Latenz) => großer Wert
        scale = torch.clamp(self.latency_scale, min=1e-3)
        act = torch.exp(-true_lat / scale)  # (B, C)

        # Filterdatenbank: (C, D, T)
        kernels_padded = self._get_padded_kernels(T)
        D = kernels_padded.shape[1]

        # (C, D, T) -> (C*D, T), Flatten über (source, filter)
        kernels_flat = kernels_padded.view(C * D, T)  # (C*D, T)

        # Basis-Koeffizienten: Gates * filter_weights
        # output_gates: (target=j, source=i)
        # filter_weights: (target=j, source=i, d)
        coeff_base = self.output_gates.unsqueeze(-1) * self.filter_weights  # (C, C, D)

        # Jetzt batch- und latenzenabhängig: act[b, i] moduliert alle Filter von Quelle i
        # gamma[b, j, i, d] = act[b, i] * coeff_base[j, i, d]
        gamma = coeff_base.unsqueeze(0) * act[:, None, :, None]  # (B, target, source, D)

        # Flatten (source, D) -> S = C*D
        gamma_flat = gamma.view(B, C, C * D)  # (B, target, S)

        # Rekonstruktion pro Batch, Target-Kanal, Zeit:
        # recon[b,j,t] = sum_k gamma_flat[b,j,k] * kernels_flat[k,t]
        recon = torch.einsum('bjs,st->bjt', gamma_flat, kernels_flat)  # (B, C, T)

        return recon, x, true_lat, act


# ------------------------
# STDP update function (batch-averaged)
# ------------------------
def stdp_update_from_latencies(
    model: nn.Module,
    latencies: torch.Tensor,
    stdp_lr=1e-3,
    A_plus=0.01,
    A_minus=0.012,
    tau_plus=2.0,
    tau_minus=2.0,
    clip=1.0,
):
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
        if hasattr(model, "zero_output_diagonal_"):
            model.zero_output_diagonal_()

    return stdp_metric


def extract_channel_adjacency(model, threshold=1e-3):
    gate = model.output_gates.detach().cpu().numpy().copy()
    np.fill_diagonal(gate, 0.0)
    adj = (np.abs(gate) >= threshold).astype(float)
    return adj, gate

def plot_reconstructed_Z(true_seq, recon_seq, sample_idx=0, channel_idx=2):
    """
    Plottet True vs. Reconstructed Z-Zeitreihe als Verlauf (ein Sample, ein Kanal).
    true_seq, recon_seq: (B, C, T)
    """
    true_seq = true_seq.detach().cpu()
    recon_seq = recon_seq.detach().cpu()

    t = np.arange(true_seq.shape[-1])

    plt.figure(figsize=(8, 4))
    plt.plot(t, true_seq[sample_idx, channel_idx], label="True Z", linewidth=2)
    plt.plot(t, recon_seq[sample_idx, channel_idx], label="Reconstructed Z", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(f"True vs Reconstructed Z (sample {sample_idx}, channel {channel_idx})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------
# MAIN: simplified training loop with reconstruction loss
# ------------------------
if __name__ == '__main__':
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # toy data
    N, C, T = 512, 3, 64
    t = torch.arange(T).float()
    X = torch.cos(t.unsqueeze(0) * 0.5).unsqueeze(0).repeat(N, 1, 1) * \
        torch.sin(t.unsqueeze(0) * 0.2).unsqueeze(0).repeat(N, 1, 1) + \
        0.01 * torch.randn(N, 1, T)
    Y = 0.5 * F.pad(X[:, 0, :-1], (1, 0)) + 0.35 * torch.randn(N, T)  # more noisy so longer latency
    Y = Y.unsqueeze(1)
    Z = 0.3 * F.pad(X[:, 0, :-1], (1, 0)) + 0.4 * F.pad(Y[:, 0, :-1], (1, 0)) + 0.25 * torch.randn(N, T)
    Z = Z.unsqueeze(1)
    data = torch.cat([X, Y, Z], dim=1)
    inp = data  # supervised target is derived from the same multichannel sequence

    # model: neue Klasse
    model = ChannelLatencySeq2Seq(
        channels=C,
        kernel_specs=[(3, 6), (5, 6), (9, 6)],
        lif_tau=5.0,
        lif_threshold=0.01
    )
    model.to(device)
    model.zero_output_diagonal_()

    # training hyperparams
    epochs = 100
    batch_size = 128
    lr = 0.003
    l1_weight = 0       # sparsity coefficient (lambda)
    beta_cycle = 0.3    # cycle suppression coefficient (beta)
    gamma_dom = 0.0     # dominance coefficient (optional)
    epsilon_dom = 1.0   # small threshold for dominance term (unused if gamma_dom==0)
    proximal_lambda = 0
    stdp_lr = 0
    adjacency_threshold = 0.05
    order_consistency_alpha = 0.0  # bleibt, aber wird jetzt nicht genutzt

    # data loader
    dataset = TensorDataset(inp)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam([p for p in model.parameters()], lr=lr)
    gate_history = []
    history = {
        'train_loss': [],
        'val_loss': [],

        'train_latency_loss': [],   # hier jetzt: Rekonstruktions-Loss
        'train_order_loss': [],
        'train_sparsity_loss': [],
        'train_cycle_loss': [],
        'train_dom_loss': [],
        'train_stdp_update': [],
    }

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_order_loss = 0.0
        total_l1 = 0.0
        total_cycle_loss = 0.0
        total_dom_loss = 0.0
        total_stdp_update = 0.0

        n_examples = 0

        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()

            # neues Forward: Rekonstruktion + Latenzen
            recon, true_seq, true_lat, act = model(xb)  # (B,C,T), (B,C,T), (B,C), (B,C)

            # Rekonstruktions-Loss:
            # MSE über Zeit, dann mit Importance (exp(-lat/scale)) gewichtet
            scale = torch.clamp(model.latency_scale, min=1e-3)
            importance = torch.exp(-true_lat / scale).detach()  # detach: Importance als Gewicht, nicht als Opt-Ziel

            se = (recon - true_seq) ** 2        # (B, C, T)
            mse_per_bc = se.mean(dim=2)         # (B, C)
            weighted_mse = importance * mse_per_bc
            recon_loss = weighted_mse.mean()    # Skalar

            # optional: wir könnten auch nur Z (channel 2) in den Loss nehmen:
            # recon_loss = weighted_mse[:, 2].mean()

            latency_loss = recon_loss  # für die alten History-Keys

            # order_loss = 0 (wird nicht benutzt)
            order_loss = 0.0

            # sparsity (L1) on gates
            l1 = l1_weight * torch.sum(torch.abs(model.output_gates))

            # cycle suppression
            G = model.output_gates
            Cc = G.shape[0]
            eye = torch.eye(Cc, device=G.device)
            mask = 1.0 - eye
            cycle_loss = beta_cycle * 0.5 * torch.sum(torch.abs((G * G.t()) * mask))

            # dominance penalty
            dom_loss = 0.0
            if gamma_dom > 0.0:
                incoming = torch.sum(torch.abs(G), dim=1)
                dom_loss = gamma_dom * torch.sum(torch.relu(incoming - epsilon_dom))

            loss = latency_loss + l1 + cycle_loss + dom_loss
            loss.backward()
            opt.step()

            # proximal shrinkage auf Gates (wie gehabt)
            if proximal_lambda > 0.0:
                with torch.no_grad():
                    param = model.output_gates
                    param.copy_(
                        torch.sign(param) * torch.clamp(torch.abs(param) - proximal_lambda, min=0.0)
                    )
                    model.zero_output_diagonal_()

            # STDP update
            stdp_metric = stdp_update_from_latencies(model, true_lat, stdp_lr=stdp_lr)
            if stdp_metric is None:
                stdp_metric = 0.0
            gate_history.append(model.output_gates.detach().cpu().numpy().copy())

            bs = xb.shape[0]
            total_loss += loss.item() * bs
            total_recon_loss += recon_loss.item() * bs
            total_order_loss += float(order_loss) * bs
            total_l1 += l1.item() * bs
            total_cycle_loss += cycle_loss.item() * bs
            total_dom_loss += (dom_loss if isinstance(dom_loss, float) else dom_loss.item()) * bs
            total_stdp_update += float(stdp_metric) * bs

            n_examples += bs

        # Epochen-Mittelwerte
        avg_loss = total_loss / n_examples
        avg_recon_loss = total_recon_loss / n_examples
        avg_order_loss = total_order_loss / n_examples
        avg_l1 = total_l1 / n_examples
        avg_cycle_loss = total_cycle_loss / n_examples
        avg_dom_loss = total_dom_loss / n_examples
        avg_stdp_update = total_stdp_update / n_examples

        history['train_loss'].append(avg_loss)
        history['train_latency_loss'].append(avg_recon_loss)
        history['train_order_loss'].append(avg_order_loss)
        history['train_sparsity_loss'].append(avg_l1)
        history['train_cycle_loss'].append(avg_cycle_loss)
        history['train_dom_loss'].append(avg_dom_loss)
        history['train_stdp_update'].append(avg_stdp_update)

        # simple "val" auf vollem Datensatz
        model.eval()
        with torch.no_grad():
            x_val = inp.to(device)
            recon_v, true_v, lat_v, act_v = model(x_val)
            scale_v = torch.clamp(model.latency_scale, min=1e-3)
            importance_v = torch.exp(-lat_v / scale_v).detach()
            se_v = (recon_v - true_v) ** 2
            mse_v = se_v.mean(dim=2)
            weighted_mse_v = importance_v * mse_v
            val_loss = weighted_mse_v.mean().item()
        history['val_loss'].append(val_loss)

        if (ep + 1) % max(1, epochs // 10) == 0 or ep < 5:
            print(
                f"Epoch {ep + 1}/{epochs}  "
                f"train_loss={avg_loss:.6f}  "
                f"recon={avg_recon_loss:.6f}  "
                f"l1={avg_l1:.6f}  "
                f"cycle={avg_cycle_loss:.6f}  "
                f"dom={avg_dom_loss:.6f}  "
                f"stdp={avg_stdp_update:.6f}  "
                f"val_loss={val_loss:.6f}"
            )

    # ------------------------
    # Post-training: adjacency and plots
    # ------------------------
    adj, gate = extract_channel_adjacency(model, threshold=adjacency_threshold)
    print('Learned adjacency (thresholded):')
    print(adj)


    # Rekonstruktion auf einem Teil-Batch anschauen
    x_val = inp[:64].to(device)
    with torch.no_grad():
        recon_v, true_v, lat_v, act_v = model(x_val)

    # Hier dein gewünschter Plot: nur true vs. rekonstruierte Z-Zeitreihe (Verlauf)
    plot_reconstructed_Z(true_v, recon_v, sample_idx=0, channel_idx=2)

    # Optional: Encoder-Kernel anschauen + Input-Kanäle plotten
    