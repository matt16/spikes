"""
Spiking predictor (PyTorch) with visualization utilities

Additions:
- Plot functions for:
  * Loss evolution over epochs
  * Predicted vs. actual outputs
  * Inferred adjacency matrix (imshow)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# ------------------------
# Surrogate gradient spike function
# ------------------------
class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        out = (input >= threshold).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        threshold = ctx.threshold
        gamma = 0.3
        grad_input = grad_output.clone()
        surrogate_grad = gamma * torch.max(torch.zeros_like(input), 1 - torch.abs((input - threshold) / gamma))
        return grad_input * surrogate_grad, None

spike_fn = FastSigmoidSpike.apply

# ------------------------
# LIF cell
# ------------------------
class LIFCell(nn.Module):
    def __init__(self, size, tau_mem=20.0, dt=1.0, threshold=1.0, detach_reset=False):
        super().__init__()
        self.size = size
        self.tau_mem = tau_mem
        self.dt = dt
        self.alpha = math.exp(-dt / tau_mem)
        self.threshold = threshold
        self.detach_reset = detach_reset

    def forward(self, input_t, mem):
        mem = self.alpha * mem + (1 - self.alpha) * input_t
        spike = spike_fn(mem, self.threshold)
        if self.detach_reset:
            mem = mem.detach() * (1.0 - spike) + mem * spike
        else:
            mem = mem * (1.0 - spike)
        return mem, spike

# ------------------------
# Depthwise conv encoder
# ------------------------
class DepthwiseEncoder(nn.Module):
    def __init__(self, channels, kernel_size=5, out_channels_per_in=4):
        super().__init__()
        self.channels = channels
        self.out_channels_per_in = out_channels_per_in
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels * out_channels_per_in,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=channels,
        )

    def forward(self, x):
        return self.conv(x)

# ------------------------
# Spiking predictor model
# ------------------------
class SpikingPredictor(nn.Module):
    def __init__(self, channels, encoder_out_per_channel=4, hidden_size=64, kernel_size=5,
                 tau_mem=20.0, threshold=1.0, device=None):
        super().__init__()
        self.channels = channels
        self.encoder = DepthwiseEncoder(channels, kernel_size=kernel_size, out_channels_per_in=encoder_out_per_channel)
        enc_dim = channels * encoder_out_per_channel
        self.enc_to_hidden = nn.Linear(enc_dim, hidden_size)
        self.hidden_cell = LIFCell(size=hidden_size, tau_mem=tau_mem, threshold=threshold)
        self.hidden_to_out = nn.Linear(hidden_size, channels, bias=True)
        self.output_gates = nn.Parameter(torch.ones(channels, channels))
        self.device = device

    def forward(self, x, timesteps=None, return_mem=False):
        if timesteps is None:
            timesteps = x.shape[-1]
        batch = x.shape[0]
        enc = self.encoder(x)
        enc = enc.permute(2, 0, 1)

        hidden_mem = torch.zeros(batch, self.hidden_cell.size, device=enc.device)
        outs_mem_seq = []

        for t in range(timesteps):
            enc_t = enc[t]
            hid_in = self.enc_to_hidden(enc_t)
            hidden_mem, hidden_spike = self.hidden_cell(hid_in, hidden_mem)
            out_preact = self.hidden_to_out(hidden_mem)
            gate = self.output_gates * (1.0 - torch.eye(self.channels, device=enc.device))
            gated = torch.matmul(out_preact, gate.t())
            out_mem = out_preact + gated
            outs_mem_seq.append(out_mem)

        outs_mem = torch.stack(outs_mem_seq, dim=2)
        return outs_mem

    def zero_output_diagonal_(self):
        with torch.no_grad():
            diag_idx = torch.eye(self.channels, device=self.output_gates.device).bool()
            self.output_gates[diag_idx] = 0.0

# ------------------------
# Utilities
# ------------------------
def l1_penalty(model, weight=1e-4):
    l1 = 0.0
    for name, p in model.named_parameters():
        if 'hidden_to_out' in name or 'output_gates' in name:
            l1 = l1 + torch.sum(torch.abs(p))
    return weight * l1

def proximal_l1(param, lam):
    with torch.no_grad():
        param.copy_(torch.sign(param) * torch.clamp(torch.abs(param) - lam, min=0.0))

def extract_channel_adjacency(model, threshold=1e-3):
    gate = model.output_gates.detach().cpu().numpy().copy()
    np.fill_diagonal(gate, 0.0)
    adj = (np.abs(gate) >= threshold).astype(float)
    return adj, gate

# ------------------------
# Plot functions
# ------------------------
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

def plot_predictions(model, x, y_true, channel=0, timesteps=100):
    model.eval()
    with torch.no_grad():
        y_pred = model(x.to(next(model.parameters()).device)).cpu()
    plt.figure(figsize=(8, 4))
    plt.plot(y_true[0, channel, :timesteps].numpy(), label='True')
    plt.plot(y_pred[0, channel, :timesteps].numpy(), label='Predicted', linestyle='--')
    plt.title(f'Channel {channel} prediction vs. actual')
    plt.legend()
    plt.show()

def plot_adjacency(adj, title='Inferred adjacency'):
    plt.figure(figsize=(5, 4))
    plt.imshow(adj, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Connection strength')
    plt.title(title)
    plt.xlabel('Source channel')
    plt.ylabel('Target channel')
    plt.show()

# ------------------------
# Training loop
# ------------------------
def train_spiking_predictor(model, data_tensor, val_tensor=None, epochs=200, batch_size=64, lr=1e-3,
                            l1_weight=1e-4, proximal_lambda=0.0, device=None,
                            adjacency_threshold=0.1, autoreg_masking=True):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.to(device)
    model.zero_output_diagonal_()

    dataset = TensorDataset(*data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    channels = model.channels
    target_mask = torch.ones(channels, device=device)

    history = {'train_loss': [], 'val_loss': [], 'adjacency': []}

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            preds = model(xb)
            mask = target_mask.view(1, -1, 1)
            mse = F.mse_loss(preds * mask, yb * mask)
            loss = mse + l1_penalty(model, weight=l1_weight)
            loss.backward()
            opt.step()

            if proximal_lambda > 0.0:
                proximal_l1(model.hidden_to_out.weight, proximal_lambda)
                proximal_l1(model.output_gates, proximal_lambda)
                diag_idx = torch.eye(channels, device=model.output_gates.device).bool()
                model.output_gates[diag_idx] = 0.0

            total_loss += loss.item() * xb.shape[0]
            n += xb.shape[0]

        avg_loss = total_loss / n
        history['train_loss'].append(avg_loss)

        if val_tensor is not None:
            model.eval()
            with torch.no_grad():
                x_val, y_val = val_tensor
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                preds_val = model(x_val)
                val_loss = F.mse_loss(preds_val, y_val).item()
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
    N, C, T = 512, 3, 32
    t = torch.arange(T).float()
    X = torch.sin(t.unsqueeze(0) * 0.2).unsqueeze(0).repeat(N, 1, 1) + 0.01 * torch.randn(N, 1, T)
    Y = 0.5 * F.pad(X[:, 0, :-1], (1, 0)) + 0.1 * torch.randn(N, T)
    Y = Y.unsqueeze(1)
    Z = 0.3 * F.pad(X[:, 0, :-1], (1, 0)) + 0.4 * F.pad(Y[:, 0, :-1], (1, 0)) + 0.1 * torch.randn(N, T)
    Z = Z.unsqueeze(1)
    data = torch.cat([X, Y, Z], dim=1)

    inp = data[:, :, :-1]
    tgt = data[:, :, 1:]

    model = SpikingPredictor(channels=C, encoder_out_per_channel=6, hidden_size=128)
    model, history = train_spiking_predictor(model, (inp, tgt), epochs=30, batch_size=64, lr=1e-3,
                                             l1_weight=1e-5, proximal_lambda=1e-4,
                                             adjacency_threshold=0.05, autoreg_masking=True)

    adj, _ = extract_channel_adjacency(model, threshold=0.05)
    print('Learned adjacency:')
    print(adj)

    plot_loss(history)
    plot_predictions(model, inp[:1], tgt[:1], channel=0, timesteps=30)
    plot_adjacency(adj, title='Inferred channel relations')
