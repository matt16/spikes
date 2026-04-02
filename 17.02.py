import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, theta=1.0):
        # Save both u and theta so the surrogate gradient can be centered at theta
        ctx.save_for_backward(u)
        ctx.theta = theta
        return (u > theta).float()

    @staticmethod
    def backward(ctx, grad_output):
        (u,) = ctx.saved_tensors
        theta = ctx.theta

        # Triangular surrogate centered at the threshold:
        # derivative ~ max(0, 1 - |u - theta|)
        grad = (1.0 - (u - theta).abs()).clamp(min=0.0)

        # Return gradients for (u, theta)
        # We do not train theta here, so return None for theta grad.
        return grad_output * grad, None


spike_fn = SurrogateSpike.apply


class LIFNeuron(nn.Module):
    def __init__(self, tau_mem=0.9, tau_trace=0.95, theta=1.0):
        super().__init__()
        self.tau_mem = tau_mem
        self.tau_trace = tau_trace
        self.theta = theta

        self.u = None
        self.trace = None

    def reset(self, batch_size, device):
        self.u = torch.zeros(batch_size, device=device)
        self.trace = torch.zeros(batch_size, device=device)

    def forward(self, input_current):
        # input_current: [B]
        self.trace = self.tau_trace * self.trace + input_current
        self.u = self.tau_mem * self.u + self.trace

        s = spike_fn(self.u, self.theta)  # [B]
        self.u = self.u * (1.0 - s)       # reset on spike
        return s


class InhibitoryNeuron(nn.Module):
    def __init__(self, tau_mem=0.7, theta=0.5):
        super().__init__()
        self.tau_mem = tau_mem
        self.theta = theta
        self.u = None

    def reset(self, batch_size, device):
        self.u = torch.zeros(batch_size, device=device)

    def forward(self, input_current):
        # input_current: [B]
        self.u = self.tau_mem * self.u + input_current

        # BUGFIX: pass theta explicitly (or equivalently use shift + theta=0)
        s = spike_fn(self.u, self.theta)

        self.u = self.u * (1.0 - s)  # reset on spike
        return s


class SNNFilterBank(nn.Module):
    def __init__(self, n_filters, window_size):
        super().__init__()
        self.n_filters = n_filters
        self.window_size = window_size

        self.weights = nn.Parameter(torch.randn(n_filters, window_size))

        self.exc = nn.ModuleList([LIFNeuron() for _ in range(n_filters)])
        self.inh = nn.ModuleList([InhibitoryNeuron() for _ in range(n_filters)])

    def reset(self, batch_size, device):
        for e, i in zip(self.exc, self.inh):
            e.reset(batch_size, device)
            i.reset(batch_size, device)

    def forward(self, x_window):
        # x_window: [B, W]
        spikes = []
        for k in range(self.n_filters):
            proj = (x_window * self.weights[k]).sum(dim=-1)  # [B]

            inh_spike = self.inh[k](proj)                    # [B]
            gated_input = proj * (1.0 - inh_spike)           # [B]

            s = self.exc[k](gated_input)                     # [B]
            spikes.append(s)

        return torch.stack(spikes, dim=1)  # [B, K]


class SNN101(nn.Module):
    def __init__(self, n_filters, window_size, n_classes):
        super().__init__()
        self.bank = SNNFilterBank(n_filters, window_size)
        self.decoder = nn.Linear(n_filters, n_classes)

    def reset(self, batch_size, device):
        self.bank.reset(batch_size=batch_size, device=device)

    def forward(self, x_windows):
        # x_windows: [T, B, W]
        spike_counts = 0
        for t in range(x_windows.shape[0]):
            s = self.bank(x_windows[t])      # [B, K]
            spike_counts = spike_counts + s  # [B, K]
        return self.decoder(spike_counts)    # [B, C]


if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    T = 10   # time steps expected by model
    B = 64   # batch size
    W = 20   # window_size
    L = 2000 # length of each long time series (must be >> W)

    t = torch.linspace(0, 20 * torch.pi, L)

    # two deterministic sequences (class 0 vs class 1)
    s0 = torch.sin(1.0 * t)  # class 0
    s1 = torch.sin(2.0 * t)  # class 1

    # sliding windows
    X0 = s0.unfold(0, W, 1)  # [N0, W]
    X1 = s1.unfold(0, W, 1)  # [N1, W]

    # labels
    y0 = torch.zeros(X0.size(0), dtype=torch.long)
    y1 = torch.ones(X1.size(0), dtype=torch.long)

    # combine + shape to [N, T, W] (repeat window across time dimension)
    # NOTE: this repeats the same window across T steps (not Hankel sequences yet)
    X = torch.cat([X0, X1], dim=0).unsqueeze(1).repeat(1, T, 1)  # [N, T, W]
    y = torch.cat([y0, y1], dim=0)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, drop_last=True)

    model = SNN101(n_filters=16, window_size=W, n_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for x_windows, y in dataloader:
        # [B, T, W] -> [T, B, W]
        x_windows = x_windows.permute(1, 0, 2)

        model.reset(batch_size=x_windows.shape[1], device=x_windows.device)

        logits = model(x_windows)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(loss.item())