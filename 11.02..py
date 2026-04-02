import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, theta=1.0):
        ctx.save_for_backward(u)
        return (u > theta).float()

    @staticmethod
    def backward(ctx, grad_output):
        (u,) = ctx.saved_tensors
        # triangular surrogate
        grad = (1.0 - u.abs()).clamp(min=0.0)
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
        # slow synaptic trace
        self.trace = self.tau_trace * self.trace + input_current

        # membrane
        self.u = self.tau_mem * self.u + self.trace

        s = spike_fn(self.u - self.theta)

        # soft reset
        self.u = self.u * (1.0 - s)

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
        self.u = self.tau_mem * self.u + input_current
        s = spike_fn(self.u - self.theta)
        self.u = self.u * (1.0 - s)
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
        """
        x_window: [batch, window_size]
        """
        spikes = []

        for k in range(self.n_filters):
            proj = (x_window * self.weights[k]).sum(dim=-1)

            inh_spike = self.inh[k](proj)
            gated_input = proj * (1.0 - inh_spike)  # veto

            s = self.exc[k](gated_input)
            spikes.append(s)

        return torch.stack(spikes, dim=1)


class SNN101(nn.Module):
    def __init__(self, n_filters, window_size, n_classes):
        super().__init__()
        self.bank = SNNFilterBank(n_filters, window_size)
        self.decoder = nn.Linear(n_filters, n_classes)

    def reset(self, batch_size, device):
        self.bank.reset(batch_size, device)

    def forward(self, x_windows):
        """
        x_windows: [time, batch, window_size]
        """
        spike_counts = 0

        for t in range(x_windows.shape[0]):
            s = self.bank(x_windows[t])
            spike_counts = spike_counts + s

        return self.decoder(spike_counts)


if __name__ == "__main__":

    model = SNN101(n_filters=16, window_size=20, n_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for x_windows, y in dataloader:
        model.reset(batch_size=x_windows.shape[1], device=x_windows.device)

        logits = model(x_windows)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()