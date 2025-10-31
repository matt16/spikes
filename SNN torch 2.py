import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import snntorch as snn
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -----------------------------
# 1️⃣ Surrogate gradient spike function
# -----------------------------
class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh=1.0):
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        return (input >= thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        thresh = ctx.thresh
        # triangular surrogate gradient
        grad_input = grad_output * torch.clamp(1.0 - torch.abs(input - thresh), min=0.0, max=1.0)
        return grad_input, None

# -----------------------------
# 2️⃣ LIF neuron with explicit temporal integration
# -----------------------------
class LIFNeuron(nn.Module):
    def __init__(self, tau=20.0, dt=1.0, thresh=1.0):
        super().__init__()
        self.tau = tau
        self.dt = dt
        self.thresh = thresh

    def forward(self, x):
        # x: [batch, features, timesteps]
        batch, features, T = x.shape
        v = torch.zeros(batch, features, device=x.device)
        spikes = []
        for t in range(T):
            dv = (x[:, :, t] - v) / self.tau
            v = v + dv * self.dt
            out = SpikeFunction.apply(v, self.thresh)
            v = v * (1 - out)
            spikes.append(out)
        spikes = torch.stack(spikes, dim=2)  # [batch, features, timesteps]
        return spikes

# -----------------------------
# 3️⃣ Temporal SNN Autoencoder
# -----------------------------
class TemporalSNN_Autoencoder(nn.Module):
    def __init__(self, input_len=50, channels=2, hidden_size=16):
        super().__init__()
        self.channels = channels
        self.input_len = input_len
        self.hidden_size = hidden_size

        # Encoder
        self.encoder_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5,
            padding=2,
            groups=channels
        )
        self.encoder_lif = LIFNeuron()

        # Hidden per timestep
        self.fc_hidden = nn.Linear(channels, hidden_size)
        self.hidden_lif = LIFNeuron()

        # Spike Decoding nach §3.4: nur Linear
        self.fc_out = nn.Linear(hidden_size, channels)
        # self.out_lif = LIFNeuron()  # <- nicht mehr benutzt

    def forward(self, x):
        # x: [B, C, T]
        x_conv   = self.encoder_conv(x)          # [B, C, T]
        x_spikes = self.encoder_lif(x_conv)      # [B, C, T] (spiking hidden states inputseitig)

        # Hidden pro Zeitschritt
        h       = self.fc_hidden(x_spikes.permute(0, 2, 1))  # [B, T, H]
        h       = h.permute(0, 2, 1)                         # [B, H, T]
        h_spike = self.hidden_lif(h)                         # [B, H, T] = S_hidden

        # §3.4 Spike Decoding: Y = Linear(S_hidden), zeitlich pro Schritt
        y_t   = self.fc_out(h_spike.permute(0, 2, 1))        # [B, T, C]
        y_hat = y_t.permute(0, 2, 1)                         # [B, C, T]

        return y_hat


# -----------------------------
# 4️⃣ Training setup
# -----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
channels = 2
timesteps = 50
hidden_size = 16

# fake financial time-series
x_train = torch.randn(batch_size, channels, timesteps, device=device)

model = TemporalSNN_Autoencoder(input_len=timesteps, channels=channels, hidden_size=hidden_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# -----------------------------
# 5️⃣ Training loop
# -----------------------------
num_epochs = 100
loss_history = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, x_train)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss={loss.item():.4f}")

# -----------------------------
# 6️⃣ Plot training loss
# -----------------------------
plt.figure(figsize=(6,4))
plt.plot(range(1,num_epochs+1), loss_history,'-o')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Evolution")
plt.grid(True)
plt.show()

# -----------------------------
# 7️⃣ Plot example reconstruction
# -----------------------------
model.eval()
with torch.no_grad():
    output = model(x_train)

time = range(timesteps)
for c in range(channels):
    plt.figure(figsize=(8,3))
    plt.plot(time, x_train[0,c].cpu(), label=f"Original channel {c}")
    plt.plot(time, output[0, c].cpu(), '--', label=f"Reconstruction channel {c}")
    plt.legend()
    plt.xlabel("Timestep")
    plt.show()