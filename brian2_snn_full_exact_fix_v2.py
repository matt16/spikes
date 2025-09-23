# brian2_snn_full_exact_fix_v2.py
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

# ---------- Config ----------
prefs.codegen.target = 'numpy'
defaultclock.dt = 0.1*ms
SEED = 0
np.random.seed(SEED); seed(SEED)

N_in, N_h1, N_h2, N_out = 3, 5, 4, 2

tau_m   = 50.0*ms
tau_syn = 5.0*ms
V_th    = 0.5
V_reset = 0.0
t_ref   = 2.0*ms

tau_pre_in_h1, tau_post_in_h1 = 5.0*ms, 30.0*ms
tau_pre_h1_h2, tau_post_h1_h2 = 5.0*ms, 30.0*ms
tau_pre_h2_out, tau_post_h2_out = 5.0*ms, 50.0*ms

tau_out = 50.0*ms
encoding_window = 10.0*ms
T = 40
inhib_h1 = 0.0
inhib_h2 = 0.0

eta_in_h1  = 0.0 * 1e-3
eta_h1_h2  = 0.0 * 1e-3
eta_h2_out = 0.0 * 1e-3
oja_coeff  = 0.0

EPOCHS = 2

# ---------- Data ----------
rng = np.random.default_rng(SEED)
X_epoch = rng.random((T, N_in)).astype(np.float32)
Y_epoch = rng.random((T, N_out)).astype(np.float32)
A = TimedArray(X_epoch, dt=encoding_window, name='A')

# ---------- Populations ----------
lif_eqs = Equations('''
dIsyn/dt = -Isyn/tau_syn : 1
dV/dt    = (-V + Isyn)/tau_m : 1 (unless refractory)
gain : 1
''')

h1 = NeuronGroup(N_h1, lif_eqs, threshold='V>V_th', reset='V=V_reset',
                 refractory=t_ref, method='euler', name='h1')
h2 = NeuronGroup(N_h2, lif_eqs, threshold='V>V_th', reset='V=V_reset',
                 refractory=t_ref, method='euler', name='h2')

out_eqs = Equations('''
dIout/dt = -Iout/tau_out : 1
''')
out = NeuronGroup(N_out, out_eqs, method='euler', name='out')

for G in (h1, h2):
    G.V = V_reset; G.Isyn = 0.0; G.gain = 1.0
out.Iout = 0.0

# ---------- Inputs ----------
times = np.repeat(np.arange(T), N_in) * float(encoding_window/ms) * ms
indices = np.tile(np.arange(N_in), T)
P_in = SpikeGeneratorGroup(N_in, indices=indices, times=times, name='in')

# ---------- Synapses: in->h1 & h1->h2 ----------
syn_model_in_h1 = '''
w     : 1
dxpre/dt = -xpre/tau_pre_in_h1 : 1 (event-driven)
dypos/dt = -ypos/tau_post_in_h1 : 1 (event-driven)
tpre : second
tpos : second
'''

S_in_h1 = Synapses(
    P_in, h1,
    model=syn_model_in_h1,
    on_pre='xpre += A(t, i_pre); tpre = t; Isyn_post += w * xpre',
    on_post='ypos += 1.0; tpos = t; w += eta_in_h1 * ((1 - 2*(tpre > tpos)) * (xpre * ypos) - oja_coeff * (ypos*ypos) * w)',
    method='euler',
    name='S_in_h1'
)
S_in_h1.connect(p=1.0)
S_in_h1.w = '0.5 + 0.5*rand()'
S_in_h1.xpre = 0.0; S_in_h1.ypos = 0.0
S_in_h1.tpre = -1e9*ms; S_in_h1.tpos = -1e9*ms

syn_model_h1_h2 = '''
w     : 1
dxpre/dt = -xpre/tau_pre_h1_h2 : 1 (event-driven)
dypos/dt = -ypos/tau_post_h1_h2 : 1 (event-driven)
tpre : second
tpos : second
'''
S_h1_h2 = Synapses(
    h1, h2,
    model=syn_model_h1_h2,
    on_pre='xpre += 1.0; tpre = t; Isyn_post += w * xpre',
    on_post='ypos += 1.0; tpos = t; w += eta_h1_h2 * ((1 - 2*(tpre > tpos)) * (xpre * ypos) - oja_coeff * (ypos*ypos) * w)',
    method='euler',
    name='S_h1_h2'
)
S_h1_h2.connect(p=1.0)
S_h1_h2.w = '0.2 + 0.3*rand()'
S_h1_h2.xpre = 0.0; S_h1_h2.ypos = 0.0
S_h1_h2.tpre = -1e9*ms; S_h1_h2.tpos = -1e9*ms

# ---------- Synapses: h2->out ----------
syn_model_h2_out = '''
w     : 1
dxpre/dt = -xpre/tau_pre_h2_out : 1 (event-driven)
tpre : second
'''
S_h2_out = Synapses(
    h2, out,
    model=syn_model_h2_out,
    on_pre='xpre += 1.0; tpre = t; Iout_post += w * xpre',
    method='euler',
    name='S_h2_out'
)
S_h2_out.connect(p=1.0)
S_h2_out.w = '0.5 + 0.5*rand()'
S_h2_out.xpre = 0.0; S_h2_out.tpre = -1e9*ms

# ---------- Monitors ----------
Msp_in = SpikeMonitor(P_in, name='Msp_in')
Msp_h1 = SpikeMonitor(h1, name='Msp_h1')
Msp_h2 = SpikeMonitor(h2, name='Msp_h2')
Mout   = StateMonitor(out, 'Iout', record=True, name='Mout')

# ---------- Optional divisive inhibition ----------
@network_operation(dt=defaultclock.dt)
def divisive_inhibition():
    if inhib_h1 > 0.0:
        total = float(np.sum(h1.Isyn[:]))
        if total > 0.0:
            h1.Isyn[:] = h1.Isyn[:] / (1.0 + inhib_h1 * total)
    if inhib_h2 > 0.0:
        total = float(np.sum(h2.Isyn[:]))
        if total > 0.0:
            h2.Isyn[:] = h2.Isyn[:] / (1.0 + inhib_h2 * total)

# ---------- Training Loop ----------
loss_log = []
hebb_log = []

def epoch_run(X_ta, Y, epochs=1):
    # expose STDP constants
    S_in_h1.namespace.update(dict(eta_in_h1=eta_in_h1, oja_coeff=oja_coeff))
    S_h1_h2.namespace.update(dict(eta_h1_h2=eta_h1_h2, oja_coeff=oja_coeff))
    for ep in range(epochs):
        for G in (h1, h2):
            G.V = V_reset; G.Isyn = 0.0; G.gain = 1.0
        out.Iout = 0.0
        for S in (S_in_h1, S_h1_h2):
            S.xpre = 0.0; S.ypos = 0.0; S.tpre = -1e9*ms; S.tpos = -1e9*ms
        S_h2_out.xpre = 0.0; S_h2_out.tpre = -1e9*ms

        prev_W_in_h1 = S_in_h1.w[:].copy()
        prev_W_h1_h2 = S_h1_h2.w[:].copy()

        epoch_loss = 0.0

        for t_idx in range(T):
            run(encoding_window)
            y_t = Y[t_idx]
            o_t = out.Iout[:].copy()
            err = o_t - y_t
            loss = float(np.dot(err, err))
            r = float(np.exp(-loss))
            if r > 1.0: r = 1.0
            epoch_loss += loss

            if eta_h2_out != 0.0:
                pre = S_h2_out.xpre[:].copy()
                i_idx = S_h2_out.i[:]
                j_idx = S_h2_out.j[:]
                mask = o_t[j_idx] > 0.0
                dw = np.zeros_like(S_h2_out.w[:])
                if np.any(mask):
                    dw[mask] = eta_h2_out * r * pre[i_idx[mask]] * (y_t[j_idx[mask]] - o_t[j_idx[mask]])
                    S_h2_out.w[:] += dw

        loss_log.append(epoch_loss / float(T))
        d1 = float(np.sum(np.abs(S_in_h1.w[:] - prev_W_in_h1)))
        d2 = float(np.sum(np.abs(S_h1_h2.w[:] - prev_W_h1_h2)))
        hebb_log.append(d1 + d2)
        print(f"Epoch {ep+1}/{epochs}: Hebb-proxy={hebb_log[-1]:.6e}, Loss={loss_log[-1]:.6e}")

# ---------- Run ----------
epoch_run(X_epoch, Y_epoch, epochs=EPOCHS)

# ---------- Plots ----------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.plot(hebb_log, marker='o'); plt.title('Hebb proxy (|ΔW|)'); plt.xlabel('Epoch'); plt.grid(alpha=0.3)
plt.subplot(1,2,2); plt.plot(loss_log, marker='o'); plt.title('Loss'); plt.xlabel('Epoch'); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

plt.figure(figsize=(9,4))
plt.eventplot([Msp_in.t/ms, Msp_h1.t/ms, Msp_h2.t/ms], lineoffsets=[0,1,2])
plt.yticks([0,1,2], ['in','h1','h2']); plt.xlabel('time (ms)'); plt.title('Raster'); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
tt = np.arange(T) * float(encoding_window/ms)
for j in range(N_out):
    plt.plot(Mout.t/ms, Mout.Iout[j], label=f'out{j}')
    plt.step(tt, Y_epoch[:, j], where='post', alpha=0.6, linewidth=1.0, label=f'tgt{j}')
plt.xlabel('time (ms)'); plt.title('Output vs Target'); plt.legend(ncol=2, fontsize=8); plt.tight_layout(); plt.show()
