from brian2 import *
import matplotlib.pyplot as plt



# --- Helpers: Gewicht-Monitor + Plot für PG→G ---
def attach_pg_to_g_weight_monitor(S_pg_g):
    """
    Legt einen StateMonitor auf 'w' der PG→G-Synapsen an.
    Muss VOR run(...) aufgerufen werden.
    """
    return StateMonitor(S_pg_g, 'w', record=True)

def plot_pg_to_g_weights(S_pg_g, mon_w):
    """
    Plottet die Gewichte aller PG→G-Verbindungen über die Zeit.
    Jede Kurve ist mit 'pre->post' gelabelt.
    Zusätzlich wird die Mittelkurve fett eingezeichnet.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    t_ms = mon_w.t / ms

    plt.figure(figsize=(9, 5))
    # alle Einzelsynapsen
    for k in range(len(S_pg_g.w)):
        pre = int(S_pg_g.i[k])
        post = int(S_pg_g.j[k])
        plt.plot(t_ms, mon_w.w[k], lw=1.0, alpha=0.9, label=f'{pre}->{post}')
    # Mittelwert über alle Synapsen
    mean_w = np.mean(mon_w.w, axis=0)
    plt.plot(t_ms, mean_w, lw=3, label='Mittelwert', zorder=10)

    plt.xlabel('Zeit [ms]')
    plt.ylabel('Gewicht w [a.u.]')
    plt.title('Gewichtsverlauf (PG→G, STDP)')
    # Bei vielen Synapsen wird die Legende groß – gern anpassen/auskommentieren:
    plt.legend(ncol=3, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.show()








start_scope()

# --------------Parameter-------------

# Neuronen Parameter
tau_m   = 5*ms
tau_syn = 20*ms
refr    = 5*ms
v_rest, v_reset, v_thresh = 0.2, 0.0, 0.9

# inhibitorischer PSC + divisive Gain
tau_i   = 5*ms
g_div_default = 1

# STDP-Parameter
tau_pre  = 5*ms
tau_post = 10*ms
A_plus   = 0.015
A_minus  = -0.012
w_min, w_max = 0.0, 1.5

# Laterale Inhibition
w_inh = 0.5

# --------------Gleichungen-------------

eqs = '''
dv/dt   = ( (-(v - v_rest) + I - I_i) ) / (tau_m * (1 + g_div*I_i)) : 1 (unless refractory)
dI/dt   = -I/tau_syn  : 1
dI_i/dt = -I_i/tau_i  : 1
g_div   : 1
'''

# 5 LIF-Neuronen
G = NeuronGroup(
    5, eqs,
    threshold='v > v_thresh',
    reset='v = v_reset',
    refractory=refr,
    method='euler'
)

# zweite Gruppe H (Größe 4) mit derselben Art wie G
H = NeuronGroup(
    4, eqs,
    threshold='v > v_thresh',
    reset='v = v_reset',
    refractory=refr, method='euler'
)
# Out Gruppe
O = NeuronGroup(
    2, eqs,
    threshold='v > v_thresh',
    reset='v = v_reset',
    refractory=refr, method='euler'
)

# Initialisierung
G.v     = v_rest
G.I     = 0.0
G.I_i   = 0.0
G.g_div = g_div_default

H.v     = v_rest
H.I     = 0.0
H.I_i   = 0.0
H.g_div = g_div_default

O.v     = v_rest
O.I     = 0.0
O.I_i   = 0.0
O.g_div = g_div_default

# 3 Poisson-Quellen
PG = PoissonGroup(3, rates=[100, 200, 400]*Hz)

# Synapsen Poisson -> G mit STDP
S = Synapses(
    PG, G,
    model='''
        w : 1
        dApre/dt  = -Apre/tau_pre  : 1 (event-driven)
        dApost/dt = -Apost/tau_post: 1 (event-driven)
    ''',
    on_pre='''
        I_post += w
        Apre += 1.
        w = clip(w + A_minus * Apost, w_min, w_max)
    ''',
    on_post='''
        Apost += 1.
        w = clip(w + A_plus * Apre,  w_min, w_max)
    '''
)
S.connect()
S.w = '0.2 + 0.8*rand()'
S.namespace.update(dict(tau_pre=tau_pre, tau_post=tau_post,
                        A_plus=A_plus, A_minus=A_minus,
                        w_min=w_min, w_max=w_max))
mon_w_pg_g = attach_pg_to_g_weight_monitor(S)


# Synapsen G -> H (gleiche Synapsen-Art: STDP + exzitatorischer PSC)
S_GH = Synapses(
    G, H,
    model='''
        w : 1
        dApre/dt  = -Apre/tau_pre  : 1 (event-driven)
        dApost/dt = -Apost/tau_post: 1 (event-driven)
    ''',
    on_pre='''
        I_post += w
        Apre += 1.
        w = clip(w + A_minus * Apost, w_min, w_max)
    ''',
    on_post='''
        Apost += 1.
        w = clip(w + A_plus * Apre,  w_min, w_max)
    '''
)
S_GH.connect()
S_GH.w = '0.2 + 0.8*rand()'
S_GH.namespace.update(dict(tau_pre=tau_pre, tau_post=tau_post,
                            A_plus=A_plus, A_minus=A_minus,
                            w_min=w_min, w_max=w_max))

# Laterale Inhibition G -> G (subtraktiv) – addiert auf I_i
Lat = Synapses(G, G, model='w:1', on_pre='I_i_post += w')
Lat.connect(condition='i != j')
Lat.w = w_inh


# Synapsen H -> = (gleiche Synapsen-Art: STDP + exzitatorischer PSC)
S_HO = Synapses(
    H, O,
    model='''
        w : 1
        dApre/dt  = -Apre/tau_pre  : 1 (event-driven)
        dApost/dt = -Apost/tau_post: 1 (event-driven)
    ''',
    on_pre='''
        I_post += w
        Apre += 1.
        w = clip(w + A_minus * Apost, w_min, w_max)
    ''',
    on_post='''
        Apost += 1.
        w = clip(w + A_plus * Apre,  w_min, w_max)
    '''
)
S_HO.connect()
S_HO.w = '0.2 + 0.8*rand()'
S_HO.namespace.update(dict(tau_pre=tau_pre, tau_post=tau_post,
                            A_plus=A_plus, A_minus=A_minus,
                            w_min=w_min, w_max=w_max))









# Monitore
spg    = SpikeMonitor(PG)             # Poisson
spgG   = SpikeMonitor(G)              # G spikes
spgH   = SpikeMonitor(H)              # H spikes (neu)
spgO   = SpikeMonitor(O)
stateG = StateMonitor(G, 'v', record=True)
stateH = StateMonitor(H, 'v', record=True)
stateO = StateMonitor(O, 'v', record=True)

run(500*ms)

# --- NEU: eine Figure mit 3 Subplots für alle Spike-Raster (Poisson, G, H) ---
fig, axes = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
axes[0].plot(spg.t/ms,  spg.i,  '.', markersize=3)
axes[0].set_ylabel('Poisson i')
axes[0].set_title('Spike-Raster – Poisson, G, H')

axes[1].plot(spgG.t/ms, spgG.i, '.', markersize=3)
axes[1].set_ylabel('G i')

axes[2].plot(spgH.t/ms, spgH.i, '.', markersize=3)
axes[2].set_ylabel('H i')
axes[2].set_xlabel('Zeit [ms]')

axes[3].plot(spgO.t/ms, spgO.i, '.', markersize=3)
axes[3].set_ylabel('O i')
axes[3].set_xlabel('Zeit [ms]')

plt.tight_layout()
plt.show()

# --- Plot 2: Membranpotentiale der 5 LIF-Neuronen (G) ---
plt.figure(figsize=(7, 4))
for i in range(5):
    plt.plot(stateG.t/ms, stateG.v[i], label=f'G Neuron {i}')
plt.xlabel('Zeit [ms]'); plt.ylabel('v [a.u.]')
plt.title('LIF-Gruppe G (5 Neuronen) – Membranpotentiale')
plt.legend(loc='upper right', ncol=2, fontsize=8)
plt.tight_layout()

# --- Plot 3: Membranpotentiale der 4 LIF-Neuronen (H) ---
plt.figure(figsize=(7, 4))
for i in range(4):
    plt.plot(stateH.t/ms, stateH.v[i], label=f'H Neuron {i}')
plt.xlabel('Zeit [ms]'); plt.ylabel('v [a.u.]')
plt.title('LIF-Gruppe H (4 Neuronen) – Membranpotentiale')
plt.legend(loc='upper right', ncol=2, fontsize=8)
plt.tight_layout(); plt.show()

# --- Plot 3: Membranpotentiale der 2 LIF-Neuronen (O) ---
plt.figure(figsize=(7, 4))
for i in range(2):
    plt.plot(stateO.t/ms, stateO.v[i], label=f'O Neuron {i}')
plt.xlabel('Zeit [ms]'); plt.ylabel('v [a.u.]')
plt.title('LIF-Gruppe O (2 Neuronen) – Membranpotentiale')
plt.legend(loc='upper right', ncol=2, fontsize=8)
plt.tight_layout(); plt.show()

plot_pg_to_g_weights(S, mon_w_pg_g)


print("Spikes Poisson pro Neuron:", spg.count)
print("Spikes LIF   pro Neuron (G):", spgG.count)
print("Spikes LIF   pro Neuron (H):", spgH.count)
