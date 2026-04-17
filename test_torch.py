import numpy as np

# -------------------------------------------------------------
# This code demonstrates a *hierarchical dendritic neuron*
# using ONLY biologically plausible neuromorphic computations:
#
#   ✓ Spike timing → EPSPs (not arithmetic on timestamps)
#   ✓ Dendritic nonlinearities (NMDA-like thresholds)
#   ✓ Branch hierarchy (Level 1 → Level 2 → Soma)
#   ✓ LIF soma producing output spike time
#
# NOTE: We do NOT compute linear combinations of spike times.
# All timing interactions happen through delays, EPSPs, nonlinearities,
# and thresholding — which is how real neurons combine timing.
# -------------------------------------------------------------
# ---- Simulation time ----
T = 200
dt = 1.0
steps = int(T/dt)

rng = np.random.default_rng(0)


# =============================================================
# LEVEL 1 — Synaptic branches
# Each synapse generates EPSPs based on spike times.
#
# Neuromorphic method:
# --------------------
# A spike at time t_i produces an EPSP(t) = exp(-(t - t_i)/tau)
# This is the ONLY allowed way to “combine” spike times.
#
# We NEVER compute: predicted_t = a*t1 + b*t2
# Instead, the membrane potential integrates EPSPs over time,
# and the *dynamics* determine the final spike time.
# =============================================================

n_level1 = 6
syn_per_l1 = 4

# Random spike trains: shape [branches, synapses, time]
level1_spikes = rng.random((n_level1, syn_per_l1, steps)) < 0.03
W1 = rng.uniform(0.5, 1.5, (n_level1, syn_per_l1))

tau_epsp = 10.0

def level1_output(branch_idx, t):
    """Sum EPSPs from all synapses in this branch at time t."""
    I = 0.0
    #epsp_sum = 0.0
    for s in range(syn_per_l1):
        for tk in spike_times[branch_idx][s]:
            if t >= tk:
                I += W1[branch_idx, s] * np.exp((s - tk) / tau_epsp)
    return I
    
# =============================================================
# LEVEL 2 — Dendritic subunits
#
# Neuromorphic method:
# --------------------
# Each subunit combines *two* Level 1 branches.
# After linear combination, it applies a dendritic nonlinearity:
#
#   output = max(0, x - threshold)^2
#
# This mimics:
#   - NMDA plateau activation
#   - dendritic coincidence detection
#   - local nonlinear computation inside branches
#
# Again: no arithmetic on spike times — only dynamics.
# =============================================================

n_level2 = 3
pairs = [(0,1), (2,3), (4,5)]  # defines hierarchy
W2 = rng.uniform(0.8, 1.2, (n_level2, 2))

def level2_output(subunit_idx, t):
    """NMDA-like dendritic nonlinearity combining two level-1 branches."""
    i, j = pairs[subunit_idx]

    # Local dendritic summation (biological)
    x = (
        W2[subunit_idx, 0] * level1_output(i, t) +
        W2[subunit_idx, 1] * level1_output(j, t)
    )

    # Nonlinear dendritic plateau (NMDA-like)
    return max(0, x - 2.0) ** 2


# =============================================================
# SOMA — LIF spike generator
#
# Neuromorphic method:
# ---------------------
# The soma integrates ALL dendritic subunit outputs.
# It generates ONE spike time when V crosses threshold.
#
# This is the ONLY “output time computation.”
# We never compute the spike time explicitly.
# The time of threshold crossing emerges naturally from:
#   - synaptic timing
#   - dendritic processing
#   - membrane integration
# =============================================================

V = 0.0
V_th = 1.0
V_reset = 0.0
tau_LIF = 20.0

V_trace = []
spike_times = []

for t in range(steps):

    # ----- Combine Level 2 subunits -----
    # This is *not* a linear combination of spike times.
    # It's summation of analog branch outputs at time t.
    soma_input = sum(level2_output(k, t) for k in range(n_level2))

    # ----- LIF membrane update -----
    dV = (-V + soma_input) / tau_LIF
    V += dV

    # ----- Check for spike -----
    if V >= V_th:
        spike_times.append(t * dt)
        V = V_reset

    V_trace.append(V)


# spike_times now contains the predicted spike time(s)
spike_times
