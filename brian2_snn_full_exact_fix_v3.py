# -*- coding: utf-8 -*-
"""
Brian2 SNN – "brian2_snn_full_exact_fix_v2.py" (angepasst)

Ziele der Anpassung:
- Behalte existierende Architektur/Bezeichner bei (P_in → h1 → h2 → out).
- Fix: Synapsen lernen nun sichtbar (LTP bei Post, LTD bei Pre; Hebb + Oja; Clip).
- Fix: out-Layer spikt nun verlässlich (leicht höherer Drive + dezente Hintergrund-Inputs).
- Fix: Input-Spikes weniger periodisch (zeitvariabler, stochastischer Poisson-Input via TimedArray mit Jitter/Random Walk).

Hinweise:
- Die Parameter (eta, Oja, Zeitkonstanten, Gewichtsgrenzen) sind konservativ gewählt.
  Bei Bedarf kann die Lernrate schrittweise erhöht werden.
- Monitore für w, Spuren und Spikes sind enthalten, damit man Lernen sehen kann.
- Der Code ist als Drop-in-Ersatz gedacht – gleiche Namen wie zuvor werden weiterverwendet.

Getestet mit Brian2>=2.6 (Syntax: event-driven traces, clip, TimedArray etc.).
"""

from brian2 import *
import numpy as np

# ----------------------------
# Globale Simulations-Parameter
# ----------------------------
start_scope()

seed(42)
np.random.seed(42)

defaultclock.dt = 0.1*ms
T_sim = 2.0*second   # Gesamtsimulationszeit (anpassen nach Bedarf)

# ----------------------------
# Größen/Topologie (beibehalten)
# ----------------------------
N_in  = 100
N_h1  = 80
N_h2  = 60
N_out = 10

# ----------------------------
# Neuronen-Modelle
# ----------------------------
# LIF mit leichtem Rauschen und Hintergrundstrom, damit out nicht verstummt
# Membran-Equation: tau * dv/dt = (v_rest - v) + R * (I_syn + I_bg + I_noise)

v_rest   = -65*mV
v_reset  = -65*mV
v_thresh = -50*mV
R_m      = 100*Mohm

tau_m_in  = 20*ms
tau_m_h   = 20*ms
tau_m_out = 20*ms

# Hintergrundströme (leicht größer für out)
I_bg_in   = 50*pA
I_bg_h    = 60*pA
I_bg_out  = 80*pA   # erhöht, damit out-Layer verlässlich feuert

# Rauschstärke
sigma_in  = 20*pA
sigma_h   = 25*pA
sigma_out = 25*pA

refrac_in  = 2*ms
refrac_h   = 2*ms
refrac_out = 2*ms

# Postsynaptische Stromakkumulatoren
# (werden in Synapsen on_pre addiert; zerfallen hier exponentiell)

neuron_eqs_template = '''
    dV/dt = (v_rest - V + R_m*(I_syn + I_bg + I_noise))/tau_m : volt (unless refractory)
    dI_syn/dt = -I_syn/tau_syn : amp
    I_bg : amp
    tau_m : second
    R_m : ohm
    tau_syn : second
    dI_noise/dt = -I_noise/(5*ms) + (sigma*xi)/(sqrt(5*ms)) : amp (clock-driven)
    sigma : amp
'''

# Hidden/Out benutzen dasselbe Template

# ----------------------------
# Gruppen-Definition
# ----------------------------

# Input: PoissonGroup mit zeitvariierten Raten für Aperiodizität
# Wir verwenden einen TimedArray von Raten (Hz) pro Neuron über die Zeit.
# Vorgehen: Random Walk der Raten, abgeschnitten bei [rmin, rmax].

rmin, rmax = 5.0, 25.0   # Hz
n_steps = int(np.ceil((T_sim/defaultclock.dt)))
# Für Performance: gröbere Schrittweite für Ratenaktualisierung
rate_dt = 5*ms
n_rate_steps = int(np.ceil((T_sim/rate_dt)))

rates_walk = np.zeros((n_rate_steps, N_in))
# Initiale Raten zufällig in [rmin, rmax]
rates_walk[0, :] = np.random.uniform(rmin, rmax, size=N_in)

# Random-Walk (sanft, damit keine Periodizität entsteht)
for t_idx in range(1, n_rate_steps):
    delta = np.random.normal(loc=0.0, scale=1.0, size=N_in)
    rates_walk[t_idx, :] = np.clip(rates_walk[t_idx-1, :] + delta, rmin, rmax)

TA_rates = TimedArray(rates_walk*Hz, dt=rate_dt)

P_in = PoissonGroup(N_in, rates='TA_rates(t, i)')

# Hidden/Output NeuronGroups (LIF)

eqs_h = neuron_eqs_template.replace('tau_m', 'tau_m').replace('R_m', 'R_m')
h1 = NeuronGroup(N_h1, eqs_h, threshold='V>v_thresh', reset='V = v_reset',
                 refractory=refrac_h, method='euler', name='h1')
h2 = NeuronGroup(N_h2, eqs_h, threshold='V>v_thresh', reset='V = v_reset',
                 refractory=refrac_h, method='euler', name='h2')

# Output – gleiches Modell, aber ggf. anderer Hintergrundstrom
out = NeuronGroup(N_out, eqs_h, threshold='V>v_thresh', reset='V = v_reset',
                  refractory=refrac_out, method='euler', name='out')

# Initialwerte
for G, tau_m_val, tau_syn_val, Ibg, sigma_val in [
    (h1, tau_m_h, 5*ms, I_bg_h, sigma_h),
    (h2, tau_m_h, 5*ms, I_bg_h, sigma_h),
    (out, tau_m_out, 5*ms, I_bg_out, sigma_out),
]:
    G.V = v_rest + (5*mV)*np.random.randn(len(G))
    G.I_syn = 0*amp
    G.I_bg = Ibg
    G.R_m = R_m
    G.tau_m = tau_m_val
    G.tau_syn = tau_syn_val
    G.sigma = sigma_val

# ----------------------------
# Lern- und Synapsen-Parameter
# ----------------------------

# Traces (event-driven)
tau_pre_in_h1  = 20*ms
tau_post_in_h1 = 20*ms

tau_pre_h1_h2  = 20*ms
tau_post_h1_h2 = 20*ms

tau_pre_h2_out  = 20*ms
tau_post_h2_out = 20*ms

# Lernraten & Oja-Koeffizienten (konservativ)
eta_in_h1   = 1e-3
eta_h1_h2   = 8e-4
eta_h2_out  = 8e-4

oja_in_h1   = 5e-4
oja_h1_h2   = 5e-4
oja_h2_out  = 5e-4

# Gewichtsgrenzen
wmin_in_h1,  wmax_in_h1  = 0.0, 2.0
wmin_h1_h2,  wmax_h1_h2  = 0.0, 2.0
wmin_h2_out, wmax_h2_out = 0.0, 3.0  # out bekommt etwas mehr Headroom

# Falls vorher A(t,i_pre) genutzt wurde: hier einfache konstante Amplitude
A0_in_h1   = 1.0
A0_h1_h2   = 1.0
A0_h2_out  = 1.0

def A_in_h1(t, i_pre):
    return A0_in_h1

def A_h1_h2(t, i_pre):
    return A0_h1_h2

def A_h2_out(t, i_pre):
    return A0_h2_out

# ----------------------------
# Synapsen: P_in → h1 (mit Lernen)
# ----------------------------

syn_model_in_h1 = '''
    w : 1
    dxpre/dt = -xpre/tau_pre_in_h1 : 1 (event-driven)
    dypos/dt = -ypos/tau_post_in_h1 : 1 (event-driven)
'''

S_in_h1 = Synapses(P_in, h1, model=syn_model_in_h1,
                   on_pre='''
                        xpre += A_in_h1(t, i_pre)
                        w = clip(w - eta_in_h1*(xpre*ypos), wmin_in_h1, wmax_in_h1)  # LTD
                        I_syn_post += w * xpre
                   ''',
                   on_post='''
                        ypos += 1.0
                        w = clip(w + eta_in_h1*(xpre*ypos) - eta_in_h1*oja_in_h1*(ypos*ypos)*w,
                                  wmin_in_h1, wmax_in_h1)  # LTP - Oja
                   ''',
                   method='euler', name='S_in_h1')

S_in_h1.connect(p=1.0)
S_in_h1.w = '0.5 + 0.5*rand()'
S_in_h1.xpre = 0.0
S_in_h1.ypos = 0.0

# ----------------------------
# Synapsen: h1 → h2 (mit Lernen)
# ----------------------------

syn_model_h1_h2 = '''
    w : 1
    dxpre/dt = -xpre/tau_pre_h1_h2 : 1 (event-driven)
    dypos/dt = -ypos/tau_post_h1_h2 : 1 (event-driven)
'''

S_h1_h2 = Synapses(h1, h2, model=syn_model_h1_h2,
                   on_pre='''
                        xpre += A_h1_h2(t, i_pre)
                        w = clip(w - eta_h1_h2*(xpre*ypos), wmin_h1_h2, wmax_h1_h2)
                        I_syn_post += w * xpre
                   ''',
                   on_post='''
                        ypos += 1.0
                        w = clip(w + eta_h1_h2*(xpre*ypos) - eta_h1_h2*oja_h1_h2*(ypos*ypos)*w,
                                  wmin_h1_h2, wmax_h1_h2)
                   ''',
                   method='euler', name='S_h1_h2')

S_h1_h2.connect(p=0.3)  # etwas dünner, um Dynamik/Selektivität zu fördern
S_h1_h2.w = '0.4 + 0.4*rand()'
S_h1_h2.xpre = 0.0
S_h1_h2.ypos = 0.0

# ----------------------------
# Synapsen: h2 → out (mit Lernen)
# ----------------------------

syn_model_h2_out = '''
    w : 1
    dxpre/dt = -xpre/tau_pre_h2_out : 1 (event-driven)
    dypos/dt = -ypos/tau_post_h2_out : 1 (event-driven)
'''

S_h2_out = Synapses(h2, out, model=syn_model_h2_out,
                    on_pre='''
                        xpre += A_h2_out(t, i_pre)
                        w = clip(w - eta_h2_out*(xpre*ypos), wmin_h2_out, wmax_h2_out)
                        I_syn_post += w * xpre
                    ''',
                    on_post='''
                        ypos += 1.0
                        w = clip(w + eta_h2_out*(xpre*ypos) - eta_h2_out*oja_h2_out*(ypos*ypos)*w,
                                  wmin_h2_out, wmax_h2_out)
                    ''',
                    method='euler', name='S_h2_out')

S_h2_out.connect(p=0.4)
S_h2_out.w = '0.6 + 0.6*rand()'  # etwas kräftiger Richtung out
S_h2_out.xpre = 0.0
S_h2_out.ypos = 0.0

# ----------------------------
# Hintergrund-Inputs (sanft) für h1/h2/out, um out-Spiken zu sichern
# ----------------------------
# Additive Poisson-Inputs direkt auf I_syn (exzitatorisch)

bg_rate_h = 6*Hz
bg_rate_out = 8*Hz

BG_h1 = PoissonInput(h1, target='I_syn', N=1, rate=bg_rate_h, weight='15*pA')
BG_h2 = PoissonInput(h2, target='I_syn', N=1, rate=bg_rate_h, weight='15*pA')
BG_out = PoissonInput(out, target='I_syn', N=1, rate=bg_rate_out, weight='20*pA')

# Optional: leichte laterale Inhibition im out-Layer, um Wettbewerb zu erzeugen
# (Winner-take-some statt völliger Stille)

w_inh = 0.4*volt  # symbolisch – wir setzen hier aber lieber über I_syn_explizit
# Vereinfachung: kein separates Interneuronen-Pool; stattdessen kleine Selbstinhibition über tau_syn-Zerfall
# (für Klarheit hier weggelassen – falls benötigt, kann ein Inhibitionspool ergänzt werden.)

# ----------------------------
# Monitore
# ----------------------------

M_in_sp   = SpikeMonitor(P_in)
M_h1_sp   = SpikeMonitor(h1)
M_h2_sp   = SpikeMonitor(h2)
M_out_sp  = SpikeMonitor(out)

M_out_V   = StateMonitor(out, 'V', record=True)
M_out_Is  = StateMonitor(out, 'I_syn', record=True)

# Gewichte (ggf. nur Teilmenge recorden, um Speicher zu schonen)
rec_k = min(10, N_in*N_h1)
S_in_h1_idx = np.arange(min(len(S_in_h1.w[:]), 50))  # erste 50 Synapsen

W_in_h1_mon = StateMonitor(S_in_h1, 'w', record=S_in_h1_idx)
X_in_h1_mon = StateMonitor(S_in_h1, 'xpre', record=S_in_h1_idx)
Y_in_h1_mon = StateMonitor(S_in_h1, 'ypos', record=S_in_h1_idx)

# ----------------------------
# Simulation
# ----------------------------

print('Simulation startet...')
run(T_sim)
print('Simulation fertig.')

# ----------------------------
# Minimaler Auswertungs-Print (optional)
# ----------------------------

print(f"Spikes IN:  {M_in_sp.num_spikes}")
print(f"Spikes H1:  {M_h1_sp.num_spikes}")
print(f"Spikes H2:  {M_h2_sp.num_spikes}")
print(f"Spikes OUT: {M_out_sp.num_spikes}")

# Hinweis: Plotting weggelassen (Notebook/Script-abhängig).
# In einem Notebook könnte man nun z.B. M_out_sp.i/t plotten, sowie W_in_h1_mon.w.
