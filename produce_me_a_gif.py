import particle_in_a_box as pib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List
import Special_States as special

lightColor = "#8bb1cc"
darkColor = "#0f4c75"

plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Dejavu Serif'
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['animation.writer'] = 'ffmpeg'
plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'

def momentum_space_gaussian(a, k_0, k) -> float:
    output = np.sqrt(2*a*np.sqrt(np.pi))*np.exp(-a**2/2*(k-k_0)**2)
    return output

def project_gaussian(L, a, l_0, num, which) -> List[list]:
    states = []
    amplitudes = []
    k_0 = np.pi/L*l_0
    state_range = range(l_0-num, l_0+num)

    for state in state_range:
        states.append(state)
        k = state*np.pi/L
        psi_append_pos = momentum_space_gaussian(a, k_0, k)
        psi_append_neg = momentum_space_gaussian(a, k_0, -k)
        if state%2 == 0:
            amplitudes.append(1j*(psi_append_pos-psi_append_neg))
        else:
            amplitudes.append(psi_append_pos + psi_append_neg)

    if (0 in state_range) and which == "neumann":
        amplitudes[state_range.index(0)] = np.sqrt(2)*momentum_space_gaussian(a, k_0, 0)

    
    return [states, amplitudes]


fps = 20
time = 50
speed = 0.01

case = "neumann"
L = np.pi
m = 1
a = L/10
l_0 = 100
k_0 = l_0*np.pi/L
k_range = 15

if case == "neumann":
    gamma = 0.000001
    gaussian_proj = project_gaussian(L, a, l_0, k_range, "neumann")
elif case == "dirichlet":
    gamma = 100000
    gaussian_proj = project_gaussian(L, a, l_0, k_range, "dirichlet")

states = gaussian_proj[0]
amplitudes = gaussian_proj[1]

myState = pib.Particle_in_Box_State(gamma, L, m, states, amplitudes)

# Create Animations
x = np.arange(-L/2, L/2+0.01, 0.01)
kb = k_0+15
k = np.arange(-kb, kb+0.01, 0.01)
kn = np.arange(-kb, kb+1, 1)

x_space_wavefunc = myState.x_space_wavefunction
x_exp_val = myState._xsp._expectation_value
k_space_wavefunc = myState.k_space_wavefunction
new_k_space_wavefunc = myState.new_k_space_wavefunction
k_exp_val = myState._ksp._expectation_value

fig = plt.figure(tight_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1,1])
k_distr_plot = fig.add_subplot(gs[1,0])
x_distr_plot = fig.add_subplot(gs[0, 0])

k_distr_plot.set_xlabel("k")
x_distr_plot.set_xlabel("x")
x_distr_plot.set_ylabel("Probability Density")
k_distr_plot.set_ylabel("Probability Distribution / Density")

k_distr_plot.set_ylim([0, 0.5])

k_lines = k_distr_plot.plot(k, np.abs(k_space_wavefunc(k, 0))**2, animated=True, color = darkColor)
x_lines = x_distr_plot.plot(x, np.abs(x_space_wavefunc(x, 0))**2, animated=True, color = darkColor)
k_bars = k_distr_plot.bar(kn, np.abs(new_k_space_wavefunc(kn, 0))**2, animated=True, color = lightColor)
x_exp_line = x_distr_plot.axvline(x_exp_val(0), animated=True, color = darkColor)
k_exp_line = k_distr_plot.axvline(k_exp_val(0), animated=True, color = darkColor)



if myState._sp._num_energy_states == 2:
    time = 2*np.pi/abs(myState._esp._energies[0]-myState._esp._energies[1])/speed
    print("auto setting time to: ", time)

time_per_frame = 1/fps*speed
num_frames = time*fps

def init():
    return x_lines + k_lines + [x_exp_line] + [k_exp_line] + list(k_bars)

def animate(i):
    x_lines[0].set_data(x, np.abs(x_space_wavefunc(x, time_per_frame*i))**2)
    k_lines[0].set_data(k, np.abs(k_space_wavefunc(k, time_per_frame*i))**2)
    x_ev = x_exp_val(time_per_frame*i)
    k_ev = k_exp_val(time_per_frame*i)
    x_exp_line.set_data([x_ev, x_ev],[0,1])
    k_exp_line.set_data([k_ev, k_ev],[0,1])

    for bar, h in zip(k_bars, np.abs(new_k_space_wavefunc(kn, i*time_per_frame))**2):
        bar.set_height(h)
    return x_lines + k_lines + [x_exp_line] + [k_exp_line] + list(k_bars)

anim = FuncAnimation(fig, animate, init_func=init, frames=int(num_frames), interval=int(1/fps*1000), blit=True)
anim.save(".\\Demo_Animations\\bouncing_gaussian_" + case + "_" + str(time) + "s.mp4")

