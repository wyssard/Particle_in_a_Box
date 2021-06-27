from __future__ import annotations

import particle_in_a_box as pib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import default_plot as dp

plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Dejavu Serif'
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['animation.writer'] = 'ffmpeg'
plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'


gamma = 10
L = np.pi
m = 1
states = [0, 1, 2, 3]
amps = [1, 1, 1, 1]


myState = pib.Particle_in_Box_State("symmetric", L, m, states, amps, gamma)
expectation_value_plot = dp.Expectation_Value_Plot(myState)

T = 11
eps = 0.1
fps = 21
num_frames = fps*T

myState.gamma = np.tan(-np.pi/2+eps)
time = np.linspace(0, 5, 500)

expectation_value_plot.axis.set_ylim([-1.5, 1.5])
expectation_value_plot.plot(time)
text = expectation_value_plot.axis.text(4, -1, r"$\gamma$: " + str(round(myState.gamma)), animated=True)

eneries = myState.energy_space_wavefunction.energies
energy_texts = "states | energies: \n"
for i in range(myState._sp.num_energy_states):
    energy_texts += str(myState._sp.energy_states[i]) + " | " +str(round(eneries[i], 4)) + "\n"

energy_text = expectation_value_plot.axis.text(2, -1.5, energy_texts, animated=True)
expectation_value_plot.axis.legend(loc="lower left")


def animate_frame(i):
    myState.gamma = np.tan(-np.pi/2+eps + (i/fps)*(np.pi-2*eps)/T)
    #myState.gamma = (-2/myState.L - eps) + (i/fps)*(2*eps/T) 

    expectation_value_plot.update()
    text.set_text(r"$\gamma$: " + str(round(myState.gamma, 4)))

    energy_texts = "states | energies: \n"
    for i in range(myState._sp.num_energy_states):
        energy_texts += str(myState._sp.energy_states[i]) + " | " +str(round(eneries[i], 4)) + "\n"
    energy_text.set_text(energy_texts)


    expectation_value_plot.x_exp_line[0].set_data(time, myState.x_space_expectation_value(time))
    expectation_value_plot.k_exp_line[0].set_data(time, myState.new_k_space_expectation_value(time))
    expectation_value_plot.x_exp_deriv_line[0].set_data(time, myState.m * myState.x_space_expectation_value_derivative(time))

    return expectation_value_plot.x_exp_line + expectation_value_plot.k_exp_line + expectation_value_plot.x_exp_deriv_line

anim = FuncAnimation(expectation_value_plot.fig, animate_frame, frames=int(num_frames), interval=int(1/fps*1000), blit=True)
anim.save(".\\Demo_Animations\\expectation_values_symmetric.mp4")
