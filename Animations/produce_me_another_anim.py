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


L = np.pi
m = 1
test_states = [2, 3]
test_amps = [1, 1]
test_gamma = 10

test_state_analytic = pib.Particle_in_Box_State("symmetric", L, m, test_states, test_amps, test_gamma)


t = 0.5

fig = plt.figure(figsize=(8,4))
test_state_analytic.theta = 0
momentum_plot = dp.Momentum_Space_Plot(test_state_analytic, fig)
momentum_plot.set_n_bound(10)
momentum_plot.expectation_value = True
momentum_plot.plot(t)


fps = 30
time = 10
num_frames = fps*time

def animframe(i):
    test_state_analytic.theta = i/num_frames*2*np.pi
    momentum_plot.update()
    k_ev = momentum_plot.k_exp_val(t)
    momentum_plot.k_exp_line.set_data([k_ev, k_ev], [0, 1])
    for bar, h, pos in zip(momentum_plot.k_bars, np.abs(momentum_plot.new_k_space_wavefunc(momentum_plot.n, t))**2, momentum_plot.kn):
        bar.set_height(h)
        bar.set_x(pos - bar.get_width()/2)
    return list(momentum_plot.k_bars) + [momentum_plot.k_exp_line] + momentum_plot.k_lines

anim = FuncAnimation(fig, animframe, frames=int(num_frames), interval=int(1/fps*1000), blit=True)
anim.save(".\\lambda_independance.mp4")