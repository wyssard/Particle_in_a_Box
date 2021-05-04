import particle_in_a_box as pib
import state_plot
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

plt.rcParams['animation.writer'] = 'ffmpeg'
plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'


gamma = -5
L = np.pi
m = 1

energy_state = pib.Energy_Space_Projection(L, gamma, m, [1, 2], [1,1])

state = pib.Particle_in_Box_State_v2(energy_state)

plot = state_plot.State_Plot(state)
plot.add_x_space_axis(-L/2, L/2, 0.01)
plot.add_k_space_axis(-50, 50, 0.01)

plot.animate_plot(20, 3, 100000)
plot._anim.save("first_and_second_comb.mp4")