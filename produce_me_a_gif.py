import numpy as np
from matplotlib import pyplot as plt
import Special_States as special
import default_plot as dp

lightColor = "#8bb1cc"
darkColor = "#0f4c75"

plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Dejavu Serif'
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['animation.writer'] = 'ffmpeg'
plt.rcParams['animation.ffmpeg_args'] = ['-loop', '-1']

case = "dirichlet"
L = np.pi
m = 1
a = L/10
l_0 = 80
l_range = 40

fps = 24
speed = 0.05
real_time = (4*m*L**2/np.pi)/4
time = (real_time/speed)

fig = plt.figure(dpi=300, constrained_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=1)

gaussian = special.Bouncing_Gaussian(case, L, m, l_0, l_range, a)

position_plot = dp.Position_Space_Plot(gaussian, fig, gs, [0,0])
position_plot.axis.set_ylabel("Probability Density")
position_plot.set_resolution(5000)

momentum_plot = dp.Momentum_Space_Plot(gaussian, fig, gs, [1,0])
momentum_plot.set_n_bound(100)
momentum_plot.axis.set_ylim([0, 0.25])
momentum_plot.set_resolution(1000)

combined_plot = dp.Multi_Plot(position_plot, momentum_plot)
combined_plot.animate(fps, time, speed).save("bouncing_gaussian_dirichlet_quarter_revival.mp4")
