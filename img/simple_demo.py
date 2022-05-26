from pib_lib import particle_in_a_box as pib
from pib_lib import update_plot as up
import numpy as np
from matplotlib import pyplot as plt

L = np.pi
m = 1
states = [1,2,3]
amps = [1,1,1]
case = "anti_symmetric"
gamma = 10

state = pib.Particle_in_Box_State(case, L, m, states, amps, gamma)

fig = plt.figure(figsize=(6,5))
gs = fig.add_gridspec(nrows=2, ncols=1)
pos_ax = fig.add_subplot(gs[0,0])
momentum_ax = fig.add_subplot(gs[1,0])

position_plot = up.position_space_plot(state, fig, pos_ax)
momentum_plot = up.momentum_space_plot(state, fig, momentum_ax)
plots = up.Update_Plot_Collection(fig, position_plot, momentum_plot)

plots.set_t(1)
plots.plot()
plt.savefig("img\\simple_demo.png")