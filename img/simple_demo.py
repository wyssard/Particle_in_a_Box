from pib_lib import particle_in_a_box as pib
from pib_lib import update_plot as up
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('Solarize_Light2')

L = np.pi
m = 1
states = [1,2,3]
amps = [1,1,1]
case = "dirichlet_neumann"
gamma = 10

state = pib.Particle_in_Box_State(case, L, m, states, amps)

fig = plt.figure(figsize=(6,5), tight_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=1)
pos_ax = fig.add_subplot(gs[0,0])
momentum_ax = fig.add_subplot(gs[1,0])

position_plot = up.position_space_plot(state, fig, pos_ax)
momentum_plot = up.momentum_space_plot(state, fig, momentum_ax)
new_momentum_plot = up.new_momentum_space_plot(state, fig, momentum_ax)
new_momentum_plot["abs_square"].plot_config(color=up.light_blue)
plots = up.Update_Plot_Collection(fig, position_plot, new_momentum_plot, momentum_plot)

plots.set_n_bound(10)
plots.set_t(0.7)

plots.plot()
plt.savefig("img\\simple_demo.svg")