from sys import path
path.append("d:\\valen\\Documents\\Uni-Bern\\Physik\\Physik_FS21\\Bachelorarbeit\\Particle_in_a_Box_Library")
from compileall import compile_file
compile_file("..\\pib_lib\\update_plot.py")


from pib_lib import particle_in_a_box as pib
from pib_lib import Special_States as sp
from pib_lib import update_plot as up
import numpy as np
import mpld3
from matplotlib import pyplot as plt
plt.rcParams['animation.writer'] = 'ffmpeg'
plt.rcParams["animation.html"] = "jshtml"

pib_state = pib.Particle_in_Box_Immediate_Mode("symmetric", 1, 1, [2,3,4],[1,1,1], gamma=100)
rev_time = up.revival_time(pib_state)

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot()
pos_exp_val = up.Position_Expectation_Value_Evolution(pib_state, fig, ax).set_t_range([0, rev_time])
mom_exp_val = up.Momentum_Expectation_Value_Evolution(pib_state, fig, ax).set_t_range([0, rev_time]).plot_config(color=up.light_blue)
c_plot = up.Update_Plot_Collection(fig, pos_exp_val, mom_exp_val)
pib_state.case = "symmetric"
pib_state.gamma = 0.01
#c_plot.draw()

#c_plot.anim("gamma", 0.001, 100, 20, 100)

c_plot.plot()
mpld3.show()
