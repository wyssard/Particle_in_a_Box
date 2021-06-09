import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
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


case = "dirichlet_neumann"
L = np.pi
m = 1
a = L/10
l_0 = 100
l_range = 15

fps = 20
speed = 0.025
real_time = (4*m*L**2/np.pi)/4
time = real_time/speed


myState = special.Bouncing_Gaussian(case, L, m, l_0, l_range, a)

# Create Animations
eps = 0.25

x = np.linspace(-L/2, L/2, 300, endpoint=True)
n_bound = l_0+l_range
k_bound = myState.boundary_lib.get_kn(n_bound)
n = np.arange(-n_bound, n_bound, 1, dtype=int)
kn = myState.boundary_lib.get_kn(n)
k = np.linspace(-k_bound, k_bound, 300, endpoint=True)


x_space_wavefunc = myState.x_space_wavefunction
x_exp_val = myState.x_space_expectation_value
k_space_wavefunc = myState.k_space_wavefunction
new_k_space_wavefunc = myState.new_k_space_wavefunction
k_exp_val = myState.new_k_space_expectation_value


fig = plt.figure(tight_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1,1])
k_distr_plot = fig.add_subplot(gs[1,0])
x_distr_plot = fig.add_subplot(gs[0, 0])

k_distr_plot.set_xlabel("k")
x_distr_plot.set_xlabel("x")
x_distr_plot.set_ylabel("Probability Density")
k_distr_plot.set_ylabel("Probability Distribution / Density")

k_distr_plot.set_ylim([0, 0.5])
x_distr_plot.set_xlim([-L/2-eps, L/2+eps])

k_lines = k_distr_plot.plot(k, np.abs(k_space_wavefunc(k, 0))**2, animated=True, color = darkColor)
x_lines = x_distr_plot.plot(x, np.abs(x_space_wavefunc(x, 0))**2, animated=True, color = darkColor)
k_bars = k_distr_plot.bar(kn, np.abs(new_k_space_wavefunc(n, 0))**2, animated=True, color = lightColor)
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

    for bar, h in zip(k_bars, np.abs(new_k_space_wavefunc(n, i*time_per_frame))**2):
        bar.set_height(h)
    return x_lines + k_lines + [x_exp_line] + [k_exp_line] + list(k_bars)

anim = FuncAnimation(fig, animate, init_func=init, frames=int(num_frames), interval=int(1/fps*1000), blit=True)
anim.save(".\\Demo_Animations\\bouncing_gaussian_" + case + "_" + str(time) + "s.mp4")

