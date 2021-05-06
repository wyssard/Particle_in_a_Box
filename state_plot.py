import particle_in_a_box as pib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class State_Plot:
    def __init__(self, state: pib.Particle_in_Box_State_v2):
        plt.rcParams["text.usetex"] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'Dejavu Serif'
        plt.rcParams["mathtext.fontset"] = "cm"
        plt.rcParams['animation.writer'] = 'ffmpeg'
        plt.rcParams["animation.html"] = "jshtml"
        plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'

        self._state = state
        self._fig = plt.figure(figsize=(7,6))
        self._gs = self._fig.add_gridspec(nrows=2, ncols=1)
        self._lightColor = "#8bb1cc"
        self._darkColor = "#0f4c75"

    def add_x_space_axis(self, x_min, x_max, x_step):
        self._x_range = np.arange(x_min, x_max+x_step, x_step)

        self._x_space_axis = self._fig.add_subplot(self._gs[1,0])
        self._x_space_axis.set_xlabel("$x$")
        self._x_space_axis.set_ylabel(r"Probability Density")
        self._x_space_axis.set_xticks([-self._state._L/2, -self._state._L/4, 0, self._state._L/4, self._state._L/2])
        self._x_space_axis.set_xticklabels([r"$-L/2$", r"$-L/4$", r"0", r"$-L/4$", r"$L/2$"])

        self._x_space_abs_sq = lambda t: np.abs(self._state._xsp._x_space_wavefunction(self._x_range, t))**2
        self._x_space_plot = self._x_space_axis.plot(self._x_range, self._x_space_abs_sq(0), c = self._darkColor, animated=True)

    def add_k_space_axis(self, k_min, k_max, k_cStep):
        self._k_range = np.arange(k_min, k_max+k_cStep, k_cStep)
        self._new_k_range = np.arange(k_min, k_max+1, 1)

        self._k_space_axis = self._fig.add_subplot(self._gs[0,0])
        self._k_space_axis.set_xlabel("$k$")
        self._k_space_axis.set_ylabel("Probability Distribution / Density")
        #self._k_space_axis.legend(loc="upper left")
        self._k_space_axis.grid(True, which="major", axis="y", lw=0.5, c="0.8")

        self._k_space_abs_sq = lambda t: np.abs(self._state._ksp._cont_k_space_wavefunction(self._k_range, t))**2
        self._new_k_space_abs_sq = lambda t: np.abs(self._state._new_ksp._new_k_space_wavefunction(self._new_k_range, t))**2
        self._k_space_plot = self._k_space_axis.plot(self._k_range, self._k_space_abs_sq(0), c = self._darkColor, animated=True)
        self._new_k_space_plot = self._k_space_axis.bar(self._new_k_range, self._new_k_space_abs_sq(0), color = self._lightColor, animated=True)

    def update_plot(self):
        pass

    def animate_plot(self, fps: int, time: float, speed: float):
        if self._state._esp._num_energy_states == 2:
            energies = self._state._esp._energies
            time = 2*np.pi/(abs(energies[0]-energies[1]))/speed
            print("{0:}{1:2.3f}{2:}".format("auto setting time to: ", time, " to enable periodic animation"))

        def init():
            return self._x_space_plot + self._k_space_plot + list(self._new_k_space_plot)
        
        time_step = 1/fps*speed
        num_frames = int(time*fps)

        def animate(i):
            self._x_space_plot[0].set_data(self._x_range, self._x_space_abs_sq(i*time_step))
            self._k_space_plot[0].set_data(self._k_range, self._k_space_abs_sq(i*time_step))

            for bar, h in zip(self._new_k_space_plot, self._new_k_space_abs_sq(i*time_step)):
                bar.set_height(h)

            return self._x_space_plot + self._k_space_plot + list(self._new_k_space_plot)
        
        self._anim = FuncAnimation(self._fig, animate, init_func=init, frames=num_frames, interval=int(1000/fps), blit=True)


    