from __future__ import annotations
from abc import ABC, abstractmethod
import particle_in_a_box as pib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Updatable_Plot(ABC):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, gs = None, pos = [0,0], light_color = "#8bb1cc", dark_color = "#0f4c75") -> None:
        self._light_color = light_color
        self._dark_color = dark_color
        self.fig = fig
        if gs == None:
            self.axis = fig.add_subplot()
        else:
            self.axis = fig.add_subplot(gs[pos[0], pos[1]])

        self._state = state
        self.expectation_value = False
        self.update()

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def plot(self):
        self.update()

    @abstractmethod
    def animate_frame(self, i: int):
        pass

    def animate(self, fps: int, time: float, speed: float) -> FuncAnimation:
        energy_space_wavefunc = self._state.energy_space_wavefunction
        if self._state._sp._num_energy_states == 2:
            time = 2*np.pi/abs(energy_space_wavefunc.energies[0]-energy_space_wavefunc.energies[1])/speed
            print("auto setting time to: ", time)
        
        self.time_per_frame = 1/fps*speed
        self.num_frames = time*fps
        self.plot(0)

        anim = FuncAnimation(self.fig, self.animate_frame, frames=int(self.num_frames), interval=int(1/fps*1000), blit=True)    
        return anim

class Position_Space_Plot(Updatable_Plot):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, gs = None, pos = [0, 0]) -> None:
        super().__init__(state, fig, gs, pos)
        self._state = state
        self.update()

    def update(self):
        self.x = np.linspace(-self._state.L/2, self._state.L/2, 300, endpoint=True)
        self.x_space_wavefunc = self._state.x_space_wavefunction
        self.x_exp_val = self._state.x_space_expectation_value

    def plot(self, time: float) -> None:
        super().plot()
        self.x_lines = self.axis.plot(self.x, np.abs(self.x_space_wavefunc(self.x, time))**2, animated=True, color = self._dark_color)
        if self.expectation_value == True:
            self.x_exp_line = self.axis.axvline(self.x_exp_val(time), animated=True, color = self._dark_color)

    def animate_frame(self, i):
        self.x_lines[0].set_data(self.x, np.abs(self.x_space_wavefunc(self.x, self.time_per_frame*i))**2)
        if self.expectation_value == False:
            return self.x_lines
        else:
            x_ev = self.x_exp_val(self.time_per_frame*i)
            self.x_exp_line.set_data([x_ev, x_ev],[0,1])
            return self.x_lines + [self.x_exp_line]

class Momentum_Space_Plot(Updatable_Plot):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, gs = None, pos = [0,0]) -> None:
        super().__init__(state, fig, gs, pos)
        self._state = state
        self.update()

        self.n_bound = 15
        self.axis.set_ylim([0, 0.75])

    def update(self):
        self.k_space_wavefunc = self._state.k_space_wavefunction
        self.new_k_space_wavefunc = self._state.new_k_space_wavefunction

    def set_n_bound(self, new_bound: int) -> None:
        self.n_bound = new_bound
        self.n = np.arange(-self.n_bound, self.n_bound, 1, dtype=int)
        self.kn = self._state.boundary_lib.get_kn(self.n)
        self.k_bound = self._state.boundary_lib.get_kn(self.n_bound*(1.001))
        self.k = np.linspace(-self.k_bound, self.k_bound, 300, endpoint=True)
        print(self.n_bound)

    def plot(self, time: float | np.ndarray):
        super().plot()
        self.k_lines = self.axis.plot(self.k, np.abs(self.k_space_wavefunc(self.k, time))**2, animated=True, color = self._dark_color)
        self.k_bars = self.axis.bar(self.kn, np.abs(self.new_k_space_wavefunc(self.n, time))**2, animated=True, color = self._light_color)

    def animate_frame(self, i: int):
        self.k_lines[0].set_data(self.k, np.abs(self.k_space_wavefunc(self.k, self.time_per_frame*i))**2)
        for bar, h in zip(self.k_bars, np.abs(self.new_k_space_wavefunc(self.n, i*self.time_per_frame))**2):
            bar.set_height(h)
        return self.k_lines + list(self.k_bars)    

class Multi_Plot:
    def __init__(self, *plots: Updatable_Plot) -> None:
        self._plots = plots

    def update(self) -> None:
        for plot in self._plots:
            plot.update()
    
    def plot(self, time: float) -> None:
        for plot in self._plots:
            plot.plot(time)
    
    def animate_frame(self, i: int):
        out = []
        for plot in self._plots:
            out += plot.animate_frame(i)
        return out

    def animate(self, fps: int, time: float, speed: float) -> FuncAnimation:
        energy_space_wavefunc = self._plots[0]._state.energy_space_wavefunction
        fig = self._plots[0].fig

        if self._plots[0]._state._sp._num_energy_states == 2:
            time = 2*np.pi/abs(energy_space_wavefunc.energies[0]-energy_space_wavefunc.energies[1])/speed
            print("auto setting time to: ", time)
        
        time_per_frame = 1/fps*speed
        for plot in self._plots:
            plot.time_per_frame = time_per_frame
        num_frames = time*fps
        self.plot(0)

        anim = FuncAnimation(fig, self.animate_frame, frames=int(num_frames), interval=int(1/fps*1000), blit=True)    
        return anim
