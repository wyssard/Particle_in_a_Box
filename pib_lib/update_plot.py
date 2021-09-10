from __future__ import annotations
from typing import Callable, List
from matplotlib.container import BarContainer
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from . import particle_in_a_box as pib
import colorsys as cs
from abc import ABC, abstractmethod

light_blue = cs.hls_to_rgb(235/360, 0.87, 1)
less_light_blue = cs.hls_to_rgb(235/360, 0.77, 1)
mid_blue = cs.hls_to_rgb(235/360, 0.67, 1)
dark_blue = cs.hls_to_rgb(235/360, 0.37, 1)
light_red = cs.hls_to_rgb(0, 0.87, 1)
mid_red = cs.hls_to_rgb(0, 0.67, 1)
dark_red = cs.hls_to_rgb(0, 0.37, 1)

def revival_time(state: pib.Particle_in_Box_State) -> float:
    return 4*state.m*state.L**2/np.pi

class Updatable_Plot(ABC):
    """This class provides basic visualization functionality (based on 
    matplotlib) for the particle-in-a-box states. For more advanced or better
    customizable plotting it is therefore more efficient to create the plots 
    from scratch by using, e.g., matplotlib. 
    """
    def __init__(self, fig: plt.Figure) -> None:
        self._fig = fig
        self._res = 300
        self._color = dark_blue
        self._ls = "-"
        self._lw = 1

    def plot_config(self, color = dark_blue, linestyle = "-", linewidth = 1) -> Updatable_Plot:
        self._color = color
        self._ls = linestyle
        self._lw = linewidth
        return self

    @abstractmethod
    def plot(self) -> plt.Line2D:
        pass

    def update(self) -> None:
        pass
    
    @abstractmethod
    def _anim_t(self) -> Callable[[int], List[plt.Line2D]]:
        pass
    
    @abstractmethod
    def _anim_L(self) -> Callable[[int], List[plt.Line2D]]:
        pass

    @abstractmethod
    def _anim_gamma(self) -> Callable[[int], List[plt.Line2D]]:
        pass

    def anim(self, var: str, start: float, stop: float, fps: int, speed: float) -> FuncAnimation:
        eff_time_per_frame = 1/fps
        system_time_per_frame = eff_time_per_frame*speed
        num_frames = (stop-start)*fps/speed
        if var=="t":
            anim=self._anim_t(start, system_time_per_frame)
        elif var=="L":
            anim=self._anim_L(start, system_time_per_frame)
        elif var=="gamma":
            anim=self._anim_gamma(start, system_time_per_frame)

        fanim = FuncAnimation(self._fig, anim, int(num_frames), interval=eff_time_per_frame*1000, blit=True)
        return fanim

    @property
    def res(self) -> int:
        return self._res
    @res.setter
    def res(self, new_res) -> None:
        self._res = new_res

class Single_Updatable_Plot(Updatable_Plot):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None) -> None:
        super().__init__(fig)
        self._res = 300
        self._state = state
        self._time = 0
        self._lines = None
        if not axis:
            self._axis = fig.add_subplot()
        else:
            self._axis = axis

    def set_t(self, time) -> Single_Updatable_Plot:
        self._time = time
        return self

    def _line_data_updater(self) -> None:
        pass

    def _line_data_init(self) -> None:
        pass

    def plot(self) -> plt.Line2D:
        self.update()
        if self._lines:
            self._line_data_updater()
        else:
            self._line_data_init()
        
    def _anim_t(self, start, system_time_per_frame) -> Callable[[int], List[plt.Line2D]]:
        self._time = start
        self.plot()
        def anim(i):
            self._time = start+i*system_time_per_frame
            self._line_data_updater()
            return self._lines
        return anim

    def _anim_gamma(self, start, system_gamma_per_frame) -> Callable[[int], List[plt.Line2D]]:
        self._state.gamma = start
        self.plot()
        def anim(i):
            self._state.gamma = start+i*system_gamma_per_frame
            self._line_data_updater()
            return self._lines
        return anim

    def _anim_L(self, start, system_L_per_frame) -> Callable[[int], List[plt.Line2D]]:
        self._state.L = start
        self.plot()
        def anim(i):
            self._state.L = start+i*system_L_per_frame
            self._line_data_updater()
            return self._lines
        return anim

class Single_Updatable_Line(Single_Updatable_Plot):
    def __init__(self, state: pib.Particle_in_Box_State, mode: str, fig: plt.Figure, axis: plt.Axes=None) -> None:
        super().__init__(state, fig, axis=axis)
        self._mode = mode

    def update(self) -> None:
        self._x = None
        self._y = None

    def _line_data_init(self) -> None:
        self._lines = self._axis.plot(self._x, self._y(self._time), color=self._color, linestyle=self._ls, linewidth=self._lw, animated=True)
    
    def _line_data_updater(self) -> None:
        self._lines[0].set_data(self._x, self._y(self._time))

    def plot(self) -> plt.Line2D:
        super().plot()
        return self._lines[0]

class Pos_Space_Plot(Single_Updatable_Line):
    def __init__(self, state: pib.Particle_in_Box_State, mode: str, fig: plt.Figure, axis: plt.Axes=None) -> None:
        super().__init__(state, mode, fig, axis=axis)

    def update(self) -> None:
        self._x = np.linspace(-self._state.L/2, self._state.L/2, self._res)
        if self._mode=="abs_square":
            self._y = lambda t: np.abs(self._state.x_space_wavefunction(self._x, t))**2
        elif self._mode=="real":
            self._y = lambda t: np.real(self._state.x_space_wavefunction(self._x, t))
        elif self._mode=="imag":
           self._y = lambda t: np.imag(self._state.x_space_wavefunction(self._x, t))

class Momentum_Space_Plot(Single_Updatable_Line):
    def __init__(self, state: pib.Particle_in_Box_State, mode: str, fig: plt.Figure, axis: plt.Axes=None) -> None:
        super().__init__(state, mode, fig, axis=axis)
        self._n_bound = 15

    def set_bound(self, new_bound) -> Momentum_Space_Plot:
        self._n_bound = new_bound
        return self

    def update(self) -> None:
        k_bound = self._state.boundary_lib.get_kn(self._n_bound)
        self._x = np.linspace(-k_bound, k_bound, self._res)
        if self._mode=="abs_square":
            self._y = lambda t: np.abs(self._state.k_space_wavefunction(self._x, t))**2
        elif self._mode=="real":
            self._y = lambda t: np.real(self._state.k_space_wavefunction(self._x, t))
        elif self._mode=="imag":
           self._y = lambda t: np.imag(self._state.k_space_wavefunction(self._x, t))

class Expectation_Value_Line(Single_Updatable_Plot):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None) -> None:
        super().__init__(state, fig, axis=axis)
        self._time = 0
        self._line = None

    def update(self) -> None:
        self._expectation_value = None

    def _line_data_init(self) -> None:
        self._line = self._line = self._axis.axvline(self._expectation_value(self._time), color=self._color, linewidth=self._lw, linestyle=self._ls, animated=True)
        self._lines = [self._line]
    
    def _line_data_updater(self) -> None:
        expectation_value_position = self._expectation_value(self._time)
        self._line.set_data([expectation_value_position, expectation_value_position],[0,1])
        self._lines = [self._line]

    def plot(self) -> plt.Line2D:
        super().plot()
        return self._line
  
class Pos_Expectation_Value(Expectation_Value_Line):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None) -> None:
        super().__init__(state, fig, axis=axis)

    def update(self) -> None:
        self._expectation_value = lambda t: self._state.x_space_expectation_value(t)

class Momentum_Expectation_Value(Expectation_Value_Line):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None) -> None:
        super().__init__(state, fig, axis=axis)

    def update(self) -> None:
        self._expectation_value = lambda t: self._state.new_k_space_expectation_value(t)

class New_Momentum_Space_Plot(Single_Updatable_Plot):
    def __init__(self, state: pib.Particle_in_Box_State, mode: str, fig: plt.Figure, axis: plt.Axes=None) -> None:
        super().__init__(state, fig, axis=axis)
        self._mode = mode
        self._bound = 15
        self._bar_color = light_blue
        self._bars = None

    def plot_config(self, bar_color) -> New_Momentum_Space_Plot:
        self._bar_color = bar_color
        return self

    def set_n_bound(self, new_bound) -> New_Momentum_Space_Plot:
        self._bound = new_bound
        return self

    def update(self) -> None:
        self._n = np.arange(-self._bound, self._bound+1, 1, dtype=int)
        self._kn = self._state.boundary_lib.get_kn(self._n)
        self._barwidth = 0.8*(self._kn[1]-self._kn[0])
        if self._mode=="abs_square":
            self._y = lambda t: np.abs(self._state.new_k_space_wavefunction(self._n, t)**2)
        elif self._mode=="real":
            self._y = lambda t: np.real(self._state.new_k_space_wavefunction(self._n, t))
        elif self._mode=="imag":
           self._y = lambda t: np.imag(self._state.new_k_space_wavefunction(self._n, t))

    def _line_data_init(self) -> None:
        self._bars = self._axis.bar(self._kn, self._y(self._time), self._barwidth, color=self._bar_color, animated=True)
        self._lines = list(self._bars)

    def _line_data_updater(self) -> None:
        for bar, h in zip(self._bars, self._y(self._time)):
            bar.set_height(h)
        self._lines = list(self._bars)

    def plot(self) -> BarContainer:
        super().plot()
        return self._bars

class Expectation_Value_Evolution(Single_Updatable_Plot):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None) -> None:
        super().__init__(state, fig, axis=axis)
        self._t_range = [0,10]
        self._time_line = None
        self._time_evolution = None
        self._value_at_time = None
        self._c_color = "0.7"

    def plot_config(self, color=dark_blue, linestyle="-", linewidth=1, c_color="0.7"):
        self._c_color = c_color
        return super().plot_config(color=color, linestyle=linestyle, linewidth=linewidth)
        
    def set_t_range(self, new_range):
        self._t_range = new_range
        return self

    def update(self) -> None:
        self._t = np.linspace(self._t_range[0], self._t_range[1], self._res)
        self._expectation_value = None

    def _line_data_init(self):
        self._time_evolution = self._axis.plot(self._t, self._expectation_value(self._t), color=self._color, linewidth=self._lw, linestyle=self._ls, animated=True)
        self._time_line = self._axis.axvline(self._time, color=self._c_color, animated=True)
        self._value_at_time = self._axis.axhline(self._expectation_value(self._time), color=self._c_color, animated=True)
        self._lines = self._time_evolution + [self._time_line] + [self._value_at_time]

    def _line_data_updater(self):
        self._time_evolution[0].set_data(self._t, self._expectation_value(self._t))
        self._time_line.set_data([self._time, self._time],[0,1])
        expectation_value = self._expectation_value(self._time)
        self._value_at_time.set_data([1,0],[expectation_value, expectation_value])
        self._lines = self._time_evolution + [self._time_line] + [self._value_at_time]
    
class Position_Expectation_Value_Evolution(Expectation_Value_Evolution):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None) -> None:
        super().__init__(state, fig, axis=axis)

    def update(self) -> None:
        self._t = np.linspace(self._t_range[0], self._t_range[1], self._res)
        self._expectation_value = lambda t: self._state.x_space_expectation_value(t)

class Momentum_Expectation_Value_Evolution(Expectation_Value_Evolution):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None) -> None:
        super().__init__(state, fig, axis=axis)

    def update(self) -> None:
        self._t = np.linspace(self._t_range[0], self._t_range[1], self._res)
        self._expectation_value = lambda t: self._state.new_k_space_expectation_value(t)

class Update_Plot_Collection(Updatable_Plot):
    def __init__(self, fig, *plots: Single_Updatable_Plot) -> None:
        super().__init__(fig)
        self._plots = plots

    def _anim_generic(self, anim_list):
        def anim(i):
            artists = anim_list[0](i)
            for plot_index in range(1, len(anim_list)):
                artists += anim_list[plot_index](i)
            return artists
        return anim
            
    def _anim_t(self, start, system_time_per_frame) -> Callable[[int], List[plt.Line2D]]:
        anim_list = []
        for plot in self._plots:
            anim_list.append(plot._anim_t(start, system_time_per_frame))
        return self._anim_generic(anim_list)

    def _anim_L(self, start, system_L_per_frame) -> Callable[[int], List[plt.Line2D]]:
        anim_list = []
        for plot in self._plots:
            anim_list.append(plot._anim_L(start, system_L_per_frame))
        return self._anim_generic(anim_list)

    def _anim_gamma(self, start, system_gamma_per_frame) -> Callable[[int], List[plt.Line2D]]:
        anim_list = []
        for plot in self._plots:
            anim_list.append(plot._anim_L(start, system_gamma_per_frame))
        return self._anim_generic(anim_list)

    def set_t(self, time) -> Update_Plot_Collection:
        for plot in self._plots:
            plot.set_t(time)
        return self

    def plot(self):
        for plot in self._plots:
            plot.plot()