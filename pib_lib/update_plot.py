from __future__ import annotations
from typing import Callable, List
from matplotlib.collections import PathCollection
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
    def __init__(self, fig: plt.Figure, identifier: str=None) -> None:
        self._fig = fig
        self._identifier = identifier

    @property
    def identifier(self):
        return self._identifier

    @identifier.setter
    def identifier(self, new_identifier):
        self._identifier = new_identifier


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


class Single_Updatable_Plot(Updatable_Plot):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(fig, identifier=identifier)
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

    @abstractmethod
    def _line_data_updater(self) -> None:
        pass
    
    @abstractmethod
    def _line_data_init(self) -> None:
        pass

    def plot(self) -> plt.Line2D:
        self.update()
        if self._lines:
            self._line_data_updater()
        else:
            self._line_data_init()

    def _init_config(self, arg_dict: dict):
        pass

    def _line_config(self, arg_dict: dict):
        pass
        
    def plot_config(self, **kwargs) -> Single_Updatable_Plot:
        if self._lines:
            self._line_config(kwargs)
        else:
            self._init_config(kwargs)
        return self

    def _anim_t(self, start, system_time_per_frame) -> Callable[[int], List[plt.Line2D]]: 
        self.plot()
        self._time = start
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
            self.update()
            self._line_data_updater()
            return self._lines
        return anim

    def _anim_L(self, start, system_L_per_frame) -> Callable[[int], List[plt.Line2D]]:
        self._state.L = start
        self.plot()
        def anim(i):
            self._state.L = start+i*system_L_per_frame
            self.update()
            self._line_data_updater()
            return self._lines
        return anim

    @property
    def axis(self) -> plt.Axes:
        return self._axis

class Single_Updatable_Line_Base(Single_Updatable_Plot):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier)    
        self._res = 300
        self._color = dark_blue
        self._ls = "-"
        self._lw = 1
        
    def _init_config(self, arg_dict: dict):
        for key, value in arg_dict.items():
            if key=="color":
                self._color = value
            elif key=="linestyle":
                self._ls = value
            elif key=="linewidth":
                self._lw = value
    
    def _line_config(self, arg_dict: dict):
        for key, value in arg_dict.items():
            if key=="color":
                self._color = value
                self._lines[0].set_color(value)
            elif key=="linestyle":
                self._ls = value
                self._lines[0].set_linestyle(value)
            elif key=="linewidth":
                self._lw = value
                self._lines[0].set_linewidth(value)

    def plot(self) -> plt.Line2D:
        super().plot()
        return self._lines[0]

    @property
    def line(self) -> plt.Line2D:
        return self._lines[0]

    @property
    def res(self) -> int:
        return self._res

    @res.setter
    def res(self, new_res) -> None:
        self._res = new_res

class Single_Updatable_Vertical_Line(Single_Updatable_Plot):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier)
        self._line = None
        self._x = None
        self._color = "0.0"
        self._lw = 1
        self._ls= "--"

    def _line_data_init(self) -> None:
        self._line = self._axis.axvline(self._x(self._time), color=self._color, linestyle=self._ls, linewidth=self._lw, animated=True)
        self._lines = [self._line]

    def _line_data_updater(self) -> None:
        self._line.set_data([self._x(self._time), self._x(self._time)], [0, 1])    
        self._lines = [self._line]

    def _init_config(self, arg_dict: dict):
        for key, value in arg_dict.items():
            if key=="color":
                self._color = value
            elif key=="linestyle":
                self._ls = value
            elif key=="linewidth":
                self._lw = value
    
    def _line_config(self, arg_dict: dict):
        for key, value in arg_dict.items():
            if key=="color":
                self._color = value
                self._line.set_color(value)
            elif key=="linestyle":
                self._ls = value
                self._line.set_linestyle(value)
            elif key=="linewidth":
                self._lw = value
                self._line.set_linewidth(value)

    def plot(self) -> plt.Line2D:
        super().plot()
        return self._line

    @property
    def line(self) -> plt.Line2D:
        return self._line



class Single_Updatable_Line(Single_Updatable_Line_Base):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier) 
        self._x = None
        self._y = None

    def _line_data_init(self) -> None:
        self._lines = self._axis.plot(self._x, self._y(self._time), c=self._color, lw=self._lw, ls=self._ls, animated=True)
    
    def _line_data_updater(self) -> None:
        self._lines[0].set_data(self._x, self._y(self._time))

class Position_Space_Plot(Single_Updatable_Line):
    def __init__(self, state: pib.Particle_in_Box_State, mode:str, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier)
        self.update()
        self.set_mode(mode)
        
    def update(self) -> None:
        self._x = np.linspace(-self._state.L/2, self._state.L/2, self._res)

    def set_mode(self, new_mode) -> Position_Space_Plot:
        self._mode = new_mode
        if self._mode=="abs_square":
            self._y = lambda t: np.abs(self._state.x_space_wavefunction(self._x, t))**2
        elif self._mode=="real":
            self._y = lambda t: np.real(self._state.x_space_wavefunction(self._x, t))
        elif self._mode=="imag":
           self._y = lambda t: np.imag(self._state.x_space_wavefunction(self._x, t))
        return self

class Momentum_Space_Plot(Single_Updatable_Line):
    def __init__(self, state: pib.Particle_in_Box_State, mode:str, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier)
        self._n_bound = int(max(self._state._sp.energy_states)*1.5)
        self._deviation_factor = 1.0001
        self.update()
        self.set_mode(mode)

    def set_n_bound(self, new_bound) -> Momentum_Space_Plot:
        self._n_bound = new_bound
        self.update()
        return self

    def update(self) -> None:
        k_bound = self._state.boundary_lib.get_kn(self._n_bound*self._deviation_factor)
        self._x = np.linspace(-k_bound, k_bound, self._res)

    def set_mode(self, new_mode) -> None:
        self._mode = new_mode
        if self._mode=="abs_square":
            self._y = lambda t: np.abs(self._state.k_space_wavefunction(self._x, t))**2
        elif self._mode=="real":
            self._y = lambda t: np.real(self._state.k_space_wavefunction(self._x, t))
        elif self._mode=="imag":
           self._y = lambda t: np.imag(self._state.k_space_wavefunction(self._x, t))
        return self

class New_Momentum_Space_Plot(Single_Updatable_Plot):
    def __init__(self, state: pib.Particle_in_Box_State, mode:str, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier)
        self._bound = int(max(self._state._sp.energy_states)*1.5)
        self._bars = None
        self._color = light_blue
        self.update()
        self.set_mode(mode)

    def _init_config(self, arg_dict: dict):
        for key, value in arg_dict.items():
            if key=="color":
                self._color = value

    def _line_config(self, arg_dict: dict):
        for key, value in arg_dict.items():
            if key=="color":
                self._color = value
                self._bars.set_color(value)
                

    def set_n_bound(self, new_bound) -> New_Momentum_Space_Plot:
        self._bound = new_bound
        self.update()
        return self

    def update(self) -> None:
        self._n = np.arange(-self._bound, self._bound+1, 1, dtype=int)
        self._kn = self._state.boundary_lib.get_kn(self._n)
        self._barwidth = 0.8*(self._kn[1]-self._kn[0])

    def _line_data_init(self) -> None:
        self._bars = self._axis.bar(self._kn, self._y(self._time), self._barwidth, color=self._color, animated=True)
        self._lines = list(self._bars)

    def _line_data_updater(self) -> None:
        for bar, h in zip(self._bars, self._y(self._time)):
            bar.set_height(h)
        self._lines = list(self._bars)

    def plot(self) -> BarContainer:
        super().plot()
        return self._bars

    def set_mode(self, new_mode) -> New_Momentum_Space_Plot:
        self._mode = new_mode
        if self._mode=="abs_square":
            self._y = lambda t: np.abs(self._state.new_k_space_wavefunction(self._n, t)**2)
        elif self._mode=="real":
            self._y = lambda t: np.real(self._state.new_k_space_wavefunction(self._n, t))
        elif self._mode=="imag":
           self._y = lambda t: np.imag(self._state.new_k_space_wavefunction(self._n, t))

    @property
    def bars(self) -> BarContainer:
        return self._bars



class Position_Expectation_Value(Single_Updatable_Vertical_Line):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier) 
        self._x = lambda t: self._state.x_space_expectation_value(t)

class Momentum_Expectation_Value(Single_Updatable_Vertical_Line):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier) 
        self._x = lambda t: self._state.new_k_space_expectation_value(t)


class Expectation_Value_Evolution(Single_Updatable_Line_Base):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier) 
        self.set_t_range([0,10])
        self._time_evolution = None
        self._expectation_value = None
        
    def set_t_range(self, new_range):
        self._t_range = new_range
        self._t = np.linspace(self._t_range[0], self._t_range[1], self._res)
        return self
        
    def _line_data_init(self):
        self._time_evolution = self._axis.plot(self._t, self._expectation_value(self._t), color=self._color, linewidth=self._lw, linestyle=self._ls, animated=True)
        self._lines = self._time_evolution

    def _line_data_updater(self):
        self._time_evolution[0].set_data(self._t, self._expectation_value(self._t))
        self._lines = self._time_evolution 

    @property
    def line(self) -> plt.Line2D:
        return self._time_evolution
        
class Position_Expectation_Value_Evolution(Expectation_Value_Evolution):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier)
        self._expectation_value = lambda t: self._state.x_space_expectation_value(t)
        
class Momentum_Expectation_Value_Evolution(Expectation_Value_Evolution):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier)
        self._expectation_value = lambda t: self._state.new_k_space_expectation_value(t)

class Pos_Exp_Deriv_Evolution(Expectation_Value_Evolution):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier)
        self._expectation_value = lambda t: self._state.x_space_expectation_value_derivative(t)


class Expectation_Value_Evolution_Markers(Single_Updatable_Plot):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier)
        self._marker = None
        self._expectation_value = None
        self._color = "0.0"
        self._lw = 1
        self._size = 200
    
    def _init_config(self, arg_dict: dict):
        for key, value in arg_dict.items():
            if key=="color":
                self._color = key
            elif key=="thickness":
                self._lw = value
            elif key=="size":
                self._size = value

    def _line_config(self, arg_dict: dict):
        for key, value in arg_dict.items():
            if key=="color":
                self._color = value
                self._marker.set_color(value)
            elif key=="thickness":
                self._lw = value
                self._marker.set_linewidth(value)
            elif key=="size":
                self._size = value
                self._marker.set_sizes([value])

    def _line_data_init(self) -> None:
        self._marker = self._axis.scatter(self._time, self._expectation_value(self._time), s=self._size, c=self._color, marker="+", linewidths=[self._lw])
        self._lines = [self._marker]

    def _line_data_updater(self) -> None:
        self._marker.set_offsets([self._time, self._expectation_value(self._time)])
        self._lines = [self._marker]

    @property
    def marker(self) -> PathCollection:
        return self._marker

class Position_Expectation_Value_Marker(Expectation_Value_Evolution_Markers):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis)
        self._expectation_value = lambda t: self._state.x_space_expectation_value(t)

class Momentum_Expectation_Value_Marker(Expectation_Value_Evolution_Markers):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis)
        self._expectation_value = lambda t: self._state.new_k_space_expectation_value(t)

class Expectation_Value_Evolution_Time(Single_Updatable_Vertical_Line):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier)
        self._x = lambda t: t


class Energy_Space_Plot(Single_Updatable_Plot):
    def __init__(self, state: pib.Particle_in_Box_State, fig: plt.Figure, axis: plt.Axes=None, identifier: str=None) -> None:
        super().__init__(state, fig, axis=axis, identifier=identifier)
        self._bars = None
        self._color = light_blue
        self.update()

    def update(self) -> None:
        self._l = self._state.energy_space_wavefunction.energies
        self._cl = self._state.energy_space_wavefunction.energy_projection_coefficients
        self._barwidth = 0.8*(self._l[1]-self._l[0]) if len(self._l) > 1 else 0.8

    def _line_data_init(self) -> None:
        self._bars = self._axis.bar(self._l, self._cl, self._barwidth, color=self._color, animated=True)
        self._lines = list(self._bars)

    def _line_data_updater(self) -> None:
        for bar, h in zip(self._bars, self._cl):
            bar.set_height(h)
        self._lines = list(self._bars)

    def plot(self) -> plt.Line2D:
        super().plot()
        return self._bars

    @property
    def bars(self) -> BarContainer:
        return self._bars
        

class Update_Plot_Collection(Updatable_Plot):
    def __init__(self, fig: plt.Figure, *plots: Single_Updatable_Plot, identifier: str=None) -> None:
        super().__init__(fig, identifier=identifier)
        self._plots = list(plots)
        for plot in self._plots:
            if plot._fig != fig:
                self._plots.remove(plot)
    
    def add_plot(self, plot: Single_Updatable_Plot) -> None:
        self._plots.append(plot)

    def remove_plot(self, identifier: str) -> None:
        for plot in self._plots:
            if plot.identifier==identifier:
                self._plots.remove(plot)

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
            anim_list.append(plot._anim_gamma(start, system_gamma_per_frame))
        return self._anim_generic(anim_list)

    def set_t(self, time) -> Update_Plot_Collection:
        for plot in self._plots:
            plot.set_t(time)
        return self

    def set_t_range(self, new_range) -> Update_Plot_Collection:
        for plot in self._plots:
            if isinstance(plot, Expectation_Value_Evolution):
                plot.set_t_range(new_range)
        return self

    def set_n_bound(self, new_bound) -> Update_Plot_Collection:
        for plot in self._plots:
            if isinstance(plot, (New_Momentum_Space_Plot, Momentum_Space_Plot, Update_Plot_Collection)):
                plot.set_n_bound(new_bound)
        return self

    def plot(self):
        for plot in self._plots:
            plot.plot()

    def __getitem__(self, key: str):
        for plot in self._plots:
            if plot._identifier==key:
                return plot

def postion_space_plot(state: pib.Particle_in_Box_State, fig: plt.Figure, 
                                ax: plt.Axes=None, abs_square=True, real_part=False, 
                                imag_part=False, expectation_value=True) -> Update_Plot_Collection:
    
    colors = [dark_blue, mid_blue, light_blue]
    plots = Update_Plot_Collection(fig, identifier="pos_space_plot")

    if not ax:
        ax = fig.add_subplot()
    if imag_part:
        plots.add_plot(Position_Space_Plot(state, "imag", fig, ax, "imag_part").plot_config(color=colors[2]))
    if real_part:
        plots.add_plot(Position_Space_Plot(state, "real", fig, ax, "real_part").plot_config(color=colors[1]))
    if abs_square:
        plots.add_plot(Position_Space_Plot(state, "abs_square", fig, ax, "abs_square").plot_config(color=colors[0]))
    if expectation_value:
        plots.add_plot(Position_Expectation_Value(state, fig, ax, "expectation_value"))

    return plots

def momentum_space_plot(state: pib.Particle_in_Box_State, fig: plt.Figure, 
                                ax: plt.Axes=None, abs_square=True, real_part=False, 
                                imag_part=False) -> Update_Plot_Collection:
    
    colors = [dark_blue, mid_blue, light_blue]
    plots = Update_Plot_Collection(fig, identifier="momentum_space_plot")

    if not ax:
        ax = fig.add_subplot()
    if imag_part:
        plots.add_plot(Momentum_Space_Plot(state, "imag", fig, ax, "imag_part").plot_config(color=colors[2]))
    if real_part:
        plots.add_plot(Momentum_Space_Plot(state, "real", fig, ax, "real_part").plot_config(color=colors[1]))
    if abs_square:
        plots.add_plot(Momentum_Space_Plot(state, "abs_square", fig, ax, "abs_square").plot_config(color=colors[0]))

    return plots

def new_momentum_space_plot(state: pib.Particle_in_Box_State, fig: plt.Figure, 
                                ax: plt.Axes=None, abs_square=True, real_part=False, 
                                imag_part=False, expectation_value=True) -> Update_Plot_Collection:
    
    colors = [dark_blue, mid_blue, light_blue]
    plots = Update_Plot_Collection(fig, identifier="new_momentum_space_plot")

    if not ax:
        ax = fig.add_subplot()
    if imag_part:
        plots.add_plot(New_Momentum_Space_Plot(state, "imag", fig, ax, "imag_part").plot_config(color=colors[2]))
    if real_part:
        plots.add_plot(New_Momentum_Space_Plot(state, "real", fig, ax, "real_part").plot_config(color=colors[1]))
    if abs_square:
        plots.add_plot(New_Momentum_Space_Plot(state, "abs_square", fig, ax, "abs_square").plot_config(color=colors[0]))
    if expectation_value:
        plots.add_plot(Momentum_Expectation_Value(state, fig, ax, "expectation_value"))

    return plots

def expectation_value_evolution(state: pib.Particle_in_Box_State, fig: plt.Figure, ax: plt.Axes=None,
                                position_expectation_value=True, momentum_expectation_value=True,
                                time_marker_position=False, time_marker_momentum=False,
                                time_indicator_line=False):
    colors = ["0.0", "0.3"]
    plots = Update_Plot_Collection(fig, identifier="expectation_value_evolution")

    if not ax:
        ax = fig.add_subplot()  
    if momentum_expectation_value:
        plots.add_plot(Momentum_Expectation_Value_Evolution(state, fig, ax, "momentum_expectation_value").plot_config(color=colors[1]))
    if position_expectation_value:
        plots.add_plot(Position_Expectation_Value_Evolution(state, fig, ax, "position_expectation_value").plot_config(color=colors[0]))
    if time_marker_position:
        plots.add_plot(Position_Expectation_Value_Marker(state, fig, ax, "time_marker_position"))
    if time_marker_momentum:
        plots.add_plot(Momentum_Expectation_Value_Marker(state, fig, ax, "time_marker_momentum"))
    if time_indicator_line:
        plots.add_plot(Expectation_Value_Evolution_Time(state, fig, ax, "time_indicator_line"))

    return plots

