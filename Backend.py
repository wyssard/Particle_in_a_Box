from __future__ import annotations
from typing import List
from typing import Callable
import numpy as np
from abc import ABC, abstractmethod

class Function_Base(ABC):
    def __init__(self, function: Callable):
        self._function = function

    @abstractmethod
    def __call__(self, args):
        pass

    @abstractmethod
    def __add__(self, other):
        pass
    
    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __rmul__(self, other):
        pass


class None_Function(Function_Base):
    def __init__(self):
        Function_Base.__init__(self, None)

    def __call__(self):
        return self._function

    def __add__(self, other: Function_Base):
        return other
    
    def __mul__(self):
        return self

    def __rmul__(self):
        return self


class Function_of_t(Function_Base):
    def __init__(self, function: Callable[[float], float]):
        Function_Base.__init__(self, function)

    def __call__(self, t: float):
        return self._function(t)
    
    def __add__(self, other: Function_Base):
        return Function_of_t(lambda t: self._function(t) + other._function(t))

        
    def __mul__(self, other):
        if isinstance(other, Function_of_t):
            return Function_of_t(lambda t: self._function(t)*other._function(t))
        elif isinstance(other, Function_of_array):
            return Function_of_array_and_t(lambda n,t: self._function(t)*other._function(n))
        elif isinstance(other, (int, float, complex)):
            return Function_of_t(lambda t: other*self._function(t))

    def __rmul__(self, other):
        return self.__mul__(other)


    def get_real_part(self):
        return Function_of_t(lambda t: np.real(self._function(t)))


class Function_of_array(Function_Base):
    def __init__(self, function: Callable[[np.ndarray], np.ndarray]):
        Function_Base.__init__(self, function)

    def __call__(self, n: np.ndarray):
        return self._function(n)

    def __add__(self, other: Function_Base):
        return Function_of_array(lambda n: self._function(n) + other._function(n))

    def __mul__(self, other):
        if isinstance(other, Function_of_t):
            return Function_of_array_and_t(lambda n,t: self._function(n)*other._function(t))
        elif isinstance(other, Function_of_array):
            return Function_of_array(lambda n: self._function(n)*other._function(n))
        elif isinstance(other, (int, float, complex)):
            return Function_of_array(lambda n: other*self._function(n))

    def __rmul__(self, other):
        return self.__mul__(other)


class Function_of_array_and_t(Function_Base):
    def __init__(self, function: Callable[[np.ndarray, float], float]):
        Function_Base.__init__(self, function)

    def __call__(self, x: np.ndarray, t: float):
        return self._function(x, t)
        
    def __add__(self, other):
        if isinstance(other, None_Function):
            return self
        else:
            return Function_of_array_and_t(lambda x,t: self._function(x,t) + other._function(x,t))

    def __mul__(self, other):
        if isinstance(other, Function_of_array_and_t):
            return Function_of_array_and_t(lambda x, t: self._function(x,t)*other._function(x,t))
        if isinstance(other, (complex, float, int)):
            return Function_of_array_and_t(lambda x, t: other*self._function(x,t))
        if isinstance(other, Wiggle_Factor):
            return Function_of_array_and_t(lambda x,t: other(t)*self._function(x,t))
    
    def __rmul__(self, other):
        return self.__mul__(other)


class Wiggle_Factor(Function_of_t):
    def __init__(self, energy: float):
        self._energy = energy
        Function_of_t.__init__(self, lambda t: np.exp(-1j*self._energy*t))


class l_to_kl_mapper:
    def __init__(self, energy_states: List[int], k_kappa_l: List[complex]) -> None:
        self._energy_states = energy_states
        self._k_kappa_l_array = k_kappa_l

    def get_kl(self, l: int) -> complex:
        return self._k_kappa_l_array[self._energy_states.index(l)]

    def get_index(self, l: int) -> int:
        return self._energy_states.index(l)
    
    @property
    def energy_states(self) -> List[int]:
        return self._energy_states

    @property
    def k_kappa_l(self) -> List[complex]:
        return self._k_kappa_l_array


class Energy_State_Projector(ABC):
    def __init__(self, L: float, l_to_k_mapper_ref: l_to_kl_mapper) -> None:
        self._L = L
        self._l_kl_map = l_to_k_mapper_ref
    
    @abstractmethod
    def get_projection(self, l: int) -> Function_of_array:
        pass

    def set_L(self, L: float) -> None:
        self._L = L


class Energy_State_Matrix_Elements(ABC):
    def __init__(self, L: float, l_to_k_mapper_ref: l_to_kl_mapper) -> None:
        self._L = L
        self._l_kl_map = l_to_k_mapper_ref

    @abstractmethod
    def get_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        pass

    def set_L(self, L: float) -> None:
        self._L = L 


class Gamma_to_k_Base(ABC):
    def __init__(self, L: float) -> None:
        self._L = L

    @abstractmethod
    def __call__(self, l: int) -> complex:
        pass

    @abstractmethod
    def set_gamma(self, gamma: float) -> None:
        pass

    def set_L(self, L: float) -> None:
        self._L = L
