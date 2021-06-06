from __future__ import annotations
from typing import List
from typing import Callable
import numpy as np
from abc import ABC, abstractmethod, abstractproperty

class Function_Base(ABC):
    def __init__(self, function: Callable) -> None:
        self._function = function

    @abstractmethod
    def __call__(self, args):
        pass

    @abstractmethod
    def __add__(self, other: Function_Base) -> Function_Base:
        pass

    def __neg__(self) -> Function_Base:
        return self.__mul__(-1)

    def __sub__(self, other: Function_Base) -> Function_Base:
        return self.__add__((-1)*other)

    @abstractmethod
    def __mul__(self, other) -> Function_Base:
        pass

    def __rmul__(self, other) -> Function_Base:
        return self.__mul__(other)

    @abstractmethod
    def get_real_part(self):
        pass

    @abstractmethod
    def get_imag_part(self):
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

    def get_real_part(self):
        return None

    def get_imag_part(self):
        return None


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

    def get_real_part(self):
        return Function_of_t(lambda t: np.real(self._function(t)))

    def get_imag_part(self):
        return Function_of_t(lambda t: np.imag(self._function(t)))


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

    def get_real_part(self):
        return Function_of_array(lambda n: np.real(self._function(n)))
    
    def get_imag_part(self):
        return Function_of_array(lambda n: np.imag(self._function(n)))


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


    def get_real_part(self):
        return Function_of_array_and_t(lambda x,t: np.real(self._function(x,t)))

    def get_imag_part(self):
        return Function_of_array_and_t(lambda x,t: np.imag(self._function(x,t)))


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
    def __init__(self, L: float, gamma: float, l_to_k_mapper_ref: l_to_kl_mapper) -> None:
        self._L = L
        self._gamma = gamma
        self._l_kl_map = l_to_k_mapper_ref
    
    @abstractmethod
    def get_projection(self, l: int) -> Function_of_array:
        pass

    def set_L(self, L: float) -> None:
        self._L = L
    
    def set_gamma(self, gamma: float) -> None:
        self._gamma = gamma


class Energy_State_Matrix_Elements(ABC):
    def __init__(self, L: float, gamma: float, l_to_k_mapper_ref: l_to_kl_mapper) -> None:
        self._L = L
        self._gamma = gamma
        self._l_kl_map = l_to_k_mapper_ref

    @abstractmethod
    def get_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        pass

    def set_L(self, L: float) -> None:
        self._L = L 

    def set_gamma(self, gamma: float) -> None:
        self._gamma = gamma


class Gamma_to_k_Base(ABC):
    def __init__(self, L: float) -> None:
        self._L = L

    @abstractmethod
    def __call__(self, l: int) -> complex:
        pass

    def set_L(self, L: float) -> None:
        self._L = L


class New_Style_Boundary(ABC):
    def __init__(self, L: float, gamma: float, l_to_kl_mapper_ref: l_to_kl_mapper) -> None:
        self._L = L
        self._gamma = gamma
        self._l_kl_map = l_to_kl_mapper_ref

    
    @abstractmethod
    def get_kl(self, l: int) -> complex:
        pass

    @abstractmethod
    def get_x_space_projection(self, l: int) -> Function_of_array:
        pass

    @abstractmethod
    def get_k_space_projection(self, l: int) -> Function_of_array:
        pass

    @abstractmethod
    def get_new_k_space_projection(self, l: int) -> Function_of_array:
        pass

    @abstractmethod
    def get_x_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        pass

    @abstractmethod
    def get_pR_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        pass

    def set_L(self, new_L: float) -> None:
        print("setting L...(backend)")
        self._L = new_L

    def set_gamma(self, new_gamma: float) -> None:
        self._gamma = new_gamma


class Boundary(ABC):
    def __init__(self, L: float, gamma: float, l_to_kl_mapper_ref: l_to_kl_mapper) -> None:
        pass

    @abstractproperty
    def X_Space_Projector(self) -> Energy_State_Projector:
        pass

    @abstractproperty
    def K_Space_Projector(self) -> Energy_State_Projector:
        pass

    @abstractproperty
    def Bra_l1_x_Ket_l2(self) -> Energy_State_Matrix_Elements:
        pass

    @abstractproperty
    def Bra_l1_pR_Ket_l2(self) -> Energy_State_Matrix_Elements:
        pass

    @abstractproperty
    def Gamma_to_k(self) -> Gamma_to_k_Base:
        pass
    
    def set_L(self, L: float) -> None:
        self.X_Space_Projector.set_L(L)
        self.K_Space_Projector.set_L(L)
        self.Bra_l1_pR_Ket_l2.set_L(L)
        self.Bra_l1_x_Ket_l2.set_L(L)
        self.Gamma_to_k.set_L(L)
    

    def set_gamma(self, gamma: float) -> None:
        self.X_Space_Projector.set_gamma(gamma)
        self.K_Space_Projector.set_gamma(gamma)
        self.Bra_l1_pR_Ket_l2.set_gamma(gamma)
        self.Bra_l1_x_Ket_l2.set_gamma(gamma)
        self.Gamma_to_k.set_gamma(gamma)

    