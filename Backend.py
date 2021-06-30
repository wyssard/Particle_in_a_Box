from __future__ import annotations
from typing import List
from typing import Callable
import numpy as np
from abc import ABC, abstractmethod


class Function_Base(ABC):
    """
    Abstract base class for function like objects that mimic behaviour of 
    mathematical functions such as addition and multiplication of functions
    """

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
    def get_real_part(self) -> Function_Base:
        """return a new instance of <Function_Base> that only contains the
        real part of the function"""
        pass

    @abstractmethod
    def get_imag_part(self) -> Function_Base:
        """return a new instance of <Function_Base> that only contains the
        imaginary part of the function"""
        pass


class None_Function(Function_Base):
    """
    Instances of this class can be used as prototypical placeholder objects
    that are later to be overwriten by more complex <Function_Base> objects.
    The main application of this class is to provide trivial function objects
    that do not represent a mathematical function but can be incremented 
    with other Function_Base objects. 

    Examples
    -------
    >>> f = None_Function()
    >>> f += g
    >>> f(x) == g(x)
    True

    for g being an arbtirary function object derived from <Function_Base> and
    x being an arbitrary value within the domain of g
    """

    def __init__(self):
        Function_Base.__init__(self, None)

    def __call__(self):
        return None

    def __add__(self, other: Function_Base) -> Function_Base:
        return other

    def __mul__(self) -> None_Function:
        return self

    def get_real_part(self):
        return None

    def get_imag_part(self):
        return None


class Function_of_t(Function_Base):
    """
    Function like object to mimic algebraic behaviour of mathematical functions 
    f: R -> C, t -> f(t) for t being the time variable.
    Functions of this type can be added and multiplied together and can also be 
    multiplied with functions of type <Function_of_n> and <Function_of_n_and_t>
    """

    def __init__(self, function: Callable[[np.ndarray], np.ndarray]):
        Function_Base.__init__(self, function)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        return self._function(t)

    def __add__(self, other: Function_Base) -> Function_Base:
        if isinstance(other, None_Function):
            return self

        elif isinstance(other, Function_of_t):
            return Function_of_t(lambda t: self._function(t) + other._function(t))

        else:
            return NotImplemented

    def __mul__(self, other: Function_Base) -> Function_Base:
        if isinstance(other, Function_of_t):
            return Function_of_t(lambda t: self._function(t)*other._function(t))

        elif isinstance(other, Function_of_n):
            return Function_of_n_and_t(lambda n, t: self._function(t)*other._function(n))

        elif isinstance(other, Function_of_n_and_t):
            return Function_of_n_and_t(lambda n, t: self._function(t)*other._function(n, t))

        elif isinstance(other, (int, float, complex)):
            return Function_of_t(lambda t: other*self._function(t))

        elif isinstance(other, None_Function):
            return other

        else:
            return NotImplemented

    def get_real_part(self) -> Function_of_t:
        return Function_of_t(lambda t: np.real(self._function(t)))

    def get_imag_part(self) -> Function_of_t:
        return Function_of_t(lambda t: np.imag(self._function(t)))


class Function_of_n(Function_Base):
    """
    Function like object to mimic algebraic behaviour of mathematical functions 
    f: R -> C, n -> f(n) for n being the position or the momentum variable.
    Functions of this type can be added and multiplied together and can also be 
    multiplied with functions of type <Function_of_t> and <Function_of_n_and_t>
    """

    def __init__(self, function: Callable[[np.ndarray], np.ndarray]):
        Function_Base.__init__(self, function)

    def __call__(self, n: np.ndarray) -> np.ndarray:
        return self._function(n)

    def __add__(self, other: Function_Base) -> Function_of_n:
        if isinstance(other, None_Function):
            return self

        elif isinstance(other, Function_of_n):
            return Function_of_n(lambda n: self._function(n) + other._function(n))

        else:
            return NotImplemented

    def __mul__(self, other: Function_Base) -> Function_Base:
        if isinstance(other, Function_of_t):
            return Function_of_n_and_t(lambda n, t: self._function(n)*other._function(t))

        elif isinstance(other, Function_of_n):
            return Function_of_n(lambda n: self._function(n)*other._function(n))

        elif isinstance(other, Function_of_n_and_t):
            return Function_of_n_and_t(lambda n, t: self._function(n)*other._function(n, t))

        elif isinstance(other, (int, float, complex)):
            return Function_of_n(lambda n: other*self._function(n))

        elif isinstance(other, None_Function):
            return other

        else:
            return NotImplemented

    def get_real_part(self) -> Function_of_n:
        return Function_of_n(lambda n: np.real(self._function(n)))

    def get_imag_part(self) -> Function_of_n:
        return Function_of_n(lambda n: np.imag(self._function(n)))


class Function_of_n_and_t(Function_Base):
    """
    Function like object to mimic algebraic behaviour of mathematical functions 
    f: R x R -> C, (n, t) -> f(n, t) for n being the position or the momentum 
    variable and t being the time.
    Functions of this type can be added and multiplied together and can also be 
    multiplied with functions of type <Function_of_t> and <Function_of_n>
    """

    def __init__(self, function: Callable[[np.ndarray, np.ndarray], np.ndarray]):
        Function_Base.__init__(self, function)

    def __call__(self, n: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self._function(n, t)

    def __add__(self, other: Function_Base) -> Function_Base:
        if isinstance(other, Function_of_n_and_t):
            return Function_of_n_and_t(lambda n, t: self._function(n, t) + other._function(n, t))

        elif isinstance(other, None_Function):
            return self

        else:
            return NotImplemented

    def __mul__(self, other: Function_Base) -> Function_Base:
        if isinstance(other, Function_of_t):
            return Function_of_n_and_t(lambda n, t: self._function(n, t)*other._function(t))

        elif isinstance(other, Function_of_n):
            return Function_of_n_and_t(lambda n, t: self._function(n, t)*other._function(n))

        elif isinstance(other, Function_of_n_and_t):
            return Function_of_n_and_t(lambda n, t: self._function(n, t)*other._function(n, t))

        elif isinstance(other, (complex, float, int)):
            return Function_of_n_and_t(lambda n, t: other*self._function(n, t))

        elif isinstance(other, None_Function):
            return other

        else:
            return NotImplemented

    def get_real_part(self) -> Function_of_n_and_t:
        return Function_of_n_and_t(lambda x, t: np.real(self._function(x, t)))

    def get_imag_part(self) -> Function_of_n_and_t:
        return Function_of_n_and_t(lambda x, t: np.imag(self._function(x, t)))


class Wiggle_Factor(Function_of_t):
    """
    Child class of <Function_of_t> that represents 'wiggle factors' that is
    functions of the form: f(t) = exp(-iEt) for 'E' being the energy of an
    energy eigenstate.
    """

    def __init__(self, energy: float):
        self._energy = energy
        Function_of_t.__init__(self, lambda t: np.exp(-1j*self._energy*t))


class l_to_kl_mapper:
    """
    Mapper class to simplify the process of retrieving the 'k_l' value from a 
    given energy quantum number 'l'. It does so, by storing references to both
    the unorderd list that contains the energy states 'l' and to the 
    correspondingly ordered list that contains the respective 'k_l' values.
    Thus no recalculation of 'k_l' is required at access.
    """

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


class Energy_Space_Wavefunction:
    """
    Container class that consists of the lists that store the energies and the
    corresponding projection coefficients of the energy eigenstates which our
    'particle in a box'-state is decomposed into.
    """

    def __init__(self, energies: list, energy_proj_coeffs: np.ndarray) -> None:
        self._energies = energies
        self._energy_proj_coeffs = energy_proj_coeffs

    @property
    def energies(self) -> list:
        return self._energies

    @property
    def energy_projection_coefficients(self) -> np.ndarray:
        return self._energy_proj_coeffs


class New_Style_Boundary(ABC):
    """
    Template class for other classes that represent the boundary conditions of 
    the 'particle in a box'-state.
    Such classes contain any parameters (attributes) and functionality 
    (behaviour) that depends on the specific chioce of the boundary conditions
    for the 'particle in a box'-state.
    """

    def __init__(self, L: float, gamma: float, l_to_kl_mapper_ref: l_to_kl_mapper) -> None:
        self._L = L
        self._gamma = gamma
        self._l_kl_map = l_to_kl_mapper_ref

    @abstractmethod
    def get_kn(self, n: int | list) -> float | list:
        """Method to compute the momentum 'k_n' form the given momentum quantum
        number 'n'
        """
        pass

    @abstractmethod
    def get_kl(self, l: int) -> complex:
        """Method to compute the 'k_l' value from the given energy qunatum
        number 'l' where the corresponding energy is given by 
        E_l = 2*m*(k_l)^2 for m being the mass of the particle
        """
        pass

    @abstractmethod
    def get_x_space_projection(self, l: int) -> Function_of_n:
        """Method to project the energy eigenstate described by the quantum
        number 'l' onto position space. Thus this method returns < x | l > which
        is a function in the position x as an instance of <Function_of_n>.
        """
        pass

    @abstractmethod
    def get_k_space_projection(self, l: int) -> Function_of_n:
        """Method to project the energy eigenstate described by the quantum
        number 'l' onto the conventional momentum space (spanned by the
        eigenfunctions of the operator -idx). Thus this method returns < k | l > 
        which is a function in the conventional momentum k as an instance of 
        <Function_of_n>.
        """
        pass

    @abstractmethod
    def get_new_k_space_projection(self, l: int) -> Function_of_n:
        """Method to project the energy eigenstate described by the quantum
        number 'l' onto the 'self adjoint' momentum space (spanned by the
        eigenfunctions of the self adjoint momentum p_R). Thus this method 
        returns < n | l > which is a function in the eigenvalues k_n of the
        self adjoint momentum as an instance of <Function_of_n>.
        """
        pass

    @abstractmethod
    def get_x_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        """Method to compute the matrix elements < lhs_state | x | rhs_state >
        where 'lhs_state' and 'rhs_state' are the quantum numbers for the 
        corresponding energy eigenstates.
        """
        pass

    @abstractmethod
    def get_pR_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        """Method to compute the matrix elements < lhs_state | p_R | rhs_state >
        where 'lhs_state' and 'rhs_state' are the quantum numbers for the 
        corresponding energy eigenstates and for p_R being the self adjoint 
        momentum operator
        """
        pass

    def set_L(self, new_L: float) -> None:
        self._L = new_L

    def set_gamma(self, new_gamma: float) -> None:
        self._gamma = new_gamma
