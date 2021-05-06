from operator import index
from types import new_class
from typing import Callable
import numpy as np
from copy import deepcopy
from scipy.optimize import fsolve
from abc import ABC, abstractmethod
from matplotlib import pyplot as plt

posRelEven = lambda g, k: g-np.arctan(k*np.tan(k/2))
posRelOdd = lambda g, k: g+np.arctan(k/(np.tan(k/2)))

negRelEven = lambda g, k: g+np.arctan(k*np.tanh(k/2))
negRelOdd = lambda g, k: g+np.arctan(k/np.tanh(k/2))

psi_l_Pos_odd = lambda L, kl, x: np.sqrt(2/L)*np.power(1+np.sin(kl*L)/(kl*L), -1/2)*np.cos(kl*x)
psi_l_Pos_even = lambda L, kl, x: np.sqrt(2/L)*np.power(1-np.sin(kl*L)/(kl*L), -1/2)*np.sin(kl*x)
psi_l_Neg_odd = lambda L, kappal, x: np.sqrt(2/L)*np.power(1+np.sinh(kappal*L)/(kappal*L), -1/2)*np.cosh(kappal*x)
psi_l_Neg_even = lambda L, kappal, x: np.sqrt(2/L)*np.power(-1+np.sinh(kappal*L)/(kappal*L), -1/2)*np.sinh(kappal*x)

momentum_Proj_Pos_even = lambda L, kl, k: np.sqrt(L/np.pi)/np.sqrt(1-np.sin(kl*L)/(kl*L))*(np.sin((kl+k)*L/2)/(kl*L+k*L) - np.sin((kl-k)*L/2)/(kl*L-k*L))
momentum_Proj_Pos_odd = lambda L, kl, k: np.sqrt(L/np.pi)/np.sqrt(1+np.sin(kl*L)/(kl*L))*(np.sin((kl+k)*L/2)/(kl*L+k*L) + np.sin((kl-k)*L/2)/(kl*L-k*L))
momentum_Proj_Neg_even = lambda L, kappal, k: (2j)*np.sqrt(L/np.pi)/np.sqrt(-1+np.sinh(kappal*L)/(kappal*L))*(k*L*np.cos(k*L/2)*np.sinh(kappal*L/2) - kappal*L*np.sin(k*L/2)*np.cosh(kappal*L/2))/((kappal*L)**2+(k*L)**2)
momentum_Proj_Neg_odd = lambda L, kappal, k: (2)*np.sqrt(L/np.pi)/np.sqrt(1+np.sinh(kappal*L)/(kappal*L))*(k*L*np.cos(k*L/2)*np.sinh(kappal*L/2) + kappal*L*np.sin(k*L/2)*np.cosh(kappal*L/2))/((kappal*L)**2+(k*L)**2)

def gamma_to_k(gamma, l, L):
    gammaPrime = np.arctan(gamma*L)
    length = np.size(gamma)

    if l > 2:
        if l%2 == 0:
            rel = posRelOdd
            #print("Odd Case")
        else:
            rel = posRelEven
            #print("Even Case")

        kGuess = np.full(length, l-1)*np.pi
        kSolve = fsolve(lambda k: rel(gammaPrime, k), kGuess)
        return kSolve/L

    if l == 1:
        gammaGreaterZero = gammaPrime[gammaPrime >= 0]
        gammaSmallerZero = gammaPrime[gammaPrime < 0]

        lGreater = np.size(gammaGreaterZero)

        kGuessPosLowestEven = np.linspace(0.5, 1, lGreater)*np.pi
        KGuessNegLowestEven = -np.tan(gammaSmallerZero)

        kSolvePosLowestEven = np.array([])
        kSolveNegLowestEven = np.array([])

        if np.size(gammaGreaterZero) > 0:
            kSolvePosLowestEven = fsolve(lambda k: posRelEven(gammaGreaterZero, k), kGuessPosLowestEven)
        if np.size(gammaSmallerZero) > 0:
            kSolveNegLowestEven = fsolve(lambda k: negRelEven(gammaSmallerZero, k), KGuessNegLowestEven)
            
        return np.concatenate((kSolveNegLowestEven*1j, kSolvePosLowestEven))/L
        #return {"k" : kSolvePosLowestEven, "kappa" : kSolveNegLowestEven}

    if l == 2:
        gammaGreaterMinusLHlaf = gammaPrime[gammaPrime >= np.arctan(-2)]
        gammaSmallerMinusLHlaf = gammaPrime[gammaPrime < np.arctan(-2)]

        lGreater = np.size(gammaGreaterMinusLHlaf)

        kGuessPosLowestOdd = np.full(lGreater, 1)*np.pi
        kGuessNegLowestOdd = -np.tan(gammaSmallerMinusLHlaf)

        kSolvePosLowestOdd = np.array([])
        kSolveNegLowestOdd = np.array([])

        if np.size(gammaGreaterMinusLHlaf) > 0:
            kSolvePosLowestOdd = fsolve(lambda k: posRelOdd(gammaGreaterMinusLHlaf, k), kGuessPosLowestOdd)
        if np.size(gammaSmallerMinusLHlaf) > 0:
            kSolveNegLowestOdd = fsolve(lambda k: negRelOdd(gammaSmallerMinusLHlaf, k), kGuessNegLowestOdd)

        return np.concatenate((kSolveNegLowestOdd*1j, kSolvePosLowestOdd))/L


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
    

class X_Space_Proj_Pos_Odd(Function_of_array):
    def __init__(self, L, kl):
        self._L = L
        self._kl = kl
        Function_of_array.__init__(self, lambda x: np.sqrt(2/L)*np.power(1+np.sin(kl*L)/(kl*L), -1/2)*np.cos(kl*x))
    
class X_Space_Proj_Pos_Even(Function_of_array):
    def __init__(self, L, kl):
        self._L = L
        self._kl = kl
        Function_of_array.__init__(self, lambda x: np.sqrt(2/L)*np.power(1-np.sin(kl*L)/(kl*L), -1/2)*np.sin(kl*x))

class X_Space_Proj_Neg_Odd(Function_of_array):
    def __init__(self, L, kappal):
        self._L = L
        self._kappal = kappal
        Function_of_array.__init__(self, lambda x: np.sqrt(2/L)*np.power(1+np.sinh(kappal*L)/(kappal*L), -1/2)*np.cosh(kappal*x))
        
class X_Space_Proj_Neg_Even(Function_of_array):
    def __init__(self, L, kappal):
        self._L = L
        self._kappal = kappal
        Function_of_array.__init__(self, lambda x: np.sqrt(2/L)*np.power(-1+np.sinh(kappal*L)/(kappal*L), -1/2)*np.sinh(kappal*x))


class K_Space_Proj_Pos_Even(Function_of_array):
    def __init__(self, L, kl):
        Function_of_array.__init__(self, lambda k: np.sqrt(L/np.pi)/np.sqrt(1-np.sin(kl*L)/(kl*L))*(np.sin((kl+k)*L/2)/(kl*L+k*L) - np.sin((kl-k)*L/2)/(kl*L-k*L)))

class K_Space_Proj_Pos_Odd(Function_of_array):
    def __init__(self, L, kl):
        Function_of_array.__init__(self, lambda k: np.sqrt(L/np.pi)/np.sqrt(1+np.sin(kl*L)/(kl*L))*(np.sin((kl+k)*L/2)/(kl*L+k*L) + np.sin((kl-k)*L/2)/(kl*L-k*L)))

class K_Space_Proj_Neg_Even(Function_of_array):
    def __init__(self, L, kappal):
        Function_of_array.__init__(self, lambda k: (2j)*np.sqrt(L/np.pi)/np.sqrt(-1+np.sinh(kappal*L)/(kappal*L))*(k*L*np.cos(k*L/2)*np.sinh(kappal*L/2) - kappal*L*np.sin(k*L/2)*np.cosh(kappal*L/2))/((kappal*L)**2+(k*L)**2))

class K_Space_Proj_Neg_Odd(Function_of_array):
    def __init__(self, L, kappal):
        Function_of_array.__init__(self, lambda k: (2)*np.sqrt(L/np.pi)/np.sqrt(1+np.sinh(kappal*L)/(kappal*L))*(k*L*np.cos(k*L/2)*np.sinh(kappal*L/2) + kappal*L*np.sin(k*L/2)*np.cosh(kappal*L/2))/((kappal*L)**2+(k*L)**2))





class Energy_Space_Projection:
    def __init__(self, L, gamma, m, energy_states, energy_proj_coeffs) -> None:
        self._gamma = gamma
        self._L = L
        self._m = m
        self._energy_states = []
        self._energies = []
        self._wiggle_factors = []
        self._energy_proj_coeffs = np.array([])
        self._num_energy_states = 0
        self._k_kappa_l = []
        self.add_states(energy_states, energy_proj_coeffs)

    def full_recompute(self):
        for l in range(self._num_energy_states):
            self._k_kappa_l[l] = gamma_to_k(self._gamma, self._energy_states[l], self._L)[0]
            self._energies[l] = (self._k_kappa_l[l]**2)/(2*self._m)
            self._wiggle_factors[l] = Wiggle_Factor(self._energies[l])

    def normalize(self) -> None:
        if self._num_energy_states == 0:
            return 0
        else:
            norm = np.sum(np.power(np.abs(self._energy_proj_coeffs), 2))
            self._energy_proj_coeffs = self._energy_proj_coeffs*(1/np.sqrt(norm))
    
    def add_states(self, the_states: list, their_coeffs: np.ndarray) -> None:
        if isinstance(the_states, int):
            the_states = [the_states]
            print("single state converted to list: ", the_states)
            their_coeffs = np.array([their_coeffs])
        
        self._num_energy_states += len(the_states)
        self._energy_proj_coeffs = np.append(self._energy_proj_coeffs, their_coeffs)
        self.normalize()

        for state in the_states:
            self._energy_states.append(state)
            k_kappa_to_append = gamma_to_k(self._gamma, state, self._L)[0]
            energy_to_append = (k_kappa_to_append**2)/(2*self._m)

            self._k_kappa_l.append(k_kappa_to_append)
            self._energies.append(energy_to_append)
            self._wiggle_factors.append(Wiggle_Factor(energy_to_append))

    def remove_states(self, the_states: list) -> None:
        if isinstance(the_states, int):
            the_states = [the_states]
            print("single state converted to list: ", the_states)
        
        self._num_energy_states -= len(the_states)
        
        for state in the_states:
            index = self._energy_states.index(state)
            self._energy_proj_coeffs = np.delete(self._energy_proj_coeffs, index)
            self._k_kappa_l.pop(index)
            self._energies.pop(index)
            self._wiggle_factors.pop(index)
            self._energy_states.remove(state)

        self.normalize()

    def __str__(self) -> str:
        fm = "[{0:2.3f} * exp(-i*{1:2.3f}*t) * |{2:}>]"
        output = fm.format(self._energy_proj_coeffs[0], self._energies[0], self._energy_states[0])
        for i in range(1, self._num_energy_states):
            output += " + " + fm.format(self._energy_proj_coeffs[i], self._energies[i], self._energy_states[i])
        return output

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, gamma: float) -> None:
        self._gamma = gamma
        self.full_recompute()

    @property
    def L(self) -> float:
        return self._L

    @L.setter
    def L(self, L: float) -> None:
        self._L = L
        self.full_recompute()

    @property
    def m(self) -> float:
        return self._m

    @m.setter
    def m(self, m) -> None:
        self._m = m
        for l in range(self._num_energy_states):
            self._energies[l] = (self._k_kappa_l[l]**2)/(2*self._m)
            self._wiggle_factors[l] = Wiggle_Factor(self._energies[l])



class New_Momentum_Space_Projection:
    def __init__(self, new_k_space_wavefunction: Function_of_array_and_t, new_k_space_single_energy_proj: list) -> None:
        self._new_k_space_wavefunction = new_k_space_wavefunction
        self._new_k_space_single_energy_proj = new_k_space_single_energy_proj

class Momentum_Space_Projection:
    def __init__(self, cont_k_space_wavefunction: Function_of_array_and_t, cont_k_space_single_energy_proj: list) -> None:
        self._cont_k_space_wavefunction = cont_k_space_wavefunction
        self._cont_k_space_single_energy_proj = cont_k_space_single_energy_proj

class Position_Space_Projection:
    def __init__(self, x_space_wavefunction: Function_of_array_and_t, x_space_single_energy_proj: list) -> None:
        self._x_space_wavefunction = x_space_wavefunction
        self._x_space_single_energy_proj = x_space_single_energy_proj


class Projection_Handler:
    def __init__(self, energy_space_projection: Energy_Space_Projection, x_space_proj: Position_Space_Projection, k_space_proj: Momentum_Space_Projection, new_k_space_proj: New_Momentum_Space_Projection) -> None:
        self._esp = energy_space_projection
        self._xsp = x_space_proj
        self._new_ksp = new_k_space_proj
        self._ksp = k_space_proj
        self._conversion_factor_k_to_new_k = np.sqrt(np.pi/self._esp._L)
        self.generate_x_and_k_projections()

    def compute_x_space_proj_component(self, the_state: int) -> Function_of_array:
        index = self._esp._energy_states.index(the_state)
        the_k = self._esp._k_kappa_l[index]
        
        if the_state%2 == 0:
            if np.imag(the_k)==0:
                psi_to_append = X_Space_Proj_Pos_Even(self._esp._L, the_k)
            else:
                psi_to_append = X_Space_Proj_Neg_Even(self._esp._L, np.imag(the_k))
        else:
            if np.imag(the_k)==0:
                psi_to_append = X_Space_Proj_Pos_Odd(self._esp._L, the_k)
            else:
                psi_to_append = X_Space_Proj_Neg_Odd(self._esp._L, np.imag(the_k))

        return psi_to_append

    def compute_k_space_proj_component(self, the_state: int) -> Function_of_array:
        index = self._esp._energy_states.index(the_state)
        the_k = self._esp._k_kappa_l[index]

        if the_state%2 == 0:
            if np.imag(the_k) == 0:
                phi_to_append = K_Space_Proj_Pos_Even(self._esp._L, the_k)
            else:
                phi_to_append = K_Space_Proj_Neg_Even(self._esp._L, np.imag(the_k))
        else:
            if np.imag(the_k) == 0:
                phi_to_append = K_Space_Proj_Pos_Odd(self._esp._L, the_k)
            else:
                phi_to_append = K_Space_Proj_Neg_Odd(self._esp._L, np.imag(the_k))

        return phi_to_append
    
    def add_x_space_proj_component(self, the_state: int):
        self._xsp._x_space_single_energy_proj.append(self.compute_k_space_proj_component(the_state))

    def add_k_space_proj_component(self, the_state: int, continuous: bool):
        phi_to_append = self.compute_k_space_proj_component(the_state)
        if continuous == True:
            self._ksp._cont_k_space_single_energy_proj.append(phi_to_append)
        else:
            self._new_ksp._new_k_space_single_energy_proj.append(self._conversion_factor_k_to_new_k*phi_to_append)
    
    def x_space_func_recombine(self):
        self._xsp._x_space_wavefunction = None_Function()
        for state_index in range(self._esp._num_energy_states):
            self._xsp._x_space_wavefunction += self._esp._energy_proj_coeffs[state_index]*self._xsp._x_space_single_energy_proj[state_index]*self._esp._wiggle_factors[state_index]
            
    def k_space_func_recombine(self, continuous: bool):
        if continuous==True:
            self._ksp._cont_k_space_wavefunction = None_Function()
            for state_index in range(self._esp._num_energy_states):
                self._ksp._cont_k_space_wavefunction += self._esp._energy_proj_coeffs[state_index]*self._ksp._cont_k_space_single_energy_proj[state_index]*self._esp._wiggle_factors[state_index]
        else:
            self._new_ksp._new_k_space_wavefunction = None_Function()
            for state_index in range(self._esp._num_energy_states):
                self._new_ksp._new_k_space_wavefunction += self._esp._energy_proj_coeffs[state_index]*self._new_ksp._new_k_space_single_energy_proj[state_index]*self._esp._wiggle_factors[state_index]
    
    def generate_x_and_k_projections(self):
        print("start proj generation...")
        print("energy states: ", self._esp._energy_states)
        for state in self._esp._energy_states:
            self.add_x_space_proj_component(state)
            self.add_k_space_proj_component(state, True)
            self.add_k_space_proj_component(state, False)
        
        self.x_space_func_recombine()
        self.k_space_func_recombine(True)
        self.k_space_func_recombine(False)

    def add_energy_states(self, the_states, their_coeffs):
        if isinstance(the_states, int):
            the_states = [the_states]
            print("single state converted to list: ", the_states)
            their_coeffs = np.array([their_coeffs])

        print("adding state(s): ", the_states)

        self._esp._energy_proj_coeffs = np.append(self._esp._energy_proj_coeffs, their_coeffs)
        self._esp._num_energy_states += len(the_states)
        self._esp.normalize()

        for state in the_states:
            
            self._esp._energy_states.append(state)
            k_kappa_to_append = gamma_to_k(self._esp._gamma, state, self._esp._L)[0]
            self._esp._k_kappa_l.append(k_kappa_to_append)

            energy_to_append = np.real(k_kappa_to_append**2)/(2*self._esp._m)
            self._esp._energies.append(energy_to_append)
            
            self._esp._wiggle_factors.append(Wiggle_Factor(energy_to_append))

            self.add_k_space_proj_component(state, True)
            self.add_k_space_proj_component(state, False)
            self.add_x_space_proj_component(state)
            

        print("current config: ",self._esp._energy_states)
        
        self.x_space_func_recombine()
        self.k_space_func_recombine(True)
        self.k_space_func_recombine(False)

    def remove_energy_states(self, the_states):
        if isinstance(the_states, int):
            print("single state converted to list: ", the_states)
            the_states = [the_states]
        
        print("removing state(s): ", the_states)
        self._esp._num_energy_states -= len(the_states)


        for state in the_states:
            index = self._esp._energy_states.index(state)
            self._esp._energy_proj_coeffs = np.delete(self._esp._energy_proj_coeffs, index)
            self._esp._k_kappa_l.pop(index)
            self._esp._energies.pop(index)
            self._esp._wiggle_factors.pop(index)
            self._xsp._x_space_single_energy_proj.pop(index)
            self._ksp._cont_k_space_single_energy_proj.pop(index)
            self._new_ksp._new_k_space_single_energy_proj.pop(index)

            # This absolutely needs to be the last action of this iteration!
            self._esp._energy_states.remove(state)

        print("current config: ", self._esp._energy_states)
        
        self._esp.normalize()
        self.x_space_func_recombine()
        self.k_space_func_recombine(True)
        self.k_space_func_recombine(False)

    def full_recompute(self):
        for l in range(self._esp._num_energy_states):
            state = self._esp._energy_states[l]
            self._xsp._x_space_single_energy_proj[l] = self.compute_x_space_proj_component(state)

            k_components = self.compute_k_space_proj_component(state)
            self._ksp._cont_k_space_single_energy_proj[l] = k_components
            self._new_ksp._new_k_space_single_energy_proj[l] = self._conversion_factor_k_to_new_k*k_components

        self.x_space_func_recombine()
        self.k_space_func_recombine(True)
        self.k_space_func_recombine(False)


class Particle_in_Box_State_v2:
    def __init__(self, energy_space_projection: Energy_Space_Projection) -> None:
        self._esp = energy_space_projection
        self._xsp = Position_Space_Projection(None_Function(), [])
        self._ksp = Momentum_Space_Projection(None_Function(), [])
        self._new_ksp = New_Momentum_Space_Projection(None_Function(), [])

        self._proj_handler = Projection_Handler(energy_space_projection, self._xsp, self._ksp, self._new_ksp)

    @staticmethod
    def init_from_scratch(gamma: float, L: float, m: float, energy_states: list, energy_proj_coeffs: np.ndarray):
        esp = Energy_Space_Projection(L, gamma, m, energy_states, energy_proj_coeffs)
        return Particle_in_Box_State_v2(esp)


    @property
    def energy_space_wavefunction(self) -> Energy_Space_Projection:
        return self._esp

    @property
    def x_space_wavefunction(self) -> Position_Space_Projection:
        return self._xsp._x_space_wavefunction

    @property
    def k_space_wavefunction(self) -> Momentum_Space_Projection:
        return self._ksp._cont_k_space_wavefunction

    @property
    def new_k_space_wavefunction(self) -> New_Momentum_Space_Projection:
        return self._new_ksp._new_k_space_wavefunction


class Particle_in_Box_State:
    """
    This class will soon be discarded from this branch
    """
    _L = np.pi
    _gamma = 0
    _m = 1

    _energy_states = None
    _energy_proj_coeff = None
    _energy_state_energies = None
    _k_kappa_l_array = None
    _num_energy_states = 0
    _wiggle_factors = None

    _x_space_wavefunc_components = None
    _x_space_wavefunc = None

    _disc_k_space_wavefunc_components = None
    _cont_k_space_wavefunc_components = None

    _disc_k_space_wavefunc = None
    _cont_k_space_wavefunc = None

    def add_x_space_proj_component(self, the_state: int):
        index = self._energy_states.index(the_state)
        the_k = self._k_kappa_l_array[index]
        
        if the_state%2 == 0:
            if np.imag(the_k)==0:
                psi_to_append = X_Space_Proj_Pos_Even(self._L, the_k)
            else:
                psi_to_append = X_Space_Proj_Neg_Even(self._L, np.imag(the_k))
        else:
            if np.imag(the_k)==0:
                psi_to_append = X_Space_Proj_Pos_Odd(self._L, the_k)
            else:
                psi_to_append = X_Space_Proj_Neg_Odd(self._L, np.imag(the_k))

        self._x_space_wavefunc_components.append(psi_to_append)

    def add_k_space_proj_component(self, the_state: int, continuous: bool):
        index = self._energy_states.index(the_state)
        the_k = self._k_kappa_l_array[index]

        if the_state%2 == 0:
            if np.imag(the_k) == 0:
                phi_to_append = K_Space_Proj_Pos_Even(self._L, the_k)
            else:
                phi_to_append = K_Space_Proj_Neg_Even(self._L, np.imag(the_k))
        else:
            if np.imag(the_k) == 0:
                phi_to_append = K_Space_Proj_Pos_Odd(self._L, the_k)
            else:
                phi_to_append = K_Space_Proj_Neg_Odd(self._L, np.imag(the_k))

        if continuous == True:
            self._cont_k_space_wavefunc_components.append(phi_to_append)
        else:
            self._disc_k_space_wavefunc_components.append(np.sqrt(np.pi/self._L)*phi_to_append)
        
    def property_change_complete_recompute(self):
        current_state_config = deepcopy(self._energy_states)
        current_amplitude_config = self._energy_proj_coeff
        print("recomputing all momentum projection coefficients...")

        self.remove_state(current_state_config)
        self.add_state(current_state_config, current_amplitude_config)

        current_state_config = None

    def add_state(self, the_states: list, the_energy_proj_coeffs: np.ndarray):
        if isinstance(the_states, int):
            the_states = [the_states]
            print("single state converted to list: ", the_states)
            the_energy_proj_coeffs = np.array([the_energy_proj_coeffs])

        print("adding state(s): ", the_states)

        self._energy_proj_coeff = np.append(self._energy_proj_coeff, the_energy_proj_coeffs)
        self._num_energy_states += len(the_states)
        self.normalize()

        for state in the_states:
            
            self._energy_states.append(state)
            k_kappa_to_append = gamma_to_k(self._gamma, state, self._L)[0]
            self._k_kappa_l_array.append(k_kappa_to_append)

            energy_to_append = np.real(k_kappa_to_append**2)/(2*self._m)
            self._energy_state_energies.append(energy_to_append)
            
            self._wiggle_factors.append(Wiggle_Factor(energy_to_append))

            self.add_k_space_proj_component(state, True)
            self.add_k_space_proj_component(state, False)
            self.add_x_space_proj_component(state)
            

        print("current config: ",self._energy_states)
        
        self.x_space_func_recombine()
        self.k_space_func_recombine(True)
        self.k_space_func_recombine(False)
        
    def remove_state(self, the_states: list):
        if isinstance(the_states, int):
            print("single state converted to list: ", the_states)
            the_states = [the_states]
        
        print("removing state(s): ", the_states)
        self._num_energy_states -= len(the_states)


        for state in the_states:
            index = self._energy_states.index(state)
            self._energy_proj_coeff = np.delete(self._energy_proj_coeff, index)
            self._k_kappa_l_array.pop(index)
            self._energy_state_energies.pop(index)
            self._wiggle_factors.pop(index)
            self._x_space_wavefunc_components.pop(index)
            self._cont_k_space_wavefunc_components.pop(index)
            self._disc_k_space_wavefunc_components.pop(index)

            # This absolutely needs to be the last action of this iteration!
            self._energy_states.remove(state)

        print("current config: ", self._energy_states)
        

        self.normalize()
        self.x_space_func_recombine()
        self.k_space_func_recombine(True)
        self.k_space_func_recombine(False)

    def normalize(self):
        if self._num_energy_states == 0:
            return 0
        else:
            Total = np.sum(np.power(np.abs(self._energy_proj_coeff), 2))
            self._energy_proj_coeff = self._energy_proj_coeff*(1/np.sqrt(Total))
        
    def __init__(self, gamma, L, energy_states, amplitudes):
        self._gamma = gamma
        self._L = L

        self._energy_states = []
        self._energy_state_energies = []
        self._k_kappa_l_array = []
        self._energy_proj_coeff = np.empty(0)
        self._wiggle_factors = []

        self._x_space_wavefunc = Function_of_array_and_t(lambda x,t: 0)
        self._x_space_wavefunc_components = []


        self._cont_k_space_wavefunc = Function_of_array_and_t(lambda k,t: 0)
        self._disc_k_space_wavefunc = Function_of_array_and_t(lambda k,t: 0)
        self._cont_k_space_wavefunc_components = []
        self._disc_k_space_wavefunc_components = []

        self.add_state(energy_states, amplitudes)

    def x_space_func_recombine(self):
        self._x_space_wavefunc = None_Function()
        for state_index in range(self._num_energy_states):
            self._x_space_wavefunc += self._energy_proj_coeff[state_index]*self._x_space_wavefunc_components[state_index]*self._wiggle_factors[state_index]
            
    def k_space_func_recombine(self, continuous: bool):
        if continuous==True:
            self._cont_k_space_wavefunc = None_Function()
            for state_index in range(self._num_energy_states):
                self._cont_k_space_wavefunc += self._energy_proj_coeff[state_index]*self._cont_k_space_wavefunc_components[state_index]*self._wiggle_factors[state_index]
        else:
            self._disc_k_space_wavefunc = None_Function()
            for state_index in range(self._num_energy_states):
                self._disc_k_space_wavefunc += self._energy_proj_coeff[state_index]*self._disc_k_space_wavefunc_components[state_index]*self._wiggle_factors[state_index]
    
    def change_energy_proj_coeff(self, the_state, the_coeff):
        self._energy_proj_coeff[self._energy_states.index(the_state)] = the_coeff
        self.normalize()

        self.x_space_func_recombine()
        self.k_space_func_recombine(True)
        self.k_space_func_recombine(False)

    @property
    def L(self):
        """Remember to call [self.property_change_complete_recompute] after calling the setter"""
        return self._L

    @L.setter
    def L(self, new_L):
        self._L = new_L

    @property
    def gamma(self):
        """Remember to call [self.property_change_complete_recompute] after calling the setter"""
        return self._gamma

    @gamma.setter
    def gamma(self, new_gamma):
        self._gamma = new_gamma

    @property
    def x_space_wavefunc(self):
        return self._x_space_wavefunc

    @property
    def disc_k_space_wavefunc(self):
        return self._disc_k_space_wavefunc
    
    @property
    def cont_k_space_wavefunc(self):
        return self._cont_k_space_wavefunc

    

