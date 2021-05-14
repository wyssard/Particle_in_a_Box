from __future__ import annotations
from typing import List
from typing import Callable
import numpy as np
from scipy.optimize import fsolve
from abc import ABC, abstractmethod


posRelEven = lambda g, k: g-np.arctan(k*np.tan(k/2))
posRelOdd = lambda g, k: g+np.arctan(k/(np.tan(k/2)))

negRelEven = lambda g, k: g+np.arctan(k*np.tanh(k/2))
negRelOdd = lambda g, k: g+np.arctan(k/np.tanh(k/2))

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
        Function_of_array.__init__(self, lambda k: 1j*np.sqrt(L/np.pi)/np.sqrt(1 - np.sin(kl*L)/(kl*L))*(np.sin((kl+k)*L/2)/(kl*L+k*L) - np.sin((kl-k)*L/2)/(kl*L-k*L)))

class K_Space_Proj_Pos_Odd(Function_of_array):
    def __init__(self, L, kl):
        Function_of_array.__init__(self, lambda k: np.sqrt(L/np.pi)/np.sqrt(1 + np.sin(kl*L)/(kl*L))*(np.sin((kl+k)*L/2)/(kl*L+k*L) + np.sin((kl-k)*L/2)/(kl*L-k*L)))

class K_Space_Proj_Neg_Even(Function_of_array):
    def __init__(self, L, kappal):
        Function_of_array.__init__(self, lambda k: (2j)*np.sqrt(L/np.pi)/np.sqrt(-1+np.sinh(kappal*L)/(kappal*L))*(k*L*np.cos(k*L/2)*np.sinh(kappal*L/2) - kappal*L*np.sin(k*L/2)*np.cosh(kappal*L/2))/((kappal*L)**2+(k*L)**2))

class K_Space_Proj_Neg_Odd(Function_of_array):
    def __init__(self, L, kappal):
        Function_of_array.__init__(self, lambda k: (2)*np.sqrt(L/np.pi)/np.sqrt(1+np.sinh(kappal*L)/(kappal*L))*(k*L*np.cos(k*L/2)*np.sinh(kappal*L/2) + kappal*L*np.sin(k*L/2)*np.cosh(kappal*L/2))/((kappal*L)**2+(k*L)**2))


class State_Properties:
    def __init__(self, gamma: float, L: float, m: float) -> None:
        self._gamma = gamma
        self._L = L
        self._m = m
        self._num_energy_states = 0 
        self._energy_states = []
        self._k_kappa_l_array = []

    @property
    def gamma(self) -> float:
        return self._gamma

    @gamma.setter
    def gamma(self, newgamma) -> None:
        self._gamma = newgamma

    @property
    def L(self) -> float:
        return self._L

    @L.setter
    def L(self, new_L) -> None:
        self._L = new_L

    @property
    def m(self) -> float:
        return self._m

    @m.setter
    def m(self, new_m) -> None:
        self._m = new_m
    
    @property
    def num_energy_states(self) -> int:
        return self._num_energy_states
    
    @num_energy_states.setter
    def num_energy_states(self, new_num_energy_states) -> None:
        self._num_energy_states = new_num_energy_states

    @property
    def energy_states(self) -> List[int]:
        return self._energy_states

    @property
    def k_kappa_l(self) -> List[complex]:
        return self._k_kappa_l_array

    
class Energy_Space_Projection:
    def __init__(self, energies: list, energy_proj_coeffs: np.ndarray, wiggle_factors: List[Wiggle_Factor], state_properties: State_Properties) -> None:
        self._energies = energies
        self._energy_proj_coeffs = energy_proj_coeffs
        self._wiggle_factors = wiggle_factors
        self._sp = state_properties
        self._Norm = 0
        self._exp_value = 0
        
    def normalize(self) -> None:
        if self._sp.num_energy_states == 0:
            self._Norm = 0
        else:
            self._Norm = np.sqrt(np.sum(np.power(np.abs(self._energy_proj_coeffs), 2)))
            self._energy_proj_coeffs = self._energy_proj_coeffs*(1/self._Norm)

    def change_coeff(self, the_state, the_coeff):
        the_index = self._sp.energy_states.index(the_state)
        self._energy_proj_coeffs *= self._Norm
        self._energy_proj_coeffs[the_index] = the_coeff
        self.normalize()

    def compute_expectation_value(self):
        self._exp_value = 0
        for i in range(self._sp.num_energy_states):
            self._exp_value += (np.abs(self._energy_proj_coeffs[i])**2)*self._energies[i]
        
class New_Momentum_Space_Projection:
    def __init__(self, new_k_space_wavefunction: Function_of_array_and_t, new_k_space_single_energy_proj: list, state_properties: State_Properties) -> None:
        self._new_k_space_wavefunction = new_k_space_wavefunction
        self._new_k_space_single_energy_proj = new_k_space_single_energy_proj
        self._sp = state_properties
    
    def recombine(self, energy_space_proj: Energy_Space_Projection) -> None:
        self._new_k_space_wavefunction = None_Function()
        temp_proj_coeff = energy_space_proj._energy_proj_coeffs
        temp_wigglers = energy_space_proj._wiggle_factors
        for i in range(len(temp_proj_coeff)):
            self._new_k_space_wavefunction += temp_proj_coeff[i]*self._new_k_space_single_energy_proj[i]*temp_wigglers[i]

class Momentum_Space_Projection:
    def __init__(self, cont_k_space_wavefunction: Function_of_array_and_t, cont_k_space_single_energy_proj: list, state_properties: State_Properties) -> None:
        self._cont_k_space_wavefunction = cont_k_space_wavefunction
        self._cont_k_space_single_energy_proj = cont_k_space_single_energy_proj
        self._sp = state_properties
        self._expectation_value = None_Function()
    
    def recombine(self, energy_space_proj: Energy_Space_Projection) -> None:
        self._cont_k_space_wavefunction = None_Function()
        temp_proj_coeff = energy_space_proj._energy_proj_coeffs
        temp_wigglers = energy_space_proj._wiggle_factors
        for i in range(len(temp_proj_coeff)):
            self._cont_k_space_wavefunction += temp_proj_coeff[i]*self._cont_k_space_single_energy_proj[i]*temp_wigglers[i]
    
    def compute_k_space_proj_component(self, the_state: int) -> Function_of_array:
        the_index = self._sp.energy_states.index(the_state)
        the_k = self._sp.k_kappa_l[the_index]

        if the_state%2 == 0:
            if np.imag(the_k) == 0:
                phi_to_append = K_Space_Proj_Pos_Even(self._sp.L, the_k)
            else:
                phi_to_append = K_Space_Proj_Neg_Even(self._sp.L, np.imag(the_k))
        else:
            if np.imag(the_k) == 0:
                phi_to_append = K_Space_Proj_Pos_Odd(self._sp.L, the_k)
            else:
                phi_to_append = K_Space_Proj_Neg_Odd(self._sp.L, np.imag(the_k))

        return phi_to_append
        
    def compute_expectation_value_component(self, left_hand_l: int, right_hand_l: int) -> complex:
        if left_hand_l%2 == right_hand_l%2:
            return 0
        
        lhs_k = self._sp.k_kappa_l[self._sp.energy_states.index(left_hand_l)]
        rhs_k = self._sp.k_kappa_l[self._sp.energy_states.index(right_hand_l)]
        L = self._sp.L

        if left_hand_l%2 == 1:
            norm_expr = np.sqrt((1+np.sin(lhs_k*L)/(lhs_k*L))*(1-np.sin(rhs_k*L)/rhs_k*L))
            sin_expr = np.sin((lhs_k+rhs_k)*L/2)/((lhs_k+rhs_k)*L) + np.sin((lhs_k-rhs_k)*L/2)/((lhs_k-rhs_k)*L)

            return -2j*rhs_k*norm_expr*sin_expr
        
        elif left_hand_l%2 == 0:
            norm_expr = np.sqrt((1-np.sin(lhs_k*L)/(lhs_k*L))*(1+np.sin(rhs_k*L)/rhs_k*L))
            sin_expr = np.sin((lhs_k+rhs_k)*L/2)/((lhs_k+rhs_k)*L) - np.sin((lhs_k-rhs_k)*L/2)/((lhs_k-rhs_k)*L)

            return -2j*rhs_k*norm_expr*sin_expr

    def compute_expectation_value(self, energy_space_proj: Energy_Space_Projection) -> None:
        self._expectation_value = None_Function()
        if self._sp.num_energy_states == 1:
            self._expectation_value = Function_of_t(lambda t: np.zeros(np.shape(t)))
        
        else:
            for rh_index in range(self._sp.num_energy_states):
                rh_coeff = energy_space_proj._energy_proj_coeffs[rh_index]
                rh_state = self._sp.energy_states[rh_index]
                rh_energy = energy_space_proj._energies[rh_index]

                for lh_index in range(self._sp.num_energy_states):
                    lh_coeff = energy_space_proj._energy_proj_coeffs[lh_index]
                    lh_state = self._sp.energy_states[lh_index]
                    lh_energy = energy_space_proj._energies[lh_index]

                    coeff = complex(np.conj(lh_coeff)*rh_coeff)
                    wiggler = Wiggle_Factor(-lh_energy+rh_energy)
                    exp_val_component = self.compute_expectation_value_component(lh_state, rh_state)

                    self._expectation_value += (coeff*wiggler*exp_val_component).get_real_part()

class Position_Space_Projection:
    def __init__(self, x_space_wavefunction: Function_of_array_and_t, x_space_single_energy_proj: list, state_properties: State_Properties) -> None:
        self._x_space_wavefunction = x_space_wavefunction
        self._x_space_single_energy_proj = x_space_single_energy_proj
        self._sp = state_properties
        self._expectation_value = None_Function()
        self._exp_t_deriv = None_Function()

    def recombine(self, energy_space_proj: Energy_Space_Projection) -> None:
        self._x_space_wavefunction = None_Function()
        temp_proj_coeff = energy_space_proj._energy_proj_coeffs
        temp_wigglers = energy_space_proj._wiggle_factors
        for i in range(len(temp_proj_coeff)):
            self._x_space_wavefunction += temp_proj_coeff[i]*self._x_space_single_energy_proj[i]*temp_wigglers[i]
    
    def compute_x_space_proj_component(self, the_state: int) -> Function_of_array:
        the_index = self._sp.energy_states.index(the_state)
        the_k = self._sp.k_kappa_l[the_index]
        
        if the_state%2 == 0:
            if np.imag(the_k)==0:
                psi_to_append = X_Space_Proj_Pos_Even(self._sp.L, the_k)
            else:
                psi_to_append = X_Space_Proj_Neg_Even(self._sp.L, np.imag(the_k))
        else:
            if np.imag(the_k)==0:
                psi_to_append = X_Space_Proj_Pos_Odd(self._sp.L, the_k)
            else:
                psi_to_append = X_Space_Proj_Neg_Odd(self._sp.L, np.imag(the_k))

        return psi_to_append

    def compute_expectation_value_component(self, odd_state: int, even_state: int) -> complex:
        if odd_state%2 == even_state%2:
            return 0
        
        if odd_state%2 == 0:
            temp = odd_state
            odd_state = even_state
            even_state = temp

        lhs_k = self._sp.k_kappa_l[self._sp.energy_states.index(odd_state)]
        rhs_k = self._sp.k_kappa_l[self._sp.energy_states.index(even_state)]
        L = self._sp.L
        cos_expr = np.cos((lhs_k-rhs_k)*L/2)/((lhs_k-rhs_k)*L) - np.cos((lhs_k+rhs_k)*L/2)/((lhs_k+rhs_k)*L)
        sin_expr = 2*(np.sin((lhs_k+rhs_k)*L/2)/(((lhs_k+rhs_k)*L)**2) - np.sin((lhs_k-rhs_k)*L/2)/(((lhs_k-rhs_k)*L)**2))
        norm_expr = np.sqrt((1+np.sin(lhs_k*L)/(lhs_k*L))*(1-np.sin(rhs_k*L)/(rhs_k*L)))
        print("<", odd_state, "| x |", even_state, "> =", (cos_expr + sin_expr)/norm_expr*L)
        return (cos_expr + sin_expr)/norm_expr*L
       
    def compute_expectation_value(self, energy_space_proj: Energy_Space_Projection) -> None:
        self._expectation_value = None_Function()
        self._exp_t_deriv = None_Function()

        if self._sp.num_energy_states == 1:
            self._expectation_value = Function_of_t(lambda t: np.zeros(np.shape(t)))
            self._exp_t_deriv = Function_of_t(lambda t: np.zeros(np.shape(t)))

        else:
            for rh_index in range(1, self._sp.num_energy_states):
                rh_energy = energy_space_proj._energies[rh_index]
                rh_coeff = energy_space_proj._energy_proj_coeffs[rh_index]
                rh_state = self._sp.energy_states[rh_index]

                for lh_index in range(0, rh_index):
                    lh_energy = energy_space_proj._energies[lh_index]
                    lh_coeff = energy_space_proj._energy_proj_coeffs[lh_index]
                    lh_state = self._sp.energy_states[lh_index]

                    wiggler = Wiggle_Factor(rh_energy-lh_energy)
                    exp_val_component = self.compute_expectation_value_component(lh_state, rh_state)    
                    coeff = complex(np.conj(lh_coeff)*rh_coeff)
                    append = coeff*wiggler

                    self._expectation_value += (2*append*exp_val_component).get_real_part()
                    self._exp_t_deriv += (2j*(lh_energy-rh_energy)*append*exp_val_component).get_real_part()

        

class Particle_in_Box_State:
    def __init__(self, gamma: float, L: float, m: float, energy_states: list, amplitudes: np.ndarray) -> None:
        self._sp = State_Properties(gamma, L , m)

        self._conversion_factor_k_to_new_k = np.sqrt(np.pi/self._sp.L)

        self._esp = Energy_Space_Projection([], np.array([]), [], self._sp)
        self._xsp = Position_Space_Projection(None_Function(), [], self._sp)
        self._ksp = Momentum_Space_Projection(None_Function(), [], self._sp)
        self._new_ksp = New_Momentum_Space_Projection(None_Function(), [], self._sp)

        self.add_state(energy_states, amplitudes)

    def change_energy_proj_coeff(self, the_state: int, the_coeff: complex) -> None:
        self._esp.change_coeff(the_state, the_coeff)
        
        self.compute_expectation_values()

        self._ksp.recombine(self._esp)
        self._xsp.recombine(self._esp)
        self._new_ksp.recombine(self._esp)

    def full_projection_recompute(self) -> None:
        print("Recomputing every property that depends on L or gamma...")
        self._conversion_factor_k_to_new_k = np.sqrt(np.pi/self._sp.L)

        for l in range(self._sp.num_energy_states):
            state = self._sp.energy_states[l]
            k_kappa = gamma_to_k(self._sp.gamma, state, self._sp.L)[0]
            self._sp.k_kappa_l[l] = k_kappa

            energy = np.real(k_kappa**2)/(2*self._sp.m)
            self._esp._energies[l] = energy
            self._esp._wiggle_factors[l] = Wiggle_Factor(energy)

            k_proj = self._ksp.compute_k_space_proj_component(state)
            self._xsp._x_space_single_energy_proj[l] = self._xsp.compute_x_space_proj_component(state)
            self._ksp._cont_k_space_single_energy_proj[l] = k_proj
            self._new_ksp._new_k_space_single_energy_proj[l] = self._conversion_factor_k_to_new_k*k_proj

        self.compute_expectation_values()
        self._ksp.recombine(self._esp)
        self._xsp.recombine(self._esp)
        self._new_ksp.recombine(self._esp)

    def add_state(self, the_states: list, the_energy_proj_coeffs: np.ndarray) -> None:
        if isinstance(the_states, int):
            the_states = [the_states]
            print("single state converted to list: ", the_states)
            the_energy_proj_coeffs = np.array([the_energy_proj_coeffs])

        print("adding state(s): ", the_states)

        self._esp._energy_proj_coeffs = np.append(self._esp._energy_proj_coeffs*(self._esp._Norm), the_energy_proj_coeffs)
        self._sp.num_energy_states += len(the_states)
        
        self._esp.normalize()

        for state in the_states:
            self._sp.energy_states.append(state)
            k_kappa_to_append = gamma_to_k(self._sp.gamma, state, self._sp.L)[0]
            self._sp.k_kappa_l.append(k_kappa_to_append)

            energy_to_append = np.real(k_kappa_to_append**2)/(2*self._sp.m)
            self._esp._energies.append(energy_to_append)
            self._esp._wiggle_factors.append(Wiggle_Factor(energy_to_append))

            k_proj_append = self._ksp.compute_k_space_proj_component(state)
            self._xsp._x_space_single_energy_proj.append(self._xsp.compute_x_space_proj_component(state))
            self._ksp._cont_k_space_single_energy_proj.append(k_proj_append)
            self._new_ksp._new_k_space_single_energy_proj.append(self._conversion_factor_k_to_new_k*k_proj_append)
        
        self.compute_expectation_values()
        
        self._ksp.recombine(self._esp)
        self._xsp.recombine(self._esp)
        self._new_ksp.recombine(self._esp)

        print("current config: ", self._sp.energy_states)
        print("energy expectation value: ", self._esp._exp_value)

    def remove_state(self, the_states: list) -> None:
        if isinstance(the_states, int):
            print("single state converted to list: ", the_states)
            the_states = [the_states]
        
        print("removing state(s): ", the_states)
        self._sp.num_energy_states -= len(the_states)

        for state in the_states:
            the_index = self._sp.energy_states.index(state)
            self._esp._energy_proj_coeffs = np.delete(self._esp._energy_proj_coeffs, the_index)
            self._sp.k_kappa_l.pop(the_index)
            self._esp._energies.pop(the_index)
            self._esp._wiggle_factors.pop(the_index)
            self._xsp._x_space_single_energy_proj.pop(the_index)
            self._ksp._cont_k_space_single_energy_proj.pop(the_index)
            self._new_ksp._new_k_space_single_energy_proj.pop(the_index)

            # This absolutely needs to be the last action of this iteration!
            self._sp.energy_states.remove(state)

        self._esp._energy_proj_coeffs *= self._esp._Norm
        self._esp.normalize()
        self.compute_expectation_values()

        self._ksp.recombine(self._esp)
        self._xsp.recombine(self._esp)
        self._new_ksp.recombine(self._esp)

        print("current config: ", self._sp.energy_states)
        print("energy expectation value: ", self._esp._exp_value)

    def compute_expectation_values(self) -> None:
        self._esp.compute_expectation_value()
        self._xsp.compute_expectation_value(self._esp)
        self._ksp.compute_expectation_value(self._esp)

    @property
    def x_space_wavefunction(self) -> Function_of_array_and_t:
        return self._xsp._x_space_wavefunction

    @property
    def k_space_wavefunction(self) -> Function_of_array_and_t:
        return self._ksp._cont_k_space_wavefunction

    @property
    def new_k_space_wavefunction(self) -> Function_of_array_and_t:
        return self._new_ksp._new_k_space_wavefunction
    
    @property
    def L(self) -> float:
        return self._sp.L

    @L.setter
    def L(self, new_L: float) -> None:
        self._sp.L = new_L
        self.full_projection_recompute()

    @property
    def gamma(self) -> float:
        return self._sp.gamma

    @gamma.setter
    def gamma(self, newgamma: float) -> None:
        self._sp.gamma = newgamma
        self.full_projection_recompute()

    @property
    def m(self) -> float:
        return self._sp.m

    @m.setter
    def m(self, new_m: float) -> None:
        self._sp.m = new_m
        for l in range(self._sp.num_energy_states):
            k_kappa = self._sp.k_kappa_l[l]
            energy = np.real(k_kappa**2)/(2*self._sp.m)
            self._esp._energies[l] = energy
            self._esp._wiggle_factors[l] = Wiggle_Factor(energy)
        
        self.compute_expectation_values()
        self._ksp.recombine(self._esp)
        self._xsp.recombine(self._esp)
        self._new_ksp.recombine(self._esp)
        
