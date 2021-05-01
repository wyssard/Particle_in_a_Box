from typing import Callable
import numpy as np
from copy import deepcopy
from scipy.optimize import fsolve

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


class Function_of_array_and_t(object):
    def __init__(self, function: Callable[[np.ndarray, float], float]):
        self._function = function

    def __call__(self, x: np.ndarray, t: float):
        return self._function(x, t)
        
    def __add__(self, other):
        return Function_of_array_and_t(lambda x,t: self._function(x,t) + other._function(x,t))

    def __mul__(self, other):
        if isinstance(other, Function_of_array_and_t):
            return Function_of_array_and_t(lambda x, t: self._function(x,t)*other._function(x,t))
        if isinstance(other, (complex, float, int)):
            return Function_of_array_and_t(lambda x, t: other*self._function(x,t))
    
    def __rmul__(self, other):
        return self.__mul__(other)



psi_l_Pos_odd = lambda L, kl, x: np.sqrt(2/L)*np.power(1+np.sin(kl*L)/(kl*L), -1/2)*np.cos(kl*x)
psi_l_Pos_even = lambda L, kl, x: np.sqrt(2/L)*np.power(1-np.sin(kl*L)/(kl*L), -1/2)*np.sin(kl*x)
psi_l_Neg_odd = lambda L, kappal, x: np.sqrt(2/L)*np.power(1+np.sinh(kappal*L)/(kappal*L), -1/2)*np.cosh(kappal*x)
psi_l_Neg_even = lambda L, kappal, x: np.sqrt(2/L)*np.power(-1+np.sinh(kappal*L)/(kappal*L), -1/2)*np.sinh(kappal*x)

momentum_Proj_Pos_even = lambda L, kl, k: np.sqrt(L/np.pi)/np.sqrt(1-np.sin(kl*L)/(kl*L))*(np.sin((kl+k)*L/2)/(kl*L+k*L) - np.sin((kl-k)*L/2)/(kl*L-k*L))
momentum_Proj_Pos_odd = lambda L, kl, k: np.sqrt(L/np.pi)/np.sqrt(1+np.sin(kl*L)/(kl*L))*(np.sin((kl+k)*L/2)/(kl*L+k*L) + np.sin((kl-k)*L/2)/(kl*L-k*L))
momentum_Proj_Neg_even = lambda L, kappal, k: (2j)*np.sqrt(L/np.pi)/np.sqrt(-1+np.sinh(kappal*L)/(kappal*L))*(k*L*np.cos(k*L/2)*np.sinh(kappal*L/2) - kappal*L*np.sin(k*L/2)*np.cosh(kappal*L/2))/((kappal*L)**2+(k*L)**2)
momentum_Proj_Neg_odd = lambda L, kappal, k: (2)*np.sqrt(L/np.pi)/np.sqrt(1+np.sinh(kappal*L)/(kappal*L))*(k*L*np.cos(k*L/2)*np.sinh(kappal*L/2) + kappal*L*np.sin(k*L/2)*np.cosh(kappal*L/2))/((kappal*L)**2+(k*L)**2)


class Particle_in_Box_State:
    _L = np.pi
    _gamma = 0

    _energy_states = None
    _energy_proj_coeff = None
    _energy_state_energies = None

    _pos_space_wavefunc_components = None
    _pos_space_wavefunc = None

    _disc_momentum_space_wavefunc_components = None
    _cont_momentum_space_wavefunc_components = None

    _disc_momentum_space_wavefunc = None
    _cont_momentum_space_wavefunc = None

    _k_kappa_l_array = None
    _momentum_kn = None
    _momentum_k = None

    _num_energy_states = 0

    _m = 1

    @property
    def pos_space_wavefunc(self):
        return self._pos_space_wavefunc

    @property
    def disc_momentum_space_wavefunc(self):
        return self._disc_momentum_space_wavefunc
    
    @property
    def cont_momentum_space_wavefunc(self):
        return self._cont_momentum_space_wavefunc

    def add_pos_space_func_component(self, the_state: int):
        index = self._energy_states.index(the_state)
        the_k = self._k_kappa_l_array[index]
        
        if the_state%2 == 0:
            if np.imag(the_k)==0:
                psi_to_append = lambda x, t: psi_l_Pos_even(self._L, the_k, x)
            else:
                psi_to_append = lambda x, t: psi_l_Neg_even(self._L, np.imag(the_k), x)
        else:
            if np.imag(the_k)==0:
                psi_to_append = lambda x, t: psi_l_Pos_odd(self._L, the_k, x)
            else:
                psi_to_append = lambda x, t: psi_l_Neg_odd(self._L, np.imag(the_k), x)

        psi_to_append = Function_of_array_and_t(psi_to_append)

        self._pos_space_wavefunc_components.append(psi_to_append)

    def add_momentum_space_func_component(self, the_state: int, continuous: bool):
        index = self._energy_states.index(the_state)
        the_k = self._k_kappa_l_array[index]

        if the_state%2 == 0:
            if np.imag(the_k) == 0:
                phi_to_append = lambda k, t: momentum_Proj_Pos_even(self._L, the_k, k)
            else:
                phi_to_append = lambda k, t: momentum_Proj_Neg_even(self._L, np.imag(the_k),k)
        else:
            if np.imag(the_k) == 0:
                phi_to_append = lambda k, t: momentum_Proj_Pos_odd(self._L, the_k, k)
            else:
                phi_to_append = lambda k, t: momentum_Proj_Neg_odd(self._L, np.imag(the_k), k)

        phi_to_append = Function_of_array_and_t(phi_to_append)

        if continuous == True:
            self._cont_momentum_space_wavefunc_components.append(phi_to_append)
        else:
            self._disc_momentum_space_wavefunc_components.append(np.sqrt(np.pi/self._L)*phi_to_append)
        
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
            self._energy_state_energies.append(np.real(k_kappa_to_append**2)/(2*self._m))
            self.add_momentum_space_func_component(state, True)
            self.add_momentum_space_func_component(state, False)
            self.add_pos_space_func_component(state)
        
        print("current config: ",self._energy_states)

        self.pos_space_func_recombine()
        self.momentum_space_func_recombine(True)
        self.momentum_space_func_recombine(False)
    
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

            self._pos_space_wavefunc_components.pop(index)
            self._cont_momentum_space_wavefunc_components.pop(index)
            self._disc_momentum_space_wavefunc_components.pop(index)

            # This absolutely needs to be the last action of this iteration!
            self._energy_states.remove(state)

        print("current config: ", self._energy_states)
        

        self.normalize()
        self.pos_space_func_recombine()
        self.momentum_space_func_recombine(True)
        self.momentum_space_func_recombine(False)

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

        self._pos_space_wavefunc = Function_of_array_and_t(lambda x,t: 0)
        self._pos_space_wavefunc_components = []

        self._cont_momentum_space_wavefunc = Function_of_array_and_t(lambda k,t: 0)
        self._disc_momentum_space_wavefunc = Function_of_array_and_t(lambda k,t: 0)
        self._cont_momentum_space_wavefunc_components = []
        self._disc_momentum_space_wavefunc_components = []

        self.add_state(energy_states, amplitudes)

    def pos_space_func_recombine(self):
        self._pos_space_wavefunc = Function_of_array_and_t(lambda x,t:0)
        for state_index in range(self._num_energy_states):
            wiggleFactor = Function_of_array_and_t(lambda x,t: np.exp(-1j*self._energy_state_energies[state_index]*t))
            self._pos_space_wavefunc += self._energy_proj_coeff[state_index]*self._pos_space_wavefunc_components[state_index]*wiggleFactor
            
    def momentum_space_func_recombine(self, continuous: bool):
        if continuous==True:
            self._cont_momentum_space_wavefunc = Function_of_array_and_t(lambda x,t:0)
            for state_index in range(self._num_energy_states):
                wiggleFactor = Function_of_array_and_t(lambda x,t: np.exp(-1j*self._energy_state_energies[state_index]*t))
                self._cont_momentum_space_wavefunc += self._energy_proj_coeff[state_index]*self._cont_momentum_space_wavefunc_components[state_index]*wiggleFactor
        else:
            self._disc_momentum_space_wavefunc = Function_of_array_and_t(lambda x,t:0)
            for state_index in range(self._num_energy_states):
                wiggleFactor = Function_of_array_and_t(lambda x,t: np.exp(-1j*self._energy_state_energies[state_index]*t))
                self._disc_momentum_space_wavefunc += self._energy_proj_coeff[state_index]*self._disc_momentum_space_wavefunc_components[state_index]*wiggleFactor
    
    def change_energy_proj_coeff(self, the_state, the_coeff):
        self._energy_proj_coeff[self._energy_states.index(the_state)] = the_coeff
        self.normalize()

        self.pos_space_func_recombine()
        self.momentum_space_func_recombine(True)
        self.momentum_space_func_recombine(False)

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

    
class State_Plot:
    def __init__(self, state: Particle_in_Box_State):
        self._state = state
