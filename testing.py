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


class Function_of_x_and_t(object):
    def __init__(self, function: Callable[[np.ndarray, float], float]):
        self._function = function

    def __call__(self, x: np.ndarray, t: float):
        return self._function(x, t)
        
    def __add__(self, other):
        return Function_of_x_and_t(lambda x,t: self._function(x,t) + other._function(x,t))

    def __mul__(self, other):
        if isinstance(other, Function_of_x_and_t):
            return Function_of_x_and_t(lambda x, t: self._function(x,t)*other._function(x,t))
        if isinstance(other, (complex, float, int)):
            return Function_of_x_and_t(lambda x, t: other*self._function(x,t))
    
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
    _bound = 0

    _energy_states = None
    _energy_proj_coeff = None
    _energy_state_energies = None

    _pos_space_wavefunc_components = None
    _pos_space_wavefunc = None
    _k_kappa_l_array = None
    _momentum_kn = None
    _momentum_k = None

    _momentum_proj_coeff_matrix_cont = None
    _momentum_proj_coeff_matrix_disc = None

    _momentum_proj_coeff_cont = None
    _momentum_proj_coeff_disc = None
    _momentum_prob_distr_cont = None
    _momentum_prob_distr_disc = None
    _num_energy_states = 0
    _num_momentum_states_cont = 0

    _cStep = 0.01
    _dStep = 1

    _m = 1

    @property
    def pos_space_wavefunc(self):
        return self._pos_space_wavefunc

    def add_pos_space_func_component(self, the_state: int):
        index = self._energy_states.index(the_state)
        the_k = self._k_kappa_l_array[index]
        the_energy = self._energy_state_energies[index]
        the_coeff = self._energy_proj_coeff[index]
        
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

        psi_to_append = Function_of_x_and_t(psi_to_append)
        wiggle_factor = Function_of_x_and_t(lambda x,t: np.exp(-1j*the_energy*t))
        final_append = the_coeff*psi_to_append*wiggle_factor

        self._pos_space_wavefunc_components.append(final_append)
        self._pos_space_wavefunc += final_append


    def property_change_complete_recompute(self):
        current_state_config = deepcopy(self._energy_states)
        current_amplitude_config = self._energy_proj_coeff
        print("recomputing all momentum projection coefficients...")

        self.remove_state(current_state_config)
        self.add_state(current_state_config, current_amplitude_config)

        current_state_config = None

    # Newly implemented
    def momentum_proj_coeff_matrix_add_row(self, the_state: int, continuous: bool):
        if continuous == True:
            momentum_k = self._momentum_k
            coefficients = self._momentum_proj_coeff_matrix_cont
        else:
            momentum_k = self._momentum_kn
            coefficients = self._momentum_proj_coeff_matrix_disc
        
        the_k = self._k_kappa_l_array[self._energy_states.index(the_state)]

        if the_state%2 == 0:
            if np.imag(the_k) == 0:
                coefficients.append(momentum_Proj_Pos_even(self._L, the_k, momentum_k))
            else:
                coefficients.append(momentum_Proj_Neg_even(self._L, np.imag(the_k), momentum_k))
        else:
            if np.imag(the_k) == 0:
                coefficients.append(momentum_Proj_Pos_odd(self._L, the_k, momentum_k))
            else:
                coefficients.append(momentum_Proj_Neg_odd(self._L, np.imag(the_k), momentum_k))

    def update_momentum_proj_coeffs(self, continuous: bool):
        if continuous == True:
            momentum_k = self._momentum_k
            coefficients = self._momentum_proj_coeff_matrix_cont
            self._num_momentum_states_cont = np.size(self._momentum_k)
        else:
            momentum_k = self._momentum_kn
            coefficients = self._momentum_proj_coeff_matrix_disc
        
        coeff_k_psi = []

        for k in range(np.size(momentum_k)):
            coeff = 0
            for i in range(self._num_energy_states):
                coeff += self._energy_proj_coeff[i]*coefficients[i][k]
            coeff_k_psi.append(coeff)

        if continuous==True:
            self._momentum_proj_coeff_cont = np.array(coeff_k_psi)
        else:
            self._momentum_proj_coeff_disc = np.sqrt(np.pi/self._L)*np.array(coeff_k_psi)

    # Newly implemented
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
            self.momentum_proj_coeff_matrix_add_row(state, True)
            self.momentum_proj_coeff_matrix_add_row(state, False)
            self.add_pos_space_func_component(state)
        
        print("current config: ",self._energy_states)

        self.update_momentum_proj_coeffs(True)
        self.update_momentum_proj_coeffs(False)
        self._momentum_prob_distr_cont = np.power(np.abs(self._momentum_proj_coeff_cont), 2)
        self._momentum_prob_distr_disc = np.power(np.abs(self._momentum_proj_coeff_disc), 2)
        
    # Newly implemented
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
            self._momentum_proj_coeff_matrix_cont.pop(index)
            self._momentum_proj_coeff_matrix_disc.pop(index)
            self._pos_space_wavefunc_components.pop(index)
            # This absolutely needs to be the last action of this iteration!
            self._energy_states.remove(state)

        print("current config: ", self._energy_states)
        

        self.normalize()
        self.update_momentum_proj_coeffs(True)
        self.update_momentum_proj_coeffs(False)
        self._momentum_prob_distr_cont = np.power(np.abs(self._momentum_proj_coeff_cont), 2)
        self._momentum_prob_distr_disc = np.power(np.abs(self._momentum_proj_coeff_disc), 2)

        self._pos_space_wavefunc = Function_of_x_and_t(lambda x,t: 0)
        for func in self._pos_space_wavefunc_components:
            self._pos_space_wavefunc += func

    def normalize(self):
        if self._num_energy_states == 0:
            return 0
        else:
            Total = np.sum(np.power(np.abs(self._energy_proj_coeff), 2))
            self._energy_proj_coeff = self._energy_proj_coeff*(1/np.sqrt(Total))
        
    def __init__(self, gamma, L, energy_states, amplitudes, momentum_k_max, momentum_k_min):
        self._gamma = gamma
        self._L = L
        self._momentum_k = np.arange(momentum_k_min, momentum_k_max, self._cStep)*np.pi/L
        self._momentum_kn = np.arange(momentum_k_min, momentum_k_max, self._dStep)*np.pi/L

        self._energy_states = []
        self._energy_state_energies = []
        self._k_kappa_l_array = []
        self._momentum_proj_coeff_matrix_cont = []
        self._momentum_proj_coeff_matrix_disc = []
        self._energy_proj_coeff = np.empty(0)
        self._pos_space_wavefunc = Function_of_x_and_t(lambda x,t: 0)
        self._pos_space_wavefunc_components = []

        self.add_state(energy_states, amplitudes)

    
    def change_energy_proj_coeff(self, the_state, the_coeff):
        self._energy_proj_coeff[self._energy_states.index(the_state)] = the_coeff
        self.normalize()
        self.update_momentum_proj_coeffs(True)
        self.update_momentum_proj_coeffs(False)

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
    def bound(self):
        """Remember to call [self.property_change_complete_recompute] after calling the setter"""
        return self._bound
    
    @bound.setter
    def bound(self, new_bound):
        new_bound = int(new_bound)
        self._bound = new_bound
        self._momentum_k = np.arange(-new_bound, new_bound, self._cStep)
        self._momentum_kn = np.arange(-new_bound, new_bound, self._dStep)

    @property
    def cStep(self):
        """Remember to call [self.property_change_complete_recompute] after calling the setter"""
        return self._cStep
    
    @cStep.setter
    def cStep(self, new_cStep):
        self._cStep = new_cStep
        self._momentum_k = np.arange(-self._bound, self._bound, new_cStep)

    @property
    def momentum_k(self):
        """Remember to call [self.property_change_complete_recompute] after calling the setter"""
        return self._momentum_k

    @property
    def momentum_kn(self):
        """Remember to call [self.property_change_complete_recompute] after calling the setter"""
        return self._momentum_kn

    
class State_Plot:
    def __init__(self, state: Particle_in_Box_State):
        self._state = state
