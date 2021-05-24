from __future__ import annotations

from Backend import *
import particle_in_a_box as pib

# This file would simplify if we implemented new specific Neumann and Dirichlet
# cases instead of considering them as limiting cases of the general 
# symmetric case


class Momentum_Space_Gaussian:
    def __init__(self, a: float, k_0: float) -> None:
        self._a = a
        self._k_0 = k_0
    
    def __call__(self, k: float | np.ndarray) -> float:
        a = self._a
        k_0 = self._k_0
        return np.sqrt(2*a*np.sqrt(np.pi))*np.exp(-a**2/2*(k-k_0)**2)

class Bouncing_Gaussian(pib.Particle_in_Box_State):
    def find_amplitudes(self) -> None:
        state_range = range(self._l_0-self._l_range, self._l_0+self._l_range)
        for l in state_range:
            self._states.append(l)
            k = self._gamma_to_k(l)
            if l%2 == 0:
                amplitude = self._k_space_gaussian(k) + self._k_space_gaussian(-k)
            else:
                amplitude = 1j*(self._k_space_gaussian(k) - self._k_space_gaussian(-k))

            self._amplitudes.append(amplitude)
        
        if (0 in state_range) and self._case == "neumann":
            index = state_range.index(0)
            amplitude[index] = np.sqrt(2)*self._k_space_gaussian(0)


    def __init__(self, case: str, L: float, m: float, l_0: int, l_range: int, a: float) -> None:
        self._case = case    
        self._states = []
        self._amplitudes = []
        self._l_0 = l_0
        self._l_range = l_range
        self._L = L

        self.switch_case()
        self._k_space_gaussian = Momentum_Space_Gaussian(a, self._k_0)
        self._gamma_to_k = pib.symmetric.Gamma_to_k(self._L, self._gamma)
        self._k_0_alt = self._gamma_to_k(l_0)

        print("k_0: ", self._k_0)
        print("k_0_alt: ", self._k_0_alt)

        self.find_amplitudes()
        super().__init__("symmetric", self._gamma, L, m, self._states, self._amplitudes)

    
    def switch_case(self) -> None:
        if self._case == "neumann":
            self._gamma = 10**(-6)
            self._k_0 = self._l_0*np.pi/self._L
        elif self._case == "dirichlet":
            self._gamma = 10**(6)
            self._k_0 = (self._l_0+1)*np.pi/self._L