from __future__ import annotations

from .Backend import *
from . import particle_in_a_box as pib


class Momentum_Space_Gaussian:
    def __init__(self, a: float, k_0: float) -> None:
        self._a = a
        self._k_0 = k_0
    
    def __call__(self, k: float | np.ndarray) -> float:
        a = self._a
        k_0 = self._k_0
        return np.sqrt(2*a*np.sqrt(np.pi))*np.exp(-a**2/2*(k-k_0)**2)

class Bouncing_Gaussian(pib.Particle_in_Box_Immediate_Mode):
    def find_amplitudes(self) -> None:
        L = self.L
        amplitudes_to_add = []
        l_range = range(self._l_0-self._l_range, self._l_0+self._l_range+1)

        if self.case == "dirichlet":
            for l in l_range:
                pos_k_append = self._k_space_gaussian(np.pi/L*(l+1))
                neg_k_append = self._k_space_gaussian(-np.pi/L*(l+1))
                if l%2 == 0:
                    amplitudes_to_add.append(pos_k_append+neg_k_append)
                else:
                    amplitudes_to_add.append(1j*(pos_k_append-neg_k_append))
        
        elif self.case == "neumann":
            for l in l_range:
                pos_k_append = self._k_space_gaussian(np.pi/L*l)
                neg_k_append = self._k_space_gaussian(-np.pi/L*l)
                if l%2 == 0:
                    amplitudes_to_add.append(pos_k_append+neg_k_append)
                else:
                    amplitudes_to_add.append(1j*(pos_k_append-neg_k_append))
            
            if 0 in l_range:
                i = l_range.index(0)
                amplitudes_to_add[i] = np.sqrt(2)*self._k_space_gaussian(0)

        elif self.case == "dirichlet_neumann":
            for l in l_range:
                pos_k_append = self._k_space_gaussian(np.pi/L*(l+0.5))*np.exp(-1j*(2*l+1)*np.pi/4)
                neg_k_append = self._k_space_gaussian(-np.pi/L*(l+0.5))*np.exp(1j*(2*l+1)*np.pi/4)
                amplitudes_to_add.append(1j*(pos_k_append-neg_k_append))

        states_to_add = list(l_range)
        self.add_state(states_to_add, amplitudes_to_add)

    @staticmethod
    def estimate_l_0(case, L, k_0) -> int:
        """THIS FUNCTION IS EXPERIMENTAL AND WILL BE LATER INCLUDED IN BOUNDARY
        CLASSES
        """
        if case == "neumann":
            return int(L/np.pi*k_0)
        elif case == "dirichlet_neumann":
            return int(L/np.pi*k_0 - 0.5)
        elif case == "dirichlet":
            return int(L/np.pi*k_0 - 1)

    def __init__(self, case: str, L: float, m: float, l_0: int, l_range: int, a: float) -> None:
        super().__init__(case, L, m)
        
        self._l_0 = l_0
        self._k_0 = self._sp.boundary_lib.get_kl(l_0)
        
        self._l_range = l_range
        self._k_space_gaussian = Momentum_Space_Gaussian(a, self._k_0)
        self.find_amplitudes()

    @classmethod
    def init_with_k_0(cls, case: str, L: float, m: float, k_0: int, l_range: int, a: float) -> Bouncing_Gaussian:
        """THIS IS NOT AN ELEGANT SOLUTION!
        """
        l_0 = cls.estimate_l_0(case, L, k_0)
        Gaussian = cls(case, L, m, l_0, l_range, a)
        Gaussian._l_0 = l_0
        Gaussian._k_0 = k_0
        Gaussian._k_space_gaussian = Momentum_Space_Gaussian(a, k_0)
        Gaussian.find_amplitudes()
        return Gaussian
        