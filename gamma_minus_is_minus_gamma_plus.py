from __future__ import annotations

from Backend import *
from scipy.integrate import quad
from scipy.misc import derivative


def x_space_projection_for_nummerics(L, gamma, l, kl) -> Function_of_array:
    i_factor = lambda l: 1 if l%2 == 0 else -1j
        
    if np.imag(kl) == 0:
        boundray_expr = np.exp(1j*kl*L)*(gamma + 1j*kl)/(gamma - 1j*kl)
        norm_expr = np.sqrt(2/L)/np.sqrt(1 - np.real(boundray_expr)*np.sin(kl*L)/(kl*L))
        return Function_of_array(lambda x: norm_expr*1/2*i_factor(l)*(np.exp(1j*kl*x) - boundray_expr*np.exp(-1j*kl*x)))
    
    else:
        kappal = np.imag(kl)
        return Function_of_array(lambda x: np.sqrt(kappal/np.sinh(kappal*L))*np.exp(-gamma*x))


class X_Space_Projector(Energy_State_Projector):
    def __init__(self, L: float, gamma: float, l_to_k_mapper_ref: l_to_kl_mapper):
        Energy_State_Projector.__init__(self, L, gamma, l_to_k_mapper_ref)

    def get_projection(self, l: int) -> Function_of_array:
        gamma = self._gamma
        L = self._L
        kl = self._l_kl_map.get_kl(l)

        return x_space_projection_for_nummerics(L, gamma, l, kl)


class K_Space_Projector(Energy_State_Projector):
    def __init__(self, L: float, gamma: float, l_to_k_mapper_ref: l_to_kl_mapper):
        Energy_State_Projector.__init__(self, L, gamma, l_to_k_mapper_ref)

    def get_projection(self, l: int) -> Function_of_array:
        gamma = self._gamma
        L = self._L
        kl = self._l_kl_map.get_kl(l)

        x_space_proj = x_space_projection_for_nummerics(L, gamma, l, kl)

        def converter(k_range: np.ndarray) -> np.ndarray:
            if isinstance(k_range, (int, float)):
                k_range = [k_range]
            
            out = []
            for k in k_range:
                integrand = lambda x: x_space_proj(x)*np.exp(-1j*k*x)
                real = quad(lambda x: np.real(integrand(x)), -L/2, L/2)[0]
                imag = quad(lambda x: np.imag(integrand(x)), -L/2, L/2)[0]
                out.append((real + 1j*imag)*1/np.sqrt(2*L))
            
            return np.array(out)
        
        return Function_of_array(converter)
        


class Bra_l1_x_Ket_l2(Energy_State_Matrix_Elements):
    def __init__(self, L: float, gamma: float, l_to_k_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, gamma, l_to_k_mapper_ref)

    def get_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        gamma = self._gamma
        L = self._L

        lhs_k = self._l_kl_map.get_kl(lhs_state)
        rhs_k = self._l_kl_map.get_kl(rhs_state)

        lhs_integrand = x_space_projection_for_nummerics(L, gamma, lhs_state, lhs_k)
        rhs_integrand = x_space_projection_for_nummerics(L, gamma, rhs_state, rhs_k)
        integrand = lambda x: np.conj(lhs_integrand(x))*x*rhs_integrand(x)
        real = quad(lambda x: np.real(integrand(x)), -L/2, L/2)[0]
        imag = quad(lambda x: np.imag(integrand(x)), -L/2, L/2)[0]

        return real + 1j*imag


class Bra_l1_pR_Ket_l2(Energy_State_Matrix_Elements):
    def __init__(self, L: float, gamma: float, l_to_k_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, gamma, l_to_k_mapper_ref)

    def get_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        gamma = self._gamma
        L = self._L

        lhs_k = self._l_kl_map.get_kl(lhs_state)
        rhs_k = self._l_kl_map.get_kl(rhs_state)

        lhs_integrand = x_space_projection_for_nummerics(L, gamma, lhs_state, lhs_k)
        rhs_integrand = x_space_projection_for_nummerics(L, gamma, rhs_state, rhs_k)

        integrand = lambda x: (-1j)*np.conj(lhs_integrand(x))*derivative(rhs_integrand, x, 0.0001)

        real = quad(lambda x: np.real(integrand(x)), -L/2, L/2)[0]
        imag = quad(lambda x: np.imag(integrand(x)), -L/2, L/2)[0]


        return real + 1j*imag
        

class Gamma_to_k(Gamma_to_k_Base):
    def __init__(self, L: float, gamma: float | np.ndarray) -> None:
        Gamma_to_k_Base.__init__(self, L)
        self._gamma = gamma

    def __call__(self, l: int) -> complex:
        if l == 0:
            return 1j*self._gamma
        else:
            return l*np.pi/self._L

    def set_gamma(self, gamma: float) -> None:
        self._gamma = gamma


