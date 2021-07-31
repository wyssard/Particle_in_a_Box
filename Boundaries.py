from __future__ import annotations

from Backend import *
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.integrate import quad
from scipy.misc import derivative
import warnings


class Symmetric_Boundary(New_Style_Boundary):
    
    def __init__(self, L: float, gamma: float, theta: float, l_to_kl_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, gamma, theta, l_to_kl_mapper_ref)
        self._pos_energy_even_state_eq = lambda gammaL, kL: gammaL - kL*np.tan(kL/2)
        self._pos_energy_odd_state_eq = lambda gammaL, kL: gammaL + kL/np.tan(kL/2)
        self._neg_energy_even_state_eq = lambda gammaL, kappaL: gammaL + kappaL*np.tanh(kappaL/2)
        self._neg_energy_odd_state_eq = lambda gammaL, kappaL: gammaL + kappaL/np.tanh(kappaL/2)

        self._eps = np.finfo(np.float32).eps

    def set_eps(self, new_eps: float) -> None:
        self._eps = new_eps

    def get_kn(self, n: int | np.ndarray) -> float | np.ndarray:
        return n*np.pi/self._L + self._theta/(2*self._L)

    def get_kl(self, l: int) -> complex:
        gammaL = self._gamma*self._L
        eps = self._eps

        if l == 0:

            if self._gamma > 0:
                transc_eq = self._pos_energy_even_state_eq
                kL_upper_bound = (l+1)*np.pi-eps
                kL_lower_bound = eps

                kL_solution = brentq(lambda Kl: transc_eq(gammaL, Kl), kL_lower_bound, kL_upper_bound)

                return kL_solution/self._L

            else:
                transc_eq = self._neg_energy_even_state_eq
                kL_approx = -gammaL
                kL_solution = fsolve(lambda Kl: transc_eq(gammaL, Kl), kL_approx)[0]
                return 1j*kL_solution/self._L

        elif l == 1:
            if self._gamma > -2/self._L:
                transc_eq = self._pos_energy_odd_state_eq
                kL_upper_bound = (l+1)*np.pi-eps
                kL_lower_bound = eps

                kL_solution = brentq(lambda Kl: transc_eq(gammaL, Kl), kL_lower_bound, kL_upper_bound)

                return kL_solution/self._L
            
            else:
                transc_eq = self._neg_energy_odd_state_eq
                kL_approx = -gammaL
                kL_solution = fsolve(lambda Kl: transc_eq(gammaL, Kl), kL_approx)[0]
                return 1j*kL_solution/self._L

        else:
            if l%2 == 0:
                transc_eq = self._pos_energy_even_state_eq
            else:
                transc_eq = self._pos_energy_odd_state_eq

            kL_upper_bound = (l+1)*np.pi-eps
            kL_lower_bound = (l-1)*np.pi+eps

            kL_solution = brentq(lambda Kl: transc_eq(gammaL, Kl), kL_lower_bound, kL_upper_bound)
            return kL_solution/self._L
        
        
        pass

    def get_x_space_projection(self, l: int) -> Function_of_n:
        L = self._L
        kl = self._l_kl_map.get_kl(l)
        if l%2 == 1:
            if np.imag(kl) == 0:
                return Function_of_n(lambda x: np.sqrt(2/L)*np.power(1-np.sin(kl*L)/(kl*L), -1/2)*np.sin(kl*x))
            else:
                kappal = np.imag(kl)
                return Function_of_n(lambda x: np.sqrt(2/L)*np.power(-1+np.sinh(kappal*L)/(kappal*L), -1/2)*np.sinh(kappal*x))
        else:
            if np.imag(kl) == 0:
                return Function_of_n(lambda x: np.sqrt(2/L)*np.power(1+np.sin(kl*L)/(kl*L), -1/2)*np.cos(kl*x))
            else:
                kappal = np.imag(kl)
                return Function_of_n(lambda x: np.sqrt(2/L)*np.power(1+np.sinh(kappal*L)/(kappal*L), -1/2)*np.cosh(kappal*x))
        
    def get_k_space_projection(self, l: int) -> Function_of_n:
        #print("computing the k_space_projection using analytic results...")
        L = self._L
        kl = self._l_kl_map.get_kl(l)

        if l%2 == 1:
            if np.imag(kl) == 0:
                return Function_of_n(lambda k: 1j*np.sqrt(L/np.pi)/np.sqrt(1 - np.sin(kl*L)/(kl*L))*(np.sin((kl+k)*L/2)/(kl*L+k*L) - np.sin((kl-k)*L/2)/(kl*L-k*L)))
            else:
                kappal = np.imag(kl)
                return Function_of_n(lambda k: (2j)*np.sqrt(L/np.pi)/np.sqrt(-1+np.sinh(kappal*L)/(kappal*L))*(k*L*np.cos(k*L/2)*np.sinh(kappal*L/2) - kappal*L*np.sin(k*L/2)*np.cosh(kappal*L/2))/((kappal*L)**2+(k*L)**2))
        else:
            if np.imag(kl) == 0:
                return Function_of_n(lambda k: np.sqrt(L/np.pi)/np.sqrt(1 + np.sin(kl*L)/(kl*L))*(np.sin((kl+k)*L/2)/(kl*L+k*L) + np.sin((kl-k)*L/2)/(kl*L-k*L)))
            else:
                kappal = np.imag(kl)
                return Function_of_n(lambda k: (2)*np.sqrt(L/np.pi)/np.sqrt(1+np.sinh(kappal*L)/(kappal*L))*(kappal*L*np.cos(k*L/2)*np.sinh(kappal*L/2) + k*L*np.sin(k*L/2)*np.cosh(kappal*L/2))/((kappal*L)**2+(k*L)**2))

    def get_x_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        if lhs_state%2 == rhs_state%2:
            return 0
        
        if lhs_state%2 == 1:
            temp = lhs_state
            lhs_state = rhs_state
            rhs_state = temp

        lhs_k = self._l_kl_map.get_kl(lhs_state)
        rhs_k = self._l_kl_map.get_kl(rhs_state)
        L = self._L


        if np.imag(lhs_k) == 0:
            if np.imag(rhs_k) == 0:
                cos_expr = np.cos((lhs_k-rhs_k)*L/2)/((lhs_k-rhs_k)*L) - np.cos((lhs_k+rhs_k)*L/2)/((lhs_k+rhs_k)*L)
                sin_expr = np.sin((lhs_k+rhs_k)*L/2)/(((lhs_k+rhs_k)*L)**2) - np.sin((lhs_k-rhs_k)*L/2)/(((lhs_k-rhs_k)*L)**2)
                norm_expr = np.sqrt((1+np.sin(lhs_k*L)/(lhs_k*L))*(1-np.sin(rhs_k*L)/(rhs_k*L)))
                return (cos_expr + 2*sin_expr)/norm_expr*L

            else:
                rhs_kappa = np.imag(rhs_k)

                norm_kappa = rhs_kappa*L/2
                norm_k = lhs_k*L/2 

                cos_cosh_expr = rhs_kappa*np.cos(norm_k)*np.cosh(norm_kappa)
                sin_sinh_expr = lhs_k*np.sin(norm_k)*np.sinh(norm_kappa)
                symm_expr = L/(rhs_kappa**2+lhs_k**2)*(sin_sinh_expr + cos_cosh_expr)

                cos_sinh_expr = (lhs_k**2-rhs_kappa**2)*np.cos(norm_k)*np.sinh(norm_kappa)
                sin_cosh_expr = 2*rhs_kappa*lhs_k*np.sin(norm_k)*np.cosh(norm_kappa)
                anti_symm_expr = 2/((rhs_kappa**2+lhs_k**2)**2)*(cos_sinh_expr - sin_cosh_expr)

                norm_expr = np.sqrt((1+np.sin(lhs_k*L)/(lhs_k*L))*(-1+np.sinh(rhs_kappa*L)/(rhs_kappa*L)))
                
                return (2/L)*(symm_expr + anti_symm_expr)/norm_expr
                
        else:
            lhs_kappa = np.imag(lhs_k)

            if np.imag(rhs_k) == 0:
                norm_kappa = lhs_kappa*L/2
                norm_k = rhs_k*L/2

                norm_expr = np.sqrt((1-np.sin(rhs_k*L)/(rhs_k*L))*(1+np.sinh(lhs_kappa*L)/(lhs_kappa*L)))

                sin_sinh_expr = lhs_kappa*np.sin(norm_k)*np.sinh(norm_kappa)
                cos_cosh_expr = rhs_k*np.cos(norm_k)*np.cosh(norm_kappa)
                symm_expr = L/(lhs_kappa**2+rhs_k**2)*(sin_sinh_expr - cos_cosh_expr)

                cos_sinh_expr = 2*lhs_kappa*rhs_k*np.cos(norm_k)*np.sinh(norm_kappa)
                sin_cosh_expr = (rhs_k**2-lhs_kappa**2)*np.sin(norm_k)*np.cosh(norm_kappa)
                anti_symm_expr = 2/((rhs_k**2+lhs_kappa**2)**2)*(sin_cosh_expr + cos_sinh_expr)

                return (2/L)*(symm_expr + anti_symm_expr)/norm_expr

            else:
                rhs_kappa = np.imag(rhs_k)
                cosh_expr = np.cosh((lhs_kappa+rhs_kappa)*L/2)/((lhs_kappa+rhs_kappa)*L) - np.cosh((lhs_kappa-rhs_kappa)*L/2)/((lhs_kappa-rhs_kappa)*L)
                sinh_expr = np.sinh((lhs_kappa-rhs_kappa)*L/2)/((lhs_kappa*L-rhs_kappa*L)**2) - np.sinh((lhs_kappa+rhs_kappa)*L/2)/((lhs_kappa*L+rhs_kappa*L)**2)
                norm_expr = np.sqrt((1+np.sinh(lhs_kappa*L)/(lhs_kappa*L))*(-1+np.sinh(rhs_kappa*L)/(rhs_kappa*L)))
                return (cosh_expr + 2*sinh_expr)/norm_expr*L

    def get_pR_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        #print("computing the pR elements using analytic results...")
        if lhs_state%2 == rhs_state%2:
            return 0
        
        lhs_k = self._l_kl_map.get_kl(lhs_state)
        rhs_k = self._l_kl_map.get_kl(rhs_state)
        L = self._L
        lhs_sign = (-1)**lhs_state
        rhs_sign = -lhs_sign

        if np.imag(lhs_k) == 0:
            if np.imag(rhs_k) == 0:

                norm_expr = np.sqrt((1 + lhs_sign*np.sin(lhs_k*L)/(lhs_k*L))*(1 + rhs_sign*np.sin(rhs_k*L)/(rhs_k*L)))
                sin_expr = np.sin((lhs_k+rhs_k)*L/2)/((lhs_k+rhs_k)*L) + lhs_sign*np.sin((lhs_k-rhs_k)*L/2)/((lhs_k-rhs_k)*L)
                return (-2j)*rhs_k*sin_expr/norm_expr

            else:
                rhs_kappa = np.imag(rhs_k)

                norm_kappa = rhs_kappa*L/2
                norm_k = lhs_k*L/2
                norm_expr = np.sqrt((1 + lhs_sign*np.sin(lhs_k*L)/(lhs_k*L))*(rhs_sign + np.sinh(rhs_kappa*L)/(rhs_kappa*L)))
                
                if rhs_state%2 == 0:
                    sin_cosh_expr = rhs_kappa*np.sin(norm_k)*np.cosh(norm_kappa)
                    cos_sinh_expr = -lhs_k*np.cos(norm_k)*np.sinh(norm_kappa)

                else:
                    sin_cosh_expr = lhs_k*np.sin(norm_k)*np.cosh(norm_kappa)
                    cos_sinh_expr = rhs_kappa*np.cos(norm_k)*np.sinh(norm_kappa)
                
                anti_symm_expr = 2/(lhs_k**2+rhs_kappa**2)*(sin_cosh_expr + cos_sinh_expr)
                return (-1j*rhs_kappa)*(2/L)*anti_symm_expr/norm_expr

        else:
            lhs_kappa = np.imag(lhs_k)
            if np.imag(rhs_k) == 0:
                norm_kappa = lhs_kappa*L/2
                norm_k = rhs_k*L/2

                norm_expr = np.sqrt((1 + rhs_sign*np.sin(rhs_k*L)/(rhs_k*L))*(lhs_sign + np.sinh(lhs_kappa*L)/(lhs_kappa*L)))
                if lhs_state%2 == 0:
                    sin_cosh_expr = rhs_k*np.sin(norm_k)*np.cosh(norm_kappa)
                    cos_sinh_expr = lhs_kappa*np.cos(norm_k)*np.sinh(norm_kappa)

                else:                 
                    sin_cosh_expr = lhs_kappa*np.sin(norm_k)*np.cosh(norm_kappa)
                    cos_sinh_expr = -rhs_k*np.cos(norm_k)*np.sinh(norm_kappa)

                anti_symm_expr = 2/(lhs_kappa**2+rhs_k**2)*(sin_cosh_expr + cos_sinh_expr)
                return (rhs_sign*1j*rhs_k)*(2/L)*anti_symm_expr/norm_expr

                
            else:
                rhs_kappa = np.imag(rhs_k)
                norm_expr = np.sqrt((lhs_sign + np.sinh(lhs_kappa*L)/(lhs_kappa*L))*(rhs_sign + np.sinh(rhs_kappa*L)/(rhs_kappa*L)))
                sinh_expr = np.sinh((lhs_kappa+rhs_kappa)*L/2)/((lhs_kappa+rhs_kappa)*L) + lhs_sign*np.sinh((lhs_kappa-rhs_kappa)*L/2)/((lhs_kappa-rhs_kappa)*L)
                return (-2j)*rhs_kappa*sinh_expr/norm_expr

    def discrete_momentum_projection_helper(self, l: int, n_array: np.ndarray) -> np.ndarray:
        kn_array = self.get_kn(n_array)
        temp_k_space_proj = np.sqrt(np.pi/self._L)*self.get_k_space_projection(l)
        return temp_k_space_proj(kn_array)

    def get_new_k_space_projection(self, l: int) -> Function_of_n:
        return Function_of_n(lambda n: self.discrete_momentum_projection_helper(l, n))

    
class Neumann_Boudnary(New_Style_Boundary):
    def __init__(self, L: float, gamma: float, theta: float, l_to_kl_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, gamma, theta, l_to_kl_mapper_ref)

    def get_kn(self, n: int | list) -> float | list:
        return n*np.pi/self._L

    def get_kl(self, l: int) -> complex:
        return l*np.pi/self._L

    def get_x_space_projection(self, l: int) -> Function_of_n:
        L = self._L
        if l == 0:
            return Function_of_n(lambda x: 1/np.sqrt(L)*np.ones(np.shape(x)))
        else:
            if l%2 == 0:
                return Function_of_n(lambda x: np.sqrt(2/L)*np.cos(l*np.pi/L*x))
            else:
                return Function_of_n(lambda x: np.sqrt(2/L)*np.sin(l*np.pi/L*x))

    def get_x_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        if lhs_state%2 == rhs_state%2:
            return 0
        
        if lhs_state%2 == 1:
            temp_state = lhs_state
            lhs_state = rhs_state
            rhs_state = temp_state
        
        L = self._L

        if lhs_state == 0:
            return (2*np.sqrt(2)*L/(np.pi*rhs_state)**2)*(-1)**((rhs_state-1)/2)
        
        else:
            return (2*L/np.pi**2)*(-1)**((lhs_state+rhs_state-1)/2)*2*(lhs_state**2 + rhs_state**2)/(lhs_state**2 - rhs_state**2)**2

    def get_k_space_projection(self, l: int) -> Function_of_n:
        L = self._L

        if l == 0:
            return Function_of_n(lambda k: np.sqrt(2*L/np.pi)*np.sin(k*L/2)/(k*L))
        
        if l%2 == 0:
            return Function_of_n(lambda k: np.sqrt(L/np.pi)*(np.sin(l*np.pi/2 + k*L/2)/(l*np.pi + k*L) + np.sin(l*np.pi/2 - k*L/2)/(l*np.pi - k*L)))

        else:
            return Function_of_n(lambda k: 1j*np.sqrt(L/np.pi)*(np.sin(l*np.pi/2 + k*L/2)/(l*np.pi + k*L) - np.sin(l*np.pi/2 - k*L/2)/(l*np.pi - k*L)))

    def get_pR_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        L = self._L
        if lhs_state%2 == rhs_state%2:
            return 0
        
        if rhs_state == 0:
            # lhs_state != 0 is already implicitly given 
            # as otherwise lhs_state%2 == rhs_state%2 
            return 1j*np.sqrt(2)/L*(-1)**((lhs_state-1)/2)
        
        elif lhs_state == 0:
            # rhs_state != 0 is already implicitly given 
            # as otherwise lhs_state%2 == rhs_state%2 
            return -1j*np.sqrt(2)/L*(-1)**((rhs_state-1)/2)

        else:
            # The only case that reamains is when neither rhs_state = 0 nor 
            # lhs_state = 0 and lhs_state%2 != rhs_state%2
            return 2j/L*(-1)**((lhs_state+rhs_state-1)/2)*(lhs_state**2 + rhs_state**2)/(lhs_state**2 - rhs_state**2)

    def discrete_momentum_projection_helper(self, l: int, n_array: np.ndarray) -> np.ndarray:
        if isinstance(n_array, int):
            n_array = [n_array]

        projection_coefficients = []
        if l == 0:
            for n in n_array:
                if n%2 == 0:
                    coeff_append = 1/np.sqrt(2) if n==0 else 0
                else:
                    coeff_append = np.sqrt(2)/(np.pi*n)*(-1)**((n-1)/2)
                
                projection_coefficients.append(coeff_append)
            
            return np.array(projection_coefficients)
        
        if l%2 == 1:
            for n in n_array:
                if n%2 == 0:
                    coeff_append = 2j/np.pi*(-1)**((n+l-1)/2)*n/(n**2-l**2)
                elif n == l:
                    coeff_append = -1j/2
                elif n == -l:
                    coeff_append = 1j/2
                else:
                    coeff_append = 0
                
                projection_coefficients.append(coeff_append)
            
            return np.array(projection_coefficients)
        
        elif l%2 == 0:
            for n in n_array:
                if n%2 == 1:
                    coeff_append = 2/np.pi*(-1)**((n+l-1)/2)*n/(n**2-l**2)
                elif abs(n) == abs(l):
                    coeff_append = 1/2
                else:
                    coeff_append = 0
                
                projection_coefficients.append(coeff_append)
            
            return np.array(projection_coefficients)

    def get_new_k_space_projection(self, l: int) -> Function_of_n:
        return Function_of_n(lambda n: self.discrete_momentum_projection_helper(l, n))

    def set_theta(self, new_theta: float) -> None:
        super().set_theta(new_theta)
        warnings.warn("setting theta has not been implemented for pure Neumann boundaries yet and will thus have no effect")


class Dirichlet_Boundary(New_Style_Boundary):
    def __init__(self, L: float, gamma: float, theta: float, l_to_kl_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, gamma, theta, l_to_kl_mapper_ref)

    def get_kn(self, n: int | list) -> float | list:
        return n*np.pi/self._L

    def get_kl(self, l: int) -> complex:
        return (l+1)*np.pi/self._L

    def get_x_space_projection(self, l: int) -> Function_of_n:
        L = self._L
        if l%2 == 0:
            return Function_of_n(lambda x: np.sqrt(2/L)*np.cos((l+1)*np.pi/L*x))
        else:
            return Function_of_n(lambda x: np.sqrt(2/L)*np.sin((l+1)*np.pi/L*x))
    
    def get_x_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        if lhs_state%2 == rhs_state%2:
            return 0
        
        else:
            L = self._L
            sign_expr = (-1)**((lhs_state+rhs_state-1)/2)
            return (2*L/np.pi**2)*sign_expr*(4*(lhs_state+1)*(rhs_state+1))/((lhs_state+1)**2 - (rhs_state+1)**2)**2      
        
    def get_pR_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        if lhs_state%2 == rhs_state%2:
            return 0
        
        else:
            L = self._L
            sign_expr = (-1)**((lhs_state+rhs_state-1)/2)
            return 1j/L*sign_expr*(4*(lhs_state+1)*(rhs_state+1))/((lhs_state+1)**2 - (rhs_state+1)**2)

    def get_k_space_projection(self, l: int) -> Function_of_n:
        L = self._L
        i_factor = lambda l: 1j if l%2 == 1 else 1
        sign_factor = (-1)**l
        return Function_of_n(lambda k: i_factor(l)*np.sqrt(L/np.pi)*(np.sin((l+1)*np.pi/2 + k*L/2)/((l+1)*np.pi + k*L) + sign_factor*np.sin((l+1)*np.pi/2 - k*L/2)/((l+1)*np.pi - k*L)))

    def discrete_momentum_projection_helper(self, l: int, n_array: np.ndarray) -> np.ndarray:
        if isinstance(n_array, int):
            n_array = [n_array]
        
        projection_coefficients = []
        if l%2 == 0:
            for n in n_array:
                if n%2 == 0:
                    coeff_append = 2/np.pi*(-1)**((l+n)/2)*(l+1)/((l+1)**2 - n**2)
                elif abs(n) == abs(l+1):
                    coeff_append = 1/2
                else:
                    coeff_append = 0
            
                projection_coefficients.append(coeff_append)
            
            return np.array(projection_coefficients)
        
        elif l%2 == 1:
            for n in n_array:
                if n%2 == 1:
                    coeff_append = 2j/np.pi*(-1)**((l+n)/2)*(l+1)/((l+1)**2 - n**2)
                elif n == l+1:
                    coeff_append = -1j/2
                elif n == -(l+1):
                    coeff_append = 1j/2
                else:
                    coeff_append = 0

                projection_coefficients.append(coeff_append)
            
            return np.array(projection_coefficients)

    def get_new_k_space_projection(self, l: int) -> Function_of_n:
        return Function_of_n(lambda n: self.discrete_momentum_projection_helper(l, n))

    def set_theta(self, new_theta: float) -> None:
        super().set_theta(new_theta)
        warnings.warn("setting theta has not been implemented for pure Dirichlet boundaries yet and will thus have no effect")


class Dirichlet_Neumann_Boundary(New_Style_Boundary):
    def __init__(self, L: float, gamma: float, theta: float, l_to_kl_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, gamma, theta, l_to_kl_mapper_ref)

    def get_kn(self, n: int | list) -> float | list:
        return (n+1/2)*np.pi/self._L
    
    def get_kl(self, l: int) -> complex:
        return (2*l+1)/2*np.pi/self._L
    
    def get_x_space_projection(self, l: int) -> Function_of_n:
        L = self._L
        kl = self._l_kl_map.get_kl(l)
        return Function_of_n(lambda x: np.sqrt(2/L)*np.sin(kl*(x+L/2)))

    def get_k_space_projection(self, l: int) -> Function_of_n:
        L = self._L
        kl = self._l_kl_map.get_kl(l)
        lhs_term = Function_of_n(lambda k: np.sin((kl+k)*L/2)/((kl+k)*L)*np.exp(-1j*(2*l+1)*np.pi/4))
        rhs_term = Function_of_n(lambda k: np.sin((kl-k)*L/2)/((kl-k)*L)*np.exp(1j*(2*l+1)*np.pi/4))
        return 1j*np.sqrt(L/np.pi)*(lhs_term - rhs_term)

    def discrete_momentum_projection_helper(self, l: int, n_array: np.ndarray) -> np.ndarray:
        if isinstance(n_array, int):
            n_array = [n_array]
        
        projection_coefficients = []

        for n in n_array:
            if n == l:
                coeff_append = 1j/np.pi*(-1)**(l)*np.exp(-1j*(2*l+1)*np.pi/4)/(2*l+1) - 1j/2*np.exp(1j*(2*l+1)*np.pi/4)
            
            elif l+n == -1:
                coeff_append = 1j/2*np.exp(-1j*(2*l+1)*np.pi/4) - 1j/np.pi*(-1)**(l)*np.exp(1j*(2*l+1)*np.pi/4)/(2*l+1)

            elif (n+l)%2 == 0:
                coeff_append = 1j/np.pi*(-1)**((l+n)/2)*np.exp(-1j*(2*l+1)*np.pi/4)/(l+n+1)
            
            elif (n+l)%2 == 1:
                coeff_append = -1j/np.pi*(-1)**((l-n-1)/2)*np.exp(1j*(2*l+1)*np.pi/4)/(l-n)
            
            else:
                print("eh?")

            projection_coefficients.append(coeff_append)
        
        return np.array(projection_coefficients)

    def get_new_k_space_projection(self, l: int) -> Function_of_n:
        return Function_of_n(lambda n: self.discrete_momentum_projection_helper(l, n))

    def get_x_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        L = self._L
       
        if lhs_state%2 == rhs_state%2:
            return (2*L/np.pi**2)/(lhs_state+rhs_state+1)**2
        else:
            return -(2*L/np.pi**2)/(lhs_state-rhs_state)**2

    def get_pR_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        L = self._L

        if lhs_state%2 == rhs_state%2:
            return 1j/L*(lhs_state-rhs_state)/(lhs_state+rhs_state+1)
        else:
            return 1j/L*(lhs_state+rhs_state+1)/(rhs_state-lhs_state)

    def set_theta(self, new_theta: float) -> None:
        super().set_theta(new_theta)
        warnings.warn("setting theta has not been implemented for Dirichlet Neumann boundaries yet and will thus have no effect")


class Anti_Symmetric_Boundary(New_Style_Boundary):
    def __init__(self, L: float, gamma: float, theta: float, l_to_kl_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, gamma, theta, l_to_kl_mapper_ref)

    @staticmethod
    def x_space_projection_for_nummerics(L, gamma, l, kl) -> Function_of_n:
        i_factor = lambda l: 1 if l%2 == 0 else -1j

        if np.imag(kl) == 0:
            boundray_expr = ((-1)**l)*(gamma + 1j*kl)/(gamma - 1j*kl)
            return Function_of_n(lambda x: i_factor(l)/(np.sqrt(2*L))*(np.exp(1j*kl*x) - boundray_expr*np.exp(-1j*kl*x)))
        
        else:
            return Function_of_n(lambda x: np.sqrt(gamma/np.sinh(gamma*L))*np.exp(-gamma*x))

    def get_kn(self, n: int | list) -> float | list:
        return n*np.pi/self._L + self._theta/(2*self._L)

    def get_kl(self, l: int) -> complex:
        if l == 0:
            return 1j*self._gamma if self._gamma > 0 else -1j*self._gamma
        else:
            return l*np.pi/self._L

    def get_x_space_projection(self, l: int) -> Function_of_n:
        gamma = self._gamma
        L = self._L
        kl = self._l_kl_map.get_kl(l)

        return self.x_space_projection_for_nummerics(L, gamma, l, kl)

    def get_x_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        gamma = self._gamma
        L = self._L

        lhs_k = self._l_kl_map.get_kl(lhs_state)
        rhs_k = self._l_kl_map.get_kl(rhs_state)

        lhs_integrand = self.x_space_projection_for_nummerics(L, gamma, lhs_state, lhs_k)
        rhs_integrand = self.x_space_projection_for_nummerics(L, gamma, rhs_state, rhs_k)
        integrand = lambda x: np.conj(lhs_integrand(x))*x*rhs_integrand(x)
        real = quad(lambda x: np.real(integrand(x)), -L/2, L/2)[0]
        imag = quad(lambda x: np.imag(integrand(x)), -L/2, L/2)[0]

        return real + 1j*imag

    def get_k_space_projection(self, l: int) -> Function_of_n:
        gamma = self._gamma
        L = self._L
        kl = self._l_kl_map.get_kl(l)

        x_space_proj = self.x_space_projection_for_nummerics(L, gamma, l, kl)

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
        
        return Function_of_n(converter)

    def discrete_momentum_projection_helper(self, l: int, n_array: np.ndarray) -> np.ndarray:
        kn_array = self.get_kn(n_array)
        temp_k_space_proj = np.sqrt(np.pi/self._L)*self.get_k_space_projection(l)
        return temp_k_space_proj(kn_array)

    def get_new_k_space_projection(self, l: int) -> Function_of_n:
        return Function_of_n(lambda n: self.discrete_momentum_projection_helper(l, n))

    def get_pR_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        gamma = self._gamma
        L = self._L

        lhs_k = self._l_kl_map.get_kl(lhs_state)
        rhs_k = self._l_kl_map.get_kl(rhs_state)

        lhs_integrand = self.x_space_projection_for_nummerics(L, gamma, lhs_state, lhs_k)
        rhs_integrand = self.x_space_projection_for_nummerics(L, gamma, rhs_state, rhs_k)

        integrand = lambda x: (-1j)*np.conj(lhs_integrand(x))*derivative(rhs_integrand, x, 0.0001)

        real = quad(lambda x: np.real(integrand(x)), -L/2, L/2)[0]
        imag = quad(lambda x: np.imag(integrand(x)), -L/2, L/2)[0]


        return real + 1j*imag
    

class Symmetric_Nummeric(Symmetric_Boundary):
    def __init__(self, L: float, gamma: float, theta: float, l_to_kl_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, gamma, theta, l_to_kl_mapper_ref)
        self._n_range = 100

    def get_pR_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        # This implementation of the <get_pR_matrix_element> method determines
        # the matrix elements <l|pR|l'> by expanding the energy states in the
        # momentum eigenbasis such that the matrix elements are obtained by
        # summing over all momentum states.

        lhs_proj_coeffs = self.get_new_k_space_projection(lhs_state)
        rhs_proj_coeffs = self.get_new_k_space_projection(rhs_state)
        
        # Construction of one or two intervals of momentum quantum numbers that
        # are taken as samples to approximate the infinite sum given in
        # <l|pR|l'> if expanded in the momentum basis. 
        n_center = (lhs_state+rhs_state)//2
        n_range_adapted = abs(lhs_state-rhs_state)//2+self._n_range
        n_range_u = n_range_adapted
        n_range_d = n_range_adapted if n_center > n_range_adapted else n_center

        n_p = np.arange(n_center-n_range_d+1, n_center+n_range_u+1, 1)
        n_n = np.arange(-n_center-n_range_u, -n_center+n_range_d+1, 1)
        n = np.append(n_n, n_p)
        
        return np.sum(self.get_kn(n)*np.conj(lhs_proj_coeffs(n))*rhs_proj_coeffs(n))

    def set_n_range(self, new_n_range) -> None:
        self._n_range = new_n_range


class Anti_Symmetric_Nummeric(Anti_Symmetric_Boundary):
    def __init__(self, L: float, gamma: float, theta: float, l_to_kl_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, gamma, theta, l_to_kl_mapper_ref)
        self._n_range = 100

    def get_pR_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        lhs_proj_coeffs = self.get_new_k_space_projection(lhs_state)
        rhs_proj_coeffs = self.get_new_k_space_projection(rhs_state)
        
        n_center = max(lhs_state, rhs_state)
        n_range_u = self._n_range
        n_range_d = self._n_range if n_center > self._n_range else n_center
        n_p = np.arange(n_center-n_range_d+1, n_center+n_range_u+1, 1)
        n_n = np.arange(-n_center-n_range_u, -n_center+n_range_d+1, 1)
        n = np.append(n_n, n_p)
        
        return np.sum(self.get_kn(n)*np.conj(lhs_proj_coeffs(n))*rhs_proj_coeffs(n))

    def set_n_range(self, new_n_range) -> None:
        self._n_range = new_n_range
