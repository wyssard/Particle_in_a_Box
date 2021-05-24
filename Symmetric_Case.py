from __future__ import annotations

from Backend import *
from scipy.optimize import fsolve
from scipy.optimize import brentq
import sys

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
            ###print("Odd Case")
        else:
            rel = posRelEven
            ###print("Even Case")

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



class X_Space_Projector(Energy_State_Projector):
    def __init__(self, L: float, l_kl_map: l_to_kl_mapper):
        Energy_State_Projector.__init__(self, L, l_kl_map)

    def get_projection(self, l: int) -> Function_of_array:
        L = self._L
        kl = self._l_kl_map.get_kl(l)
        if l%2 == 1:
            if np.imag(kl) == 0:
                return Function_of_array(lambda x: np.sqrt(2/L)*np.power(1-np.sin(kl*L)/(kl*L), -1/2)*np.sin(kl*x))
            else:
                kappal = np.imag(kl)
                return Function_of_array(lambda x: np.sqrt(2/L)*np.power(-1+np.sinh(kappal*L)/(kappal*L), -1/2)*np.sinh(kappal*x))
        else:
            if np.imag(kl) == 0:
                return Function_of_array(lambda x: np.sqrt(2/L)*np.power(1+np.sin(kl*L)/(kl*L), -1/2)*np.cos(kl*x))
            else:
                kappal = np.imag(kl)
                return Function_of_array(lambda x: np.sqrt(2/L)*np.power(1+np.sinh(kappal*L)/(kappal*L), -1/2)*np.cosh(kappal*x))

class K_Space_Projector(Energy_State_Projector):
    def __init__(self, L: float, l_kl_map: l_to_kl_mapper):
        Energy_State_Projector.__init__(self, L, l_kl_map)

    def get_projection(self, l: int) -> Function_of_array:
        L = self._L
        kl = self._l_kl_map.get_kl(l)

        if l%2 == 1:
            if np.imag(kl) == 0:
                return Function_of_array(lambda k: 1j*np.sqrt(L/np.pi)/np.sqrt(1 - np.sin(kl*L)/(kl*L))*(np.sin((kl+k)*L/2)/(kl*L+k*L) - np.sin((kl-k)*L/2)/(kl*L-k*L)))
            else:
                kappal = np.imag(kl)
                return Function_of_array(lambda k: (2j)*np.sqrt(L/np.pi)/np.sqrt(-1+np.sinh(kappal*L)/(kappal*L))*(k*L*np.cos(k*L/2)*np.sinh(kappal*L/2) - kappal*L*np.sin(k*L/2)*np.cosh(kappal*L/2))/((kappal*L)**2+(k*L)**2))
        else:
            if np.imag(kl) == 0:
                return Function_of_array(lambda k: np.sqrt(L/np.pi)/np.sqrt(1 + np.sin(kl*L)/(kl*L))*(np.sin((kl+k)*L/2)/(kl*L+k*L) + np.sin((kl-k)*L/2)/(kl*L-k*L)))
            else:
                kappal = np.imag(kl)
                return Function_of_array(lambda k: (2)*np.sqrt(L/np.pi)/np.sqrt(1+np.sinh(kappal*L)/(kappal*L))*(k*L*np.cos(k*L/2)*np.sinh(kappal*L/2) + kappal*L*np.sin(k*L/2)*np.cosh(kappal*L/2))/((kappal*L)**2+(k*L)**2))


class Bra_l1_x_Ket_l2(Energy_State_Matrix_Elements):
    def __init__(self, L: float, l_to_k_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, l_to_k_mapper_ref)

    def get_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        #print("computing x matrix element...")
        #print("lhs_state: ", lhs_state, " rhs_state: ", rhs_state)
        if lhs_state%2 == rhs_state%2:
            #print("terminating early")
            return 0
        
        if lhs_state%2 == 1:
            temp = lhs_state
            lhs_state = rhs_state
            rhs_state = temp

        ##print("compute <", lhs_state, "| x |", rhs_state, ">...")

        lhs_k = self._l_kl_map.get_kl(lhs_state)
        rhs_k = self._l_kl_map.get_kl(rhs_state)
        L = self._L

        #print("rhs_k: ", rhs_k, "lhs_k: ", lhs_k)

        if np.imag(lhs_k) == 0:
            if np.imag(rhs_k) == 0:
                cos_expr = np.cos((lhs_k-rhs_k)*L/2)/((lhs_k-rhs_k)*L) - np.cos((lhs_k+rhs_k)*L/2)/((lhs_k+rhs_k)*L)
                sin_expr = np.sin((lhs_k+rhs_k)*L/2)/(((lhs_k+rhs_k)*L)**2) - np.sin((lhs_k-rhs_k)*L/2)/(((lhs_k-rhs_k)*L)**2)
                norm_expr = np.sqrt((1+np.sin(lhs_k*L)/(lhs_k*L))*(1-np.sin(rhs_k*L)/(rhs_k*L)))
                return (cos_expr + 2*sin_expr)/norm_expr*L

            else:
                rhs_kappa = np.imag(rhs_k)
                #print("Aborting Program: the case <l1|x|l2> where |l1> has positive and |l2> has negative energy is not yet implemented")
                sys.exit()
        else:
            lhs_kappa = np.imag(lhs_k)
            if np.imag(rhs_k) == 0:
                #print("Aborting Program: the case <l1|x|l2> where |l1> has negative and |l2> has positive energy is not yet implemented")
                sys.exit()
            else:
                rhs_kappa = np.imag(rhs_k)
                cosh_expr = np.cosh((lhs_kappa+rhs_kappa)*L/2)/((lhs_kappa+rhs_kappa)*L) - np.cosh((lhs_kappa-rhs_kappa)*L/2)/((lhs_kappa-rhs_kappa)*L)
                sinh_expr = np.sinh((lhs_kappa-rhs_kappa)*L/2)/((lhs_kappa*L-rhs_kappa*L)**2) - np.sinh((lhs_kappa+rhs_kappa)*L/2)/((lhs_kappa*L+rhs_kappa*L)**2)
                norm_expr = np.sqrt((1+np.sinh(lhs_kappa*L)/(lhs_kappa*L))*(-1+np.sinh(rhs_kappa*L)/(rhs_kappa*L)))
                return (cosh_expr + 2*sinh_expr)/norm_expr*L

class Bra_l1_pR_Ket_l2(Energy_State_Matrix_Elements):
    def __init__(self, L: float, l_to_k_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, l_to_k_mapper_ref)

    def get_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        #print("computing p matrix element...")
        if lhs_state%2 == rhs_state%2:
            return 0
        
        lhs_k = self._l_kl_map.get_kl(lhs_state)
        rhs_k = self._l_kl_map.get_kl(rhs_state)
        L = self._L
        lhs_sign = (-1)**lhs_state
        rhs_sign = -lhs_sign

        if np.imag(lhs_k) == 0:
            if np.imag(rhs_k) == 0:
                norm_expr = np.sqrt((1 + lhs_sign*np.sin(lhs_k*L)/(lhs_k*L))*(1 + rhs_sign*np.sin(rhs_k*L)/rhs_k*L))
                sin_expr = np.sin((lhs_k+rhs_k)*L/2)/((lhs_k+rhs_k)*L) + lhs_sign*np.sin((lhs_k-rhs_k)*L/2)/((lhs_k-rhs_k)*L)
                return (-2j)*rhs_k*sin_expr/norm_expr

            else:
                rhs_kappa = np.imag(rhs_k)
                #print("Aborting Program: the case <l1|x|l2> where |l1> has positive and |l2> has negative energy is not yet implemented")
                sys.exit()

        else:
            lhs_kappa = np.imag(lhs_k)
            if np.imag(rhs_k) == 0:
                #print("Aborting Program: the case <l1|x|l2> where |l1> has negative and |l2> has positive energy is not yet implemented")
                sys.exit()

            else:

                rhs_kappa = np.imag(rhs_k)
                norm_expr = np.sqrt((lhs_sign + np.sinh(lhs_kappa*L)/(lhs_kappa*L))*(rhs_sign + np.sinh(rhs_kappa*L)/(rhs_kappa*L)))
                sinh_expr = np.sinh((lhs_kappa+rhs_kappa)*L/2)/((lhs_kappa+rhs_kappa)*L) + lhs_sign*np.sinh((lhs_kappa-rhs_kappa)*L/2)/((lhs_kappa-rhs_kappa)*L)
                return (-2j)*rhs_kappa*sinh_expr/norm_expr
        

class Gamma_to_k(Gamma_to_k_Base):
    def __init__(self, L: float, gamma: float | np.ndarray) -> None:
        Gamma_to_k_Base.__init__(self, L)
        self._gamma = gamma
        self._eps = np.finfo(np.float32).eps

        # Observe: kL = k*L =/= kl! We use the product of k and L to get a relation between gamma 
        # and k that does not explicitly depend on L for computational convenience. For the same
        # reason we also use gamma * L instead of gamma
        self._pos_energy_even_state_eq = lambda gammaL, kL: gammaL - kL*np.tan(kL/2)
        self._pos_energy_odd_state_eq = lambda gammaL, kL: gammaL + kL/np.tan(kL/2)
        self._neg_energy_even_state_eq = lambda gammaL, kappaL: gammaL + kappaL*np.tanh(kappaL/2)
        self._neg_energy_odd_state_eq = lambda gammaL, kappaL: gammaL + kappaL/np.tanh(kappaL/2)

    def __call__(self, l: int) -> complex:
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
        
    
    def set_eps(self, eps: float) -> None:
        self._eps = eps


    def set_gamma(self, gamma: float) -> None:
        self._gamma = gamma

