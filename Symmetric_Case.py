from Backend import *
from scipy.optimize import fsolve

posRelEven = lambda g, k: g-np.arctan(k*np.tan(k/2))
posRelOdd = lambda g, k: g+np.arctan(k/(np.tan(k/2)))

negRelEven = lambda g, k: g+np.arctan(k*np.tanh(k/2))
negRelOdd = lambda g, k: g+np.arctan(k/np.tanh(k/2))

def gamma_to_k(gamma, l, L):
    gammaPrime = np.arctan(gamma*L)
    length = np.size(gamma)

    if gamma == 0:
        return np.full(length, l*np.pi/L)
    
    if gamma == np.inf:
        return np.full(length, (l-1)*np.pi/L)

    if l > 2:
        if l%2 == 0:
            rel = posRelOdd
            ##print("Odd Case")
        else:
            rel = posRelEven
            ##print("Even Case")

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
        if l%2 == 0:
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

        if l%2 == 0:
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
        if lhs_state%2 == rhs_state%2:
            return 0
        
        if lhs_state%2 == 0:
            temp = lhs_state
            lhs_state = rhs_state
            rhs_state = temp

        lhs_k = self._l_kl_map.get_kl(lhs_state)
        rhs_k = self._l_kl_map.get_kl(rhs_state)
        L = self._L

        cos_expr = np.cos((lhs_k-rhs_k)*L/2)/((lhs_k-rhs_k)*L) - np.cos((lhs_k+rhs_k)*L/2)/((lhs_k+rhs_k)*L)
        sin_expr = 2*(np.sin((lhs_k+rhs_k)*L/2)/(((lhs_k+rhs_k)*L)**2) - np.sin((lhs_k-rhs_k)*L/2)/(((lhs_k-rhs_k)*L)**2))
        norm_expr = np.sqrt((1+np.sin(lhs_k*L)/(lhs_k*L))*(1-np.sin(rhs_k*L)/(rhs_k*L)))
        return (cos_expr + sin_expr)/norm_expr*L
    
class Bra_l1_pR_Ket_l2(Energy_State_Matrix_Elements):
    def __init__(self, L: float, l_to_k_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, l_to_k_mapper_ref)

    def get_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
        if lhs_state%2 == rhs_state%2:
            return 0
        
        lhs_k = self._l_kl_map.get_kl(lhs_state)
        rhs_k = self._l_kl_map.get_kl(rhs_state)
        L = self._L

        if lhs_state%2 == 1:
            norm_expr = np.sqrt((1+np.sin(lhs_k*L)/(lhs_k*L))*(1-np.sin(rhs_k*L)/rhs_k*L))
            sin_expr = np.sin((lhs_k+rhs_k)*L/2)/((lhs_k+rhs_k)*L) + np.sin((lhs_k-rhs_k)*L/2)/((lhs_k-rhs_k)*L)

            return -2j*rhs_k*norm_expr*sin_expr
        
        elif lhs_state%2 == 0:
            norm_expr = np.sqrt((1-np.sin(lhs_k*L)/(lhs_k*L))*(1+np.sin(rhs_k*L)/rhs_k*L))
            sin_expr = np.sin((lhs_k+rhs_k)*L/2)/((lhs_k+rhs_k)*L) - np.sin((lhs_k-rhs_k)*L/2)/((lhs_k-rhs_k)*L)

            return -2j*rhs_k*norm_expr*sin_expr


class Gamma_to_k(Gamma_to_k_Base):
    def __init__(self, L: float, gamma: float) -> None:
        Gamma_to_k_Base.__init__(self, L)
        self._gamma = gamma

    def __call__(self, l: int) -> complex:
        return gamma_to_k(self._gamma, l, self._L)[0]

    def set_gamma(self, gamma: float) -> None:
        self._gamma = gamma

