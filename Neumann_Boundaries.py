from Backend import *

class Gamma_to_k(Gamma_to_k_Base):
    def __init__(self, L: float) -> None:
        super().__init__(L)
    
    def __call__(self, l: int) -> complex:
        return l*np.pi/self._L


class X_Space_Projector(Energy_State_Projector):
    def __init__(self, L: float, gamma: float, l_to_k_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, gamma, l_to_k_mapper_ref)
    
    def get_projection(self, l: int) -> Function_of_array:
        L = self._L
        if l == 0:
            return Function_of_array(lambda x: 1/np.sqrt(L)*np.ones(np.shape(x)))
        else:
            if l%2 == 0:
                return Function_of_array(lambda x: np.sqrt(2/L)*np.cos(l*np.pi/L*x))
            else:
                return Function_of_array(lambda x: np.sqrt(2/L)*np.sin(l*np.pi/L*x))


class Bra_l1_x_Ket_l2(Energy_State_Matrix_Elements):
    def __init__(self, L: float, gamma: float, l_to_k_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, gamma, l_to_k_mapper_ref)
    
    def get_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
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


def discrete_momentum_projection(l: int, n_array: np.ndarray) -> np.ndarray:
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
                coeff_append = -2j/np.pi*(-1)**((n+l-1)/2)*n/(n**2-l**2)
            elif n == l:
                coeff_append = 1j/2
            elif n == -l:
                coeff_append = -1j/2
            else:
                coeff_append = 0
            
            projection_coefficients.append(coeff_append)
        
        return np.array(projection_coefficients)
    
    if l%2 == 0:
        for n in n_array:
            if n%2 == 1:
                coeff_append = 2/np.pi*(-1)**((n+l-1)/2)*n/(n**2-l**2)
            elif abs(n) == abs(l):
                coeff_append = 1/2
            else:
                coeff_append = 0
            
            projection_coefficients.append(coeff_append)
        
        return np.array(projection_coefficients)

            
class New_K_Space_Projector(Energy_State_Projector):
    def __init__(self, L: float, gamma: float, l_to_k_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, gamma, l_to_k_mapper_ref)

    def get_projection(self, l: int) -> Function_of_array:
        return Function_of_array(lambda n: discrete_momentum_projection(l, n))
        

class K_Space_Projector(Energy_State_Projector):
    def __init__(self, L: float, gamma: float, l_to_k_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, gamma, l_to_k_mapper_ref)

    def get_projection(self, l: int) -> Function_of_array:
        L = self._L

        if l == 0:
            return Function_of_array(lambda k: np.sqrt(2*L/np.pi)*np.sin(k*L/2)/(k*L))
        
        if l%2 == 0:
            return Function_of_array(lambda k: np.sqrt(L/np.pi)*(np.sin(l*np.pi/2 + k*L/2)/(l*np.pi + k*L) + np.sin(l*np.pi/2 - k*L/2)/(l*np.pi - k*L)))

        else:
            return Function_of_array(lambda k: 1j*np.sqrt(L/np.pi)*(np.sin(l*np.pi/2 + k*L/2)/(l*np.pi + k*L) - np.sin(l*np.pi/2 - k*L/2)/(l*np.pi - k*L)))


class Bra_l1_pR_Ket_l2(Energy_State_Matrix_Elements):
    def __init__(self, L: float, gamma: float, l_to_k_mapper_ref: l_to_kl_mapper) -> None:
        super().__init__(L, gamma, l_to_k_mapper_ref)

    def get_matrix_element(self, lhs_state: int, rhs_state: int) -> complex:
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
            

class Neumann_Boundary(Boundary):
    def __init__(self, L: float, gamma: float, l_to_kl_mapper_ref: l_to_kl_mapper) -> None:
        pass

    def X_Space_Projector(self) -> Energy_State_Projector:
        return super().X_Space_Projector

    def K_Space_Projector(self) -> Energy_State_Projector:
        return super().K_Space_Projector

    def Bra_l1_x_Ket_l2(self) -> Energy_State_Matrix_Elements:
        return super().Bra_l1_x_Ket_l2

    def Bra_l1_pR_Ket_l2(self) -> Energy_State_Matrix_Elements:
        return super().Bra_l1_pR_Ket_l2

    def Gamma_to_k(self) -> Gamma_to_k_Base:
        return super().Gamma_to_k