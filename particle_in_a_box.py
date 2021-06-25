from __future__ import annotations
from copy import deepcopy

from Backend import *
import Boundaries

class State_Properties:
    """
    Class that contains all the information of the 'particle in a box' state.
    The latter information consists of the specific boundary condition 'case', 
    the extension parameter 'gamma', the mass of the particle 'm', the width of 
    the box 'L' as well as of a list containing the quantum numbers 'l' of the 
    energy states that our state consits of; the corresponding 'k_l' values are 
    also stored as a list
    """

    def __init__(self, case: str, gamma: float, L: float, m: float) -> None:
        self._gamma = gamma
        self._L = L
        self._m = m
        self._energy_states = []
        self._k_kappa_l_array = []
        self._num_energy_states = 0 

        self._l_kl_map = l_to_kl_mapper(self._energy_states, self._k_kappa_l_array)

        self._case = case
        self.switch_case(case)

         
    def switch_case(self, case: str):
        if case == "symmetric":
            self._boudary_lib = Boundaries.Symmetric_Boundary(self._L, self._gamma, self._l_kl_map)
        
        elif case == "neumann":
            self._boudary_lib = Boundaries.Neumann_Boudnary(self._L, self._gamma, self._l_kl_map)

        elif case == "dirichlet":
            self._boudary_lib = Boundaries.Dirichlet_Boundary(self._L, self._gamma, self._l_kl_map)

        elif case == "dirichlet_neumann":
            self._boudary_lib = Boundaries.Dirichlet_Neumann_Boundary(self._L, self._gamma, self._l_kl_map)

        elif case == "anti_symmetric":
            self._boudary_lib = Boundaries.Anti_Symmetric_Boundary(self._L, self._gamma, self._l_kl_map)

    @property
    def case(self) -> str:
        """string that repersents the boundary condition imposed to the energy
        space wavefunctions"""
        return self._case

    @case.setter
    def case(self, new_case: str) -> None:
        self._case = new_case
        self.switch_case(new_case)

    @property
    def boundary_lib(self) -> New_Style_Boundary:
        """object of type 'New_Style_Boundary' containing all the funcionality
        that specifically depends on the choice of the boundary condition'"""
        return self._boudary_lib

    @property
    def gamma(self) -> float:
        """parameter to specify the Robin boundary conditions under the 
        restrictions for symmetric (gamma_- = gamma_+ =: gamma) or anti 
        symmetric (-gamma_- = gamma_+ =: gamma) boundaries"""
        return self._gamma

    @gamma.setter
    def gamma(self, new_gamma) -> None:
        self._gamma = new_gamma
        self._boudary_lib.set_gamma(new_gamma)
        
        
    @property
    def L(self) -> float:
        """the length of the interval (box)"""
        return self._L

    @L.setter
    def L(self, new_L) -> None:
        print("setting L...(state_properties)")
        self._L = new_L
        self._boudary_lib.set_L(new_L)

    @property
    def m(self) -> float:
        """the mass of the particle"""
        return self._m

    @m.setter
    def m(self, new_m) -> None:
        self._m = new_m
    
    @property
    def num_energy_states(self) -> int:
        """the number of energy states the 'particle in a box'-state is 
        projected onto"""
        return self._num_energy_states
    
    @num_energy_states.setter
    def num_energy_states(self, new_num_energy_states) -> None:
        self._num_energy_states = new_num_energy_states

    @property
    def energy_states(self) -> List[int]:
        """unorderd list containing the quantum numbers 'l' of the energy states
        onto which the 'particle in a box'-state is projected"""
        return self._energy_states

    @property
    def k_kappa_l(self) -> List[complex]:
        """list of the same order as 'energy_states' containing the k_l values
        that correspond to the quantum numbers 'l' found in 'energy_states'"""
        return self._k_kappa_l_array

    @property
    def l_kl_map(self) -> l_to_kl_mapper:
        """see documentation of 'l_to_kl_mapper'"""
        return self._l_kl_map


class Energy_Space_Projection:
    def __init__(self, energies: list, energy_proj_coeffs: np.ndarray, wiggle_factors: List[Wiggle_Factor], state_properties: State_Properties) -> None:
        self._energies = energies
        self._energy_proj_coeffs = energy_proj_coeffs
        self._wiggle_factors = wiggle_factors
        self._sp = state_properties
        self._Norm = 0
        self._exp_value = 0
        self._energy_space_wavefunc = Energy_Space_Wavefunction(self._energies, self._energy_proj_coeffs)
        
    @staticmethod
    def empty_init(state_props: State_Properties) -> Energy_Space_Projection:
        return Energy_Space_Projection([], np.array([]), [], state_props)


    def normalize(self) -> None:
        if self._sp.num_energy_states == 0:
            self._Norm = 0
        else:
            self._Norm = np.sqrt(np.sum(np.power(np.abs(self._energy_proj_coeffs), 2)))
            self._energy_proj_coeffs = self._energy_proj_coeffs*(1/self._Norm)


    def change_coeff(self, the_state, the_coeff):
        the_index = self._sp.l_kl_map.get_index(the_state)
        self._energy_proj_coeffs *= self._Norm
        self._energy_proj_coeffs[the_index] = the_coeff
        self.normalize()

    def compute_expectation_value(self):
        self._exp_value = 0
        for i in range(self._sp.num_energy_states):
            self._exp_value += (np.abs(self._energy_proj_coeffs[i])**2)*self._energies[i]
        
class New_Momentum_Space_Projection:
    def __init__(self, new_k_space_wavefunction: Function_of_n_and_t, new_k_space_single_energy_proj: list, state_properties: State_Properties) -> None:
        self._new_k_space_wavefunction = new_k_space_wavefunction
        self._new_k_space_single_energy_proj = new_k_space_single_energy_proj
        self._sp = state_properties
        self._expectation_value = None_Function()

    @staticmethod
    def empty_init(state_props: State_Properties) -> New_Momentum_Space_Projection:
        return New_Momentum_Space_Projection(None_Function(), [], state_props)

    def combine_wavefunction_components(self, energy_space_proj: Energy_Space_Projection) -> None:
        self._new_k_space_wavefunction = None_Function()
        temp_proj_coeff = energy_space_proj._energy_proj_coeffs
        temp_wigglers = energy_space_proj._wiggle_factors
        for i in range(len(temp_proj_coeff)):
            self._new_k_space_wavefunction += temp_proj_coeff[i]*self._new_k_space_single_energy_proj[i]*temp_wigglers[i]

    def compute_expectation_value(self, energy_space_proj: Energy_Space_Projection) -> None:
        self._expectation_value = None_Function()
        
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

                exp_val_component = self._sp.boundary_lib.get_pR_matrix_element(lh_state, rh_state)

                self._expectation_value += (coeff*wiggler*exp_val_component).get_real_part()

class Momentum_Space_Projection:
    def __init__(self, cont_k_space_wavefunction: Function_of_n_and_t, cont_k_space_single_energy_proj: list, state_properties: State_Properties) -> None:
        self._cont_k_space_wavefunction = cont_k_space_wavefunction
        self._cont_k_space_single_energy_proj = cont_k_space_single_energy_proj
        self._sp = state_properties

    def empty_init(state_props: State_Properties) -> Momentum_Space_Projection:
        return Momentum_Space_Projection(None_Function(), [], state_props)
    
    def combine_wavefunction_components(self, energy_space_proj: Energy_Space_Projection) -> None:
        self._cont_k_space_wavefunction = None_Function()
        temp_proj_coeff = energy_space_proj._energy_proj_coeffs
        temp_wigglers = energy_space_proj._wiggle_factors
        for i in range(len(temp_proj_coeff)):
            self._cont_k_space_wavefunction += temp_proj_coeff[i]*self._cont_k_space_single_energy_proj[i]*temp_wigglers[i]

class Position_Space_Projection:
    def __init__(self, x_space_wavefunction: Function_of_n_and_t, x_space_single_energy_proj: list, state_properties: State_Properties) -> None:
        self._x_space_wavefunction = x_space_wavefunction
        self._x_space_single_energy_proj = x_space_single_energy_proj
        self._sp = state_properties
        self._expectation_value = None_Function()
        self._exp_t_deriv = None_Function()

    @staticmethod
    def empty_init(state_props: State_Properties) -> Position_Space_Projection:
        return Position_Space_Projection(None_Function(), [], state_props)

    def combine_wavefunction_components(self, energy_space_proj: Energy_Space_Projection) -> None:
        self._x_space_wavefunction = None_Function()
        temp_proj_coeff = energy_space_proj._energy_proj_coeffs
        temp_wigglers = energy_space_proj._wiggle_factors
        for i in range(len(temp_proj_coeff)):
            self._x_space_wavefunction += temp_proj_coeff[i]*self._x_space_single_energy_proj[i]*temp_wigglers[i]
       
    def compute_expectation_value(self, energy_space_proj: Energy_Space_Projection) -> None:
        self._expectation_value = None_Function()
        self._exp_t_deriv = None_Function()

        for rh_index in range(1, self._sp.num_energy_states):
            rh_energy = energy_space_proj._energies[rh_index]
            rh_coeff = energy_space_proj._energy_proj_coeffs[rh_index]
            rh_state = self._sp.energy_states[rh_index]

            for lh_index in range(0, rh_index):
                lh_energy = energy_space_proj._energies[lh_index]
                lh_coeff = energy_space_proj._energy_proj_coeffs[lh_index]
                lh_state = self._sp.energy_states[lh_index]

                wiggler = Wiggle_Factor(rh_energy-lh_energy)

                exp_val_component = self._sp.boundary_lib.get_x_matrix_element(lh_state, rh_state) 
                coeff = complex(np.conj(lh_coeff)*rh_coeff)
                append = wiggler*coeff

                self._expectation_value += (2*append*exp_val_component).get_real_part()
                self._exp_t_deriv += (2*(rh_energy-lh_energy)*append*exp_val_component).get_imag_part()

        
        for index in range(self._sp.num_energy_states):
            coeff = energy_space_proj._energy_proj_coeffs[index]
            state = self._sp.energy_states[index]
            exp_val_component = self._sp.boundary_lib.get_x_matrix_element(state, state)

            self._expectation_value += Function_of_t(lambda t: np.abs(coeff)**2*exp_val_component*np.ones(np.shape(t)))


class Particle_in_Box_State:
    def __init__(self, case: str, L: float, m: float, energy_states = [], amplitudes = np.array([]), gamma=None) -> None:
        self._sp = State_Properties(case, gamma, L, m)

        self._esp = Energy_Space_Projection.empty_init(self._sp)
        self._xsp = Position_Space_Projection.empty_init(self._sp)
        self._ksp = Momentum_Space_Projection.empty_init(self._sp)
        self._new_ksp = New_Momentum_Space_Projection.empty_init(self._sp)

        self.add_state(energy_states, amplitudes)


    def change_energy_proj_coeff(self, the_state: int, the_coeff: complex) -> None:
        self._esp.change_coeff(the_state, the_coeff)
        
        self.compute_expectation_values()
        self.combine_wavefunction_components()

    def full_projection_recompute(self) -> None:
        ##print("Recomputing every property that depends on L or gamma...")

        for l in range(self._sp.num_energy_states):
            state = self._sp.energy_states[l]
            k_kappa = self._sp.boundary_lib.get_kl(state)

            self._sp.k_kappa_l[l] = k_kappa

            energy = np.real(k_kappa**2)/(2*self._sp.m)
            self._esp._energies[l] = energy
            self._esp._wiggle_factors[l] = Wiggle_Factor(energy)

            k_proj = self._sp.boundary_lib.get_k_space_projection(state)
            x_proj = self._sp.boundary_lib.get_x_space_projection(state)
            new_k_proj = self._sp.boundary_lib.get_new_k_space_projection(state)
            self._xsp._x_space_single_energy_proj[l] = x_proj
            self._ksp._cont_k_space_single_energy_proj[l] = k_proj
            self._new_ksp._new_k_space_single_energy_proj[l] = new_k_proj

        self.compute_expectation_values()
        self.combine_wavefunction_components()

    def add_state(self, the_states: list, the_energy_proj_coeffs: np.ndarray) -> None:
        if isinstance(the_states, int):
            the_states = [the_states]
            the_energy_proj_coeffs = [the_energy_proj_coeffs]

        print("adding state(s): ", the_states)

        self._esp._energy_proj_coeffs = (self._esp._energy_proj_coeffs*(self._esp._Norm)).tolist()
        
        for state in the_states:
            if  state not in self._sp.energy_states:
                self._sp.num_energy_states += 1
                self._esp._energy_proj_coeffs.append(the_energy_proj_coeffs[the_states.index(state)])

                self._sp.energy_states.append(state)
                k_kappa_to_append = self._sp.boundary_lib.get_kl(state)
                self._sp.k_kappa_l.append(k_kappa_to_append)

                energy_to_append = np.real(k_kappa_to_append**2)/(2*self._sp.m)
                self._esp._energies.append(energy_to_append)
                self._esp._wiggle_factors.append(Wiggle_Factor(energy_to_append))

                k_proj_append = self._sp.boundary_lib.get_k_space_projection(state)
                x_proj_append = self._sp.boundary_lib.get_x_space_projection(state)
                new_k_proj_append = self._sp.boundary_lib.get_new_k_space_projection(state)

                self._xsp._x_space_single_energy_proj.append(x_proj_append)
                self._ksp._cont_k_space_single_energy_proj.append(k_proj_append)
                self._new_ksp._new_k_space_single_energy_proj.append(new_k_proj_append)
        
        self._esp._energy_proj_coeffs = np.array(self._esp._energy_proj_coeffs)
        self._esp.normalize()

        self.compute_expectation_values()
        self.combine_wavefunction_components()

        print("current config: ", self._sp.energy_states)

    def remove_state(self, the_states: list) -> None:
        if isinstance(the_states, int):
            print("single state converted to list: ", the_states)
            the_states = [the_states]
        
        print("removing state(s): ", the_states)
        self._sp.num_energy_states -= len(the_states)

        for state in the_states:
            the_index = self._sp.l_kl_map.get_index(state)
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
        self.combine_wavefunction_components()

        print("current config: ", self._sp.energy_states)
        ##print("energy expectation value: ", self._esp._exp_value)

    def reset(self) -> None:
        self.remove_state(deepcopy(self._sp.energy_states))

    def compute_expectation_values(self) -> None:
        self._esp.compute_expectation_value()
        self._xsp.compute_expectation_value(self._esp)
        self._new_ksp.compute_expectation_value(self._esp)

    def combine_wavefunction_components(self) -> None:
        self._xsp.combine_wavefunction_components(self._esp)
        self._ksp.combine_wavefunction_components(self._esp)
        self._new_ksp.combine_wavefunction_components(self._esp)


    @property
    def x_space_wavefunction(self) -> Function_of_n_and_t:
        L = self._sp.L
        frame_function = Function_of_n(lambda x: np.heaviside(x+L/2, 0.5)*np.heaviside(L/2-x, 0.5))
        return self._xsp._x_space_wavefunction*frame_function

    @property
    def k_space_wavefunction(self) -> Function_of_n_and_t:
        return self._ksp._cont_k_space_wavefunction

    @property
    def new_k_space_wavefunction(self) -> Function_of_n_and_t:
        return self._new_ksp._new_k_space_wavefunction
    
    @property
    def x_space_expectation_value(self) -> Function_of_t:
        return self._xsp._expectation_value

    @property
    def new_k_space_expectation_value(self) -> Function_of_t:
        return self._new_ksp._expectation_value

    @property
    def x_space_expectation_value_derivative(self) -> Function_of_t:
        return self._xsp._exp_t_deriv
     
    @property
    def energy_expectation_value(self) -> float:
        return self._esp._exp_value

    @property
    def energy_space_wavefunction(self) -> Energy_Space_Wavefunction:
        return self._esp._energy_space_wavefunc
    
    @property
    def boundary_lib(self) -> New_Style_Boundary:
        return self._sp.boundary_lib

    @property
    def L(self) -> float:
        return self._sp.L

    @L.setter
    def L(self, new_L: float) -> None:
        print("setting L...(particle_in_a_box)")
        self._sp.L = new_L
        self.full_projection_recompute()

    @property
    def gamma(self) -> float:
        return self._sp.gamma

    @gamma.setter
    def gamma(self, new_gamma: float) -> None:
        self._sp.gamma = new_gamma
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
        self.combine_wavefunction_components()

    @property
    def case(self) -> str:
        return self._sp.case

    @case.setter
    def case(self, new_case: str) -> None:
        self._sp.case = new_case
        self.full_projection_recompute()
