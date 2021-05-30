from numpy import pi
import particle_in_a_box as pib


state = pib.Particle_in_Box_State("symmetric", pi, 1, [2,3], [1,1], 0.0001)

state.gamma = 1

