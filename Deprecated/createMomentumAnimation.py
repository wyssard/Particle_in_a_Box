import nummericalResources as nr
import sympy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

plt.rcParams['animation.writer'] = 'ffmpeg'
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

res = 50
eps = 10**(-2)
gamma = np.tan(np.linspace(-np.pi/2+eps, 0, res))
gamma = np.concatenate((gamma, np.flip(gamma)))
L = np.pi

l = 2

bound = 15
cStep = 0.01
dStep = 1

k = np.arange(-bound, bound+cStep, cStep)*np.pi/L
kn = np.arange(-bound, bound+dStep, dStep)*np.pi/L

distr = nr.P_momentumContinuous(k, gamma, l, L)
#plt.plot(k, distr[0], c="r")

distrDisc = nr.getP_momentumDiscrete(kn, gamma, l, L)
#plt.bar(kn, distrDisc[0])

fig = plt.figure()
ax = plt.axes(xlim=(-bound, bound), ylim=(0, 0.25))
line = ax.plot([],[])


def init():
    line[0].set_data([],[])
    return line

def animate(i):
    distr = nr.P_momentumContinuous(k, gamma[i], l, L)
    line[0].set_data(k, distr)
    return line

anim = FuncAnimation(fig, animate, init_func=init, frames=2*res, interval=20)
anim.save("anotherTest.gif")

