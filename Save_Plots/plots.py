import nummericalResources as nr
import sympy as sp
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.gridspec as grdspc
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

lightColor = "#FBBD53"
darkColor = "#a16d15"

plt.rcParams["text.usetex"] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Dejavu Serif'
plt.rcParams["mathtext.fontset"] = "cm"



# Testing momentum distribution (general case with gamma_+ = gamm_- =: gamma)

L = np.pi
gamma = 1000000
l = 7


bound = 10
cStep = 0.001
dStep = 1
x = np.linspace(-L/2, L/2, 500)
k = np.arange(-bound-1, bound+1+cStep, cStep)*np.pi/L
kn = np.arange(-bound, bound+dStep, dStep)*np.pi/L
# Computing all kl values corresponding to gamma
allKl = nr.gamma_to_k(gamma, l, L)
# Computing the continuous momentum prob. density (old concept)
distr = nr.P_momentumContinuous(k, allKl, l, L)
# Computing the discrete monmentum prob. dirst. (new concept)
distrDisc = nr.getP_momentumDiscrete(kn, allKl, l, L)

fig = plt.figure(figsize=(7, 4.5), tight_layout=True)
pDistrPlot = plt.axes()

pDistrPlot.plot(k, distr[0], c="red", ls="-", lw=1.25)
pDistrPlot.bar(kn, distrDisc[0], zorder=2, color=lightColor, edgecolor=darkColor, linewidth=0.75,label=r"Probability Distribution $\left\vert \langle n \vert l \rangle \right\vert^2$")
pDistrPlot.set_xticks(kn)
pDistrPlot.set_xlim([-bound-0.75, bound+0.75])
pDistrPlot.yaxis.set_minor_locator(MultipleLocator(0.01))
pDistrPlot.grid(True, which="major", axis="y", lw=0.5, c="0.8", zorder=0)


plt.savefig("6th_exited_dirichlet.pdf")