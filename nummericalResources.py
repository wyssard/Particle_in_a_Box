import matplotlib.pyplot as plt
import numpy as np

# Determination of the momentum probability distribution/density for gamma_+ = gamma_- -> infinity
def P_momentum(n, l):
    if ((-1)**int(n) == (-1)**int(l) and abs(int(n))!=abs(int(l))):
        return 0
    elif (abs(int(n))==abs(int(l))):
        return 1/4
    elif ((-1)**int(n) == -(-1)**int(l)):
        return ((2*l)/(np.pi*(n**2-l**2)))**2

def P_momentumArray(n, l):
    P_return = []
    for element in n:
        P_return.append(P_momentum(element, l))
    return P_return

def P_momentumContinuousGammaInf(k, l, L):
    if (-1)**int(l)==1:
        return L/np.pi*np.power(np.sin(l*np.pi/2 + k*L/2)/(l*np.pi + k*L) - np.sin(l*np.pi/2 - k*L/2)/(l*np.pi - k*L), 2)
    elif (-1)**int(l)==-1:
        return L/np.pi*np.power(np.sin(l*np.pi/2 + k*L/2)/(l*np.pi + k*L) + np.sin(l*np.pi/2 - k*L/2)/(l*np.pi - k*L), 2)

# Correnspondance between gamma and k resp. kappa for gamma_+ = gamma_ =: gamma
# EXTREMELY IMPORTANT: the variables we are working with are g = arctan(gamma*L) and k*L instead of k
from scipy.optimize import fsolve

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
            #print("Odd Case")
        else:
            rel = posRelEven
            #print("Even Case")

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

# Determination of the momentum probability distribution/density for gamma_+ = gamma_- =: gamma
def P_momentumCountiuous_kl(k, kl, l, L):
    if l%2 == 0:
        return (L/np.pi)/(1-np.sin(kl)/kl)*np.power(np.sin(kl/2+k/2)/(kl+k)-np.sin(kl/2-k/2)/(kl-k),2)
    else:
        return (L/np.pi)/(1+np.sin(kl)/kl)*np.power(np.sin(kl/2+k/2)/(kl+k)+np.sin(kl/2-k/2)/(kl-k),2)

def P_momentumCountiuous_kappal(k, kappal, l, L):
    if l%2 == 0:
        return (L/np.pi)*4/(-1+np.sinh(kappal)/kappal)*np.power((k*np.cos(k/2)*np.sinh(kappal/2)-kappal*np.sin(k/2)*np.cosh(kappal/2))/(kappal**2 + np.power(k,2)) ,2)
    else:
        return (L/np.pi)*4/(1+np.sinh(kappal)/kappal)*np.power((k*np.sin(k/2)*np.cosh(kappal/2)-kappal*np.cos(k/2)*np.sinh(kappal/2))/(kappal**2 + np.power(k,2)) ,2)

def P_momentumContinuous(k, allkl, l, L):

    kappal = np.imag(allkl[np.real(allkl)==0])
    kl = np.real(allkl[np.imag(allkl)==0])

    distr = []
    for i in kappal:
        distr.append(P_momentumCountiuous_kappal(k*L, i*L, l, L))
    for i in kl:
        distr.append(P_momentumCountiuous_kl(k*L, i*L, l, L))

    return np.array(distr)

def getP_momentumDiscrete(kn, allkl, l, L):
    return P_momentumContinuous(kn, allkl, l, L)*np.pi/L

psi_l_Pos_odd = lambda L, kl, x: np.sqrt(2/L)*np.power(1+np.sin(kl*L)/(kl*L), -1/2)*np.cos(kl*x)
psi_l_Pos_even = lambda L, kl, x: np.sqrt(2/L)*np.power(1-np.sin(kl*L)/(kl*L), -1/2)*np.sin(kl*x)
psi_l_Neg_odd = lambda L, kappal, x: np.sqrt(2/L)*np.power(1+np.sinh(kappal*L)/(kappal*L), -1/2)*np.cosh(kappal*x)
psi_l_Neg_even = lambda L, kappal, x: np.sqrt(2/L)*np.power(-1+np.sinh(kappal*L)/(kappal*L), -1/2)*np.sinh(kappal*x)

def energyState(allkl, l, L, x):
    kappal = np.imag(allkl[np.real(allkl)==0])
    kl = np.real(allkl[np.imag(allkl)==0])

    states = []

    if l%2 == 0:
        psi_l_Pos = lambda kl: psi_l_Pos_even(L, kl, x)
        psi_l_Neg = lambda kappal: psi_l_Neg_even(L, kappal, x)
    else:
        psi_l_Pos = lambda kl: psi_l_Pos_odd(L, kl, x)
        psi_l_Neg = lambda kappal: psi_l_Neg_odd(L, kappal, x)
    
    for k in kappal:
        states.append(psi_l_Neg(k))
    for k in kl:
        states.append(psi_l_Pos(k))
    
    return np.array(states)

class DumbPlot:
    def update(self):
        self.allKl = gamma_to_k(self.gamma, self.l, self.L)
        # Computing the continuous momentum prob. density (old concept)
        self.distr = P_momentumContinuous(self.k, self.allKl, self.l, self.L)
        # Computing the discrete monmentum prob. dirst. (new concept)
        self.distrDisc = getP_momentumDiscrete(self.kn, self.allKl, self.l, self.L)
        # Computing the energy eigenstates corresponding to the determined kl values
        self.states = energyState(self.allKl, self.l, self.L, self.x)

        self.energy = np.real(np.power(gamma_to_k(self.gamma, self.l, self.L),2))*(self.L/np.pi)**2

    def __init__(self, gamma, L, l, k, kn, x):
        self.gamma = gamma
        self.L = L
        self.l = l
        self.k = k
        self.kn = kn
        self.x = x
        self.update()

        self.lightColor = "#8bb1cc"
        self.darkColor = "#0f4c75"

        plt.rcParams["text.usetex"] = False
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = 'STIXGeneral'
        plt.rcParams["mathtext.fontset"] = "cm"

        self.fig2 = plt.figure(constrained_layout=True, figsize=(7,7))
        gs = self.fig2.add_gridspec(nrows=2, ncols=2, height_ratios=[1.5,1])
        self.pDistrPlot = self.fig2.add_subplot(gs[0,:])
        self.gammaDepPlot = self.fig2.add_subplot(gs[1, 1])
        self.wavefuncPlot = self.fig2.add_subplot(gs[1, 0])

        # Plotting prob. distr.
        self.pDistrLine = self.pDistrPlot.plot(k, self.distr[0], c=self.darkColor, ls="--",label=r"Probalbility Density $\left\vert \langle k \vert l \rangle \right\vert^2$")
        self.pDistrBars = self.pDistrPlot.bar(kn, self.distrDisc[0], color=self.lightColor,label=r"Probability Distribution $\left\vert \langle n \vert l \rangle \right\vert^2$")
        self.pDistrPlot.set_xlabel("$k$")
        self.pDistrPlot.set_ylabel("Probability Distribution / Density")
        self.pDistrPlot.legend(loc="upper left")
        self.pDistrPlot.grid(True, which="major", axis="y", lw=0.5, c="0.8")
        self.pDistrPlot.set_ylim([0, 0.5])

        # Plotting energy dependance on gamma
        gammaRange = np.tan(np.linspace(-np.pi/2+10**(-2), np.pi/2-10**(-2), 300))
        self.gammaDepLines = []
        lArr = [1,2,3,4,5,6]
        for i in lArr:
            if i==l:
                col=self.darkColor
            else:
                col=self.lightColor
            self.gammaDepLines.append(self.gammaDepPlot.plot(np.arctan(gammaRange*L), np.real(np.power(gamma_to_k(gammaRange, i, L),2))*(L/np.pi)**2, color=col))
        self.gammaLine = self.gammaDepPlot.set_ylim([-4, 16])
        self.gVline = self.gammaDepPlot.axvline(np.arctan(gamma*L), ls="--", lw=1, color="0.5")
        self.gHline = self.gammaDepPlot.axhline(np.real(np.power(gamma_to_k(gamma, l, L),2))*(L/np.pi)**2, ls="--", lw=1, color="0.5")
        self.gammaDepPlot.set_ylabel(r"Energy in units $\frac{\pi^2}{2m L^2}$")
        self.gammaDepPlot.set_xlabel(r"$\arctan(\gamma L)$")
        self.gammaDepPlot.set_xticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
        self.gammaDepPlot.set_xticklabels([r"$-\pi/2$", r"$-\pi/4$", r"0", r"$-\pi/4$", r"$\pi/2$"])
        self.gammaDepPlot.set_yticks(np.arange(-4, 17, 1), minor=True)
        self.gammaDepPlot.set_yticks(np.power(np.arange(0,5,1),2))
        self.gammaDepPlot.grid(True, which="major", axis="y", lw=0.5, c="0.8")

        # Plotting the energy states
        self.wavefuncLine = self.wavefuncPlot.plot(x, self.states[0], color=self.darkColor)
        self.wavefuncPlot.set_xlabel("$x$", loc="right")
        self.wavefuncPlot.set_ylabel(r"$\langle x \vert l \rangle$", loc="top")
        self.wavefuncPlot.set_xticks([-L/2, -L/4, 0, L/4, L/2])
        self.wavefuncPlot.set_xticklabels([r"$-L/2$", r"$-L/4$", r"0", r"$-L/4$", r"$L/2$"])
        self.wavefuncPlot.spines["bottom"].set_position("center")
        self.wavefuncPlot.spines["left"].set_position("center")
        self.wavefuncPlot.spines["top"].set_color("none")
        self.wavefuncPlot.spines["right"].set_color("none")
        self.wavefuncPlot.tick_params(direction="inout")
        self.wavefuncPlot.set_ylim(-1, 1)

    def updatePlot(self):
        self.pDistrLine[0].set_data(self.k, self.distr[0])
        self.wavefuncLine[0].set_data(self.x, self.states[0])
        self.gVline.set_data([np.arctan(np.pi*self.gamma), np.arctan(np.pi*self.gamma)],[0, 1])
        self.gHline.set_data([0,1],[self.energy, self.energy])

        for rect, h in zip(self.pDistrBars, self.distrDisc[0]):
            rect.set_height(h)

        for i in range(1,6):
            if i==self.l:
                col=self.darkColor
            else:
                col=self.lightColor
            self.gammaDepLines[i-1][0].set_color(col)

    def display(self):
        plt.show()

    def get_densitiy(self):
        return self.distr

    def get_distr(self):
        return self.distrDisc

    def get_energy(self):
        return self.energy
    
    def get_states(self):
        return self.states

    def set_gamma(self, gamma):
        self.gamma = gamma
    
    def set_L(self, L):
        self.L = L

    def set_l(self, l):
        self.l = l
    
    def set_k(self, k):
        self.k = k

    def set_kn(self, kn):
        self.kn = kn
