import tkinter
from tkinter.constants import S
import nummericalResources as nr
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np

root = tkinter.Tk()
root.wm_title("Embedding in Tk")

L = np.pi
gamma = 0.000001
l = 5

bound = 15
cStep = 0.1
dStep = 1

x = np.linspace(-L/2, L/2, 500)
k = np.arange(-bound, bound+cStep, cStep)*np.pi/L
kn = np.arange(-bound, bound+dStep, dStep)*np.pi/L

myPlot = nr.DumbPlot(gamma, L, l, k, kn, x)

canvas = FigureCanvasTkAgg(myPlot.fig2, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

button = tkinter.Button(master=root, text="Quit", command=_quit)
button.pack(side=tkinter.BOTTOM)

lvar = tkinter.IntVar(root)

def gamma_input(get):
    g = np.tan(float(get))/L
    if g == 0:
        g = 0.001
    myPlot.set_l(lvar.get())
    myPlot.set_gamma(g)
    myPlot.update()
    myPlot.updatePlot()
    canvas.draw()

slider = tkinter.Scale(root, from_=-np.pi/2+10**(-2), to=np.pi/2-10**(-2), orient="horizontal", length=600, resolution=0.001, command=gamma_input)
slider.set(0.001)
slider.pack()

def l_input(input):
    l = input
    g = np.tan(float(slider.get()))/L
    if g == 0:
        g = 0.001

    myPlot.set_gamma(g)
    myPlot.set_l(l)
    myPlot.update()
    myPlot.updatePlot()

    canvas.draw()

lselect = tkinter.OptionMenu(root, lvar, 1,2,3,4,5, command=l_input)
lvar.set(l)
lselect.pack()

tkinter.mainloop()
# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.