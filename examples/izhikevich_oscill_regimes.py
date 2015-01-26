# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 16:59:32 2014

@author: pavel
"""

from nnsim import *
import matplotlib
import matplotlib.pyplot as pl
import numpy as np

h = .1
SimTime = 1000.

n_exc = create(3, n_type="exc", Ie=[35, 35, 55], d=[100, 50, 30])
set_nparam(n_exc[2], Cm=50., a=0.02, Vr=-50)
n_inh = create(1, n_type="inh", Ie=[75, 75], d=[40, 100])

neur_rec = n_exc+n_inh
record(neur_rec)

simulate(h, SimTime)
spikes = get_spk_times()

(Vm, Um, Isyn, y_exc, y_inh, x, u) = get_results()
t = np.linspace(0, SimTime, len(Vm[0]))

ax = []
for i in range(len(neur_rec)):
    ax.append(pl.subplot(len(neur_rec), 1, i+1))
    ax[i].plot(t, Vm[i])
    ax[i].set_yticks([-40, 0, 40])

pl.xlabel("Time, ms")
pl.show()
