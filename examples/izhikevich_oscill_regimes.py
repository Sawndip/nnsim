# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 16:59:32 2014

@author: pavel
"""

import nnsim
from nnsim import connect, create, get_results, simulate, get_spk_times, order_spikes, set_nparam, record
import matplotlib
import matplotlib.pyplot as pl
import numpy as np

h = .1
SimTime = 1000.

n_exc = create(3, n_type="exc", Ie=[35, 35, 55], d=[100, 50, 30])
set_nparam(n_exc[2], Cm=50., a=0.02, Vr=-50)
n_inh = create(1, n_type="inh", Ie=[75, 75], d=[40, 100])


#con = connect(n_exc, n_inh, 'all_to_all')
#con = connect(n_exc, n_inh, weight=10., conn_spec={"rule": 'fixed_outdegree', "N": 1}, 
#              delay={'distr': 'std', 'mean': 15., 'std': 5.}, syn='inh')

neur_rec = n_exc+n_inh
record(neur_rec)
print "Num conns", nnsim.NumConns

simulate(h, SimTime)
spikes = get_spk_times()

(Vm, Um, Isyn, y_exc, y_inh, x, u) = get_results()
t = np.linspace(0, SimTime, len(Vm[0]))
fig = pl.figure(figsize=(12, 9))
#fig = pl.figure(figsize=(24, 18))
matplotlib.rc('lines', linewidth=2.)
matplotlib.rc('font', size=26.)

ax = []
for i in range(len(neur_rec)):
    ax.append(pl.subplot(len(neur_rec), 1, i+1))
    ax[i].plot(t, Vm[i])
    ax[i].set_yticks([-40, 0, 40])
#    ax[i].set_ylabel("Vm_"+str(neur_rec[i]))
pl.xlabel("Time, ms")
pl.show()
pl.subplots_adjust(left = 0.15, bottom = 0.13, right = 0.92, top = 0.96, hspace = 0.33)
