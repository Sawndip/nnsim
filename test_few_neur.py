# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 16:59:32 2014

@author: pavel
"""

from nnsim import connect, create, get_results, init_recorder, simulate, get_spk_times, order_spikes, mean_rec_from_neurs, mean_rec_from_conns
import matplotlib.pyplot as pl
import numpy as np

h = .1
SimTime = 20000.

n_exc = create(1, n_type="exc", psn_rate=120., psn_seed=0, psn_weight=1.)
n_inh = create(1, n_type="inh", psn_rate=10., psn_seed=1)

con = connect(n_exc, n_inh, 'all_to_all')

#neur_rec = n_exc+n_inh
#init_recorder(neur_rec, [])
init_recorder()

mean_rec_from_neurs(n_exc)

simulate(h, SimTime)

(Vm, Um, Isyn, x, y, u) = get_results(True)
t = np.linspace(0, SimTime, len(Vm[0]))
pl.figure()
ax = []
for i in range(len(neur_rec)):
    ax.append(pl.subplot(len(neur_rec), 1, i+1))
    ax[i].plot(t, Vm[i])
    ax[i].set_ylabel("Vm_"+str(neur_rec[i]))

pl.show()
