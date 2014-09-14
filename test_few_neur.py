# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 16:59:32 2014

@author: pavel
"""

import nnsim
from nnsim import connect, create, get_results, init_recorder, simulate, get_spk_times, order_spikes
import matplotlib.pyplot as pl
import numpy as np

h = .1
SimTime = 2000.

n_exc = create(10, n_type="exc", psn_rate=0., psn_seed=0, Ie=35, d={'distr': 'normal', 'mean': 70., 'std': 25.},)
n_inh = create(1, n_type="inh", psn_rate=0., psn_seed=1)

con = connect(n_exc, n_inh, 'all_to_all')
# con = connect(n_exc, n_inh, conn_spec={"rule": 'mean_outdegree', "N_mean": 2, "N_std": 1})

neur_rec = n_exc+n_inh
syn_rec = con
init_recorder(neur_rec, syn_rec)

simulate(h, SimTime)

(Vm, Um, Isyn, x, y, u) = get_results()
t = np.linspace(0, SimTime, len(Vm[0]))
pl.figure()
ax = []
for i in range(len(neur_rec)):
    ax.append(pl.subplot(len(neur_rec), 1, i+1))
    ax[i].plot(t, Vm[i])
    ax[i].set_ylabel("Vm_"+str(neur_rec[i]))

pl.show()
