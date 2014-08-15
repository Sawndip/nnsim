# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 18:14:26 2014

@author: pavel
"""

from nnsim import connect, create, get_results, init_recorder, simulate, get_spk_times
import matplotlib.pyplot as pl
import numpy as np

h = .2
SimTime = 5000.

n_exc = create(1, n_type="exc", params={'Ie':33.})
n_inh = create(1, n_type="inh")

con = connect(n_exc, n_inh, conn_spec='all_to_all', weight=60.0, delay={'std':0., 'mean': 0.2})

neur_rec = n_inh+n_exc
init_recorder(neur_rec, [])

simulate(h, SimTime)

spikes = get_spk_times()

(Vm, Um, Isyn, x, y, u) = get_results()
t = np.linspace(0, SimTime, len(Vm[0]))

pl.figure()
ax = []
for i in range(len(neur_rec)):
    ax.append(pl.subplot(len(neur_rec), 1, i))
    ax[i].plot(t, Vm[i])
    ax[i].set_ylabel("Vm_"+str(neur_rec[i]))

#ax0 = pl.subplot(211)
#ax1 = pl.subplot(212, sharex=ax0)
#ax0.plot(t, Vm[0])
#ax1.plot(t, Vm[1])
