# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 18:14:26 2014

@author: pavel
"""

from nnsim import connect, create, get_results, init_recorder, simulate, get_spk_times, order_spikes, mean_rec
import nnsim 
import matplotlib.pyplot as pl
import numpy as np

h = .5
SimTime = 16000.
bin_sz = 5.

n_exc = create(400, n_type="exc", 
               Ie={'distr': 'normal', 'mean': 0., 'std': 0.},
                psn_rate=50.)

n_inh = create(100, n_type="inh")

con = connect(n_exc, n_inh+n_exc, conn_spec={'rule': 'fixed_total_num', 'N': 20000}, 
              delay={'distr': 'uniform', 'low': 0., 'high': 40.},
              x={'distr': 'uniform', 'low': 0., 'high': .5},
              weight={'distr': 'uniform', 'low': .0, 'high': 6.})

con2 = connect(n_inh, n_inh+n_exc, syn="inh", conn_spec={'rule': 'fixed_total_num', 'N': 5000}, 
              delay={'distr': 'uniform', 'low': 0., 'high': 40.},
              x={'distr': 'uniform', 'low': 0., 'high': .5},
              weight={'distr': 'uniform', 'low': .0, 'high': 6.})

init_recorder()

mean_rec(n_exc, 'neur', 'exc')
mean_rec(n_inh, 'neur', 'inh')

mean_rec(con, 'syn', 'syn_exc')
mean_rec(con2, 'syn', 'syn_inh')

import time
start = time.time()
simulate(h, SimTime)
print "Elapsed % s", time.time() - start

spikes = get_spk_times()
(times, senders) = order_spikes(spikes)
pl.figure()
ax1 = pl.subplot(211)
ax2 = pl.subplot(212, sharex=ax1)
ax1.plot(times, senders, '.k')
ax1.set_ylabel("Neuron #")
ahis = np.histogram(times, range=(0, SimTime), bins=int(SimTime/bin_sz))
ax2.plot(ahis[1][:-1], ahis[0])
ax2.set_ylabel("num spike in {0} ms".format(bin_sz))
ax2.set_xlabel("Time, ms")

(Vm, Um, Isyn, x, y, u) = get_results(True)
t = np.linspace(0, SimTime, len(Vm[0]))
name = 'syn'
#name = 'neur'
pl.figure()
ax = []
for i in range(nnsim.pop_idx[name]):
    ax.append(pl.subplot(nnsim.pop_idx[name], 1, i+1))
    ax[i].plot(t, x[i])
    ax[i].set_ylabel(nnsim.pop_names[name][i])

pl.xlabel("Time, ms")
pl.show()
