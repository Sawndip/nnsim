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
SimTime = 1000.

n_exc = create(400, n_type="exc", 
               Ie={'distr': 'normal', 'mean': 0., 'std': 0.},
                psn_rate=100.)

n_inh = create(100, n_type="inh")

con = connect(n_exc, n_inh+n_exc, conn_spec={'rule': 'fixed_total_num', 'N': 20000}, 
              delay={'distr': 'uniform', 'low': 0., 'high': 40.},
              x={'distr': 'uniform', 'low': 0., 'high': .5},
              weight={'distr': 'normal', 'mean': 16.0, 'std': 3.})

con2 = connect(n_inh, n_inh+n_exc, syn="inh", conn_spec={'rule': 'fixed_total_num', 'N': 5000}, 
              delay={'distr': 'uniform', 'low': 0., 'high': 40.},
              x={'distr': 'uniform', 'low': 0., 'high': .5},
              weight={'distr': 'normal', 'mean': 6.0, 'std': 3.})

init_recorder()

mean_rec(n_exc, 'neur', 'exc')
mean_rec(n_inh, 'neur', 'inh')

mean_rec(con, 'syn', 'syn_exc')
mean_rec(con2, 'syn', 'syn_inh')

simulate(h, SimTime)

spikes = get_spk_times()
(times, senders) = order_spikes(spikes)
pl.figure()
pl.plot(times, senders, '.k')

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
