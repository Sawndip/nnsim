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
SimTime = 10000.
bin_sz = 5.
NumExc = 400
NumInh = 100
p_con = 0.5

n_exc = create(1400, n_type="exc", d={'distr': 'normal', 'mean': 70., 'std': 25.},
#               Ie={'distr': 'normal', 'mean': 10., 'std': 5., 'abs': True},
               psn_rate=140.
               )

n_pmkr = create(200, n_type='exc', 
                Ie={'distr': 'normal', 'mean': 35., 'std': 10.},
                psn_rate=0.
                )

n_exc = n_pmkr + n_exc

n_inh = create(400, n_type="inh")

N_exc_con = int(round(np.sqrt(p_con) * NumExc))
N_inh_con = int(round(np.sqrt(p_con) * NumInh))

pre = np.random.permutation(n_exc)[:N_exc_con]
#con = connect(pre, n_inh+n_exc, conn_spec={'rule': 'fixed_outdegree', 'N': N_exc_con},
#              delay={'distr': 'uniform', 'low': 0., 'high': 10.},
#              tau_rec=800.,
#              x={'distr': 'uniform', 'low': 0., 'high': .5},
#              weight={'distr': 'normal', 'mean': 5., 'std': 5., 'abs': True})
#
#pre = np.random.permutation(n_inh)[:N_inh_con]
#con2 = connect(pre, n_inh+n_exc, syn="inh", conn_spec={'rule': 'fixed_outdegree', 'N': N_inh_con},
#              delay={'distr': 'uniform', 'low': 0., 'high': 10.},
#              x={'distr': 'uniform', 'low': 0., 'high': .5},
#              weight={'distr': 'normal', 'mean': 6., 'std': 2., 'abs': True})

#init_recorder([])

#mean_rec(n_exc, 'neur', 'exc')
#mean_rec(n_inh, 'neur', 'inh')

#mean_rec(con, 'syn', 'syn_exc')
#mean_rec(con2, 'syn', 'syn_inh')

import time
start = time.time()
simulate(h, SimTime, gpu=True)
print "Elapsed %f s" % (time.time() - start)

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

#(Vm, Um, Isyn, y_exc, y_inh, x, u) = get_results()
#t = np.linspace(0, SimTime, len(Vm[0]))
#pl.figure()
#pl.plot(t, Vm[0])
#pl.show()

#(Vm, Um, Isyn, y_exc, y_inh, x, u) = get_results(True)
#t = np.linspace(0, SimTime, len(Vm[0]))
#name = 'syn'
##name = 'neur'
#pl.figure()
#ax = []
#for i in range(nnsim.pop_idx[name]):
#    ax.append(pl.subplot(nnsim.pop_idx[name], 1, i+1))
#    ax[i].plot(t, x[i])
#    ax[i].set_ylabel(nnsim.pop_names[name][i])
#
#pl.xlabel("Time, ms")

pl.show()
