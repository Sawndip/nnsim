# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 18:14:26 2014

@author: pavel
"""

from nnsim import connect, create, get_results, init_recorder, simulate, get_spk_times, order_spikes
import matplotlib.pyplot as pl
import numpy as np

h = .5
SimTime = 1000.

n_exc = create(400, n_type="exc", Ie={'mean': 35., 'std': 10.})
n_inh = create(100, n_type="inh")

con = connect(n_exc, n_inh+n_exc, conn_spec={'rule': 'fixed_total_num', 'N': 20000}, 
              weight={'mean': 6.0, 'std': 1.},
              delay={'mean': 30.0, 'std': 5.})

con2 = connect(n_inh, n_inh+n_exc, syn="inh", conn_spec={'rule': 'fixed_total_num', 'N': 5000}, 
              weight={'mean': 6.0, 'std': 1.},
              delay={'mean': 30.0, 'std': 5.})

# neur_rec = n_exc
# init_recorder(neur_rec, [])
init_recorder()

simulate(h, SimTime)

spikes = get_spk_times()
 
(times, senders) = order_spikes(spikes)
pl.plot(times, senders, '.k')
# 
# (Vm, Um, Isyn, x, y, u) = get_results()
# t = np.linspace(0, SimTime, len(Vm[0]))
# pl.figure()
# ax = []
# for i in range(len(neur_rec)):
#     ax.append(pl.subplot(len(neur_rec), 1, i+1))
#     ax[i].plot(t, Vm[i])
#     ax[i].set_ylabel("Vm_"+str(neur_rec[i]))

#ax0 = pl.subplot(211)
#ax1 = pl.subplot(212, sharex=ax0)
#ax0.plot(t, Vm[0])
#ax1.plot(t, Vm[1])

pl.show()
