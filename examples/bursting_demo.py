# -*- coding: utf-8 -*-
"""
Script for simulting large scale network with using GPU for
performing calculations
"""

from nnsim import *
import nnsim
import matplotlib.pyplot as pl
import matplotlib
import numpy as np
matplotlib.rc('lines', linewidth=1.5)
matplotlib.rc('font', size=26.)
h 	= .5
SimTime = 1000.
bin_sz  = 5.0	# размер временног окна в котором будет считаться кол-во спайков
NumExc  = 1400  # кол-во возбуждающих нейронов
NumPmkr = 100   # кол-во генераторов ритма (pacemaker)
NumInh  = 400   # кол-во тормозных нейронов

# Параметры для большой сети
#NumExc  = 7000  # кол-во возбуждающих нейронов
#NumPmkr = 500   # кол-во генераторов ритма (pacemaker)
#NumInh  = 2000   # кол-во тормозных нейронов
p_con   = 0.025 # вероятность 2-х нейронов быть связанными

for i in range(1, 3):
    init()
    # создние возбуждающей популяции, генераторов ритма (pmkr) и тормозных нейронов
    n_exc = create(NumExc, n_type="exc", d={'distr': 'normal', 'mean': 70., 'std': 25.})
    n_pmkr = create(NumPmkr*i, n_type='exc', Ie={'distr': 'normal', 'mean': 35., 'std': 10.})
    n_inh = create(NumInh, n_type="inh")

    # расчёт средндего количества исходящих связей на каждый нейрон
    # заднной популяции исходя из вероятности быть связанными
    N_exc_con = int(round(np.sqrt(p_con) * NumExc + NumPmkr))
    N_inh_con = int(round(np.sqrt(p_con) * NumInh))

    # выбор из популяции возбуждающих нейронов N_exc_con случайных нейронов
    # каждый из которых будет связан с N_exc_con других нейронов из всех нейронов
    pre = np.random.permutation(n_exc + n_pmkr)[:N_exc_con]
    con = connect(pre, n_inh+n_exc+n_pmkr, conn_spec={'rule': 'fixed_outdegree', 'N': N_exc_con},
                  delay={'distr': 'uniform', 'low': 0., 'high': 10.},
                  tau_rec=800.,
                  x={'distr': 'uniform', 'low': 0., 'high': .5},
                  weight={'distr': 'normal', 'mean': 7., 'std': 4.})
# Параметры для большой сети
                  weight={'distr': 'normal', 'mean': 2., 'std': 5.})
    # тоже самое для тормозной популяции
    pre = np.random.permutation(n_inh)[:N_inh_con]
    con2 = connect(pre, n_inh+n_exc+n_pmkr, syn="inh", conn_spec={'rule': 'fixed_outdegree', 'N': N_inh_con},
                  delay={'distr': 'uniform', 'low': 0., 'high': 10.},
                  x={'distr': 'uniform', 'low': 0., 'high': .5},
                  weight={'distr': 'normal', 'mean': 6., 'std': 2.})
# Параметры для большой сети
                  weight={'distr': 'normal', 'mean': 1.2, 'std': .4})
    # замер времени симуляции
    import time
    start = time.time()
    simulate(h, SimTime, gpu=False)
    print "Elapsed %f s" % (time.time() - start)

# визуализация полученных данных
(times, senders) = get_ordered_spikes()
fig = pl.figure(figsize=(12, 9))
ax1 = pl.subplot(211)
ax2 = pl.subplot(212, sharex=ax1)
ax1.set_ylim([0, max(senders)])
ax1.plot(times, senders, '.k')
ax1.set_ylabel("Neuron #")
ax2.hist(times, range=(0, SimTime), bins=int(SimTime/bin_sz), histtype='step')
ax2.set_ylabel("Num spikes in {0} ms".format(bin_sz))
ax2.set_xlabel("Time, ms")

pl.show()
