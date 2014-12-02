# -*- coding: utf-8 -*-
"""
Example of using NNSim for modeling short term placticity
"""
# Импорт различных библиотек
import nnsim
from nnsim import *
import matplotlib.pyplot as pl
import numpy as np
matplotlib.rc('lines', linewidth=1.5)
matplotlib.rc('font', size=26.)
h = .2 # шаг интегрирования
SimTime = 2000. # время симуляции в мс

# Нейрон генератор, от которого спайки будут идти к 2-м разным нейронам
gnrtr = create(1, n_type="exc", Ie=35., d=100.)
# тормозный и возбуждающий нейрон, устновлены только времена затухания
# постсинптических токов, остальные параметры по умолчанию
n_inh = create(1, n_type="inh", tau_psc_inh=15., tau_psc_exc=5.)
n_exc = create(1, n_type="exc", tau_psc_inh=15., tau_psc_exc=5.)

# соединение генератора с другим нейроном посредством тормозной связи
con1 = connect(gnrtr, n_inh, weight=10., conn_spec='one_to_one', syn='inh')
# соединение генератора с другим нейроном посредством возбуждающей связи
con2 = connect(gnrtr, n_exc, weight=7., conn_spec='one_to_one', syn='exc')
# задание списки нейронов с которых будет вестись запись осциллограммы
neur_rec = gnrtr + n_exc + n_inh
con_rec = con1 + con2
record(neur_rec)
record(con_rec, 'syn')

simulate(h, SimTime) # запуск симуляции без применения GPU

# получение осциллограмм и их рисование
(Vm, Um, Isyn, y_exc, y_inh, x, u) = get_results()
t = np.linspace(0, SimTime, len(Vm[0]))

fig = pl.figure(figsize=(16, 12))
#ax = []
#for i in range(len(con_rec)+1):
#    ax.append(pl.subplot(len(neur_rec), 1, i+1))
#    ax[i].plot(t, Vm[i])
##    ax[i].set_ylabel("Vm_"+str(neur_rec[i]) + ", mV")

ax = []
ax.append(pl.subplot(311))
ax.append(pl.subplot(312))
ax.append(pl.subplot(313))

ax[0].plot(t, Vm[0], 'r')
ax[1].plot(t, Vm[1])
ax[2].plot(t, Vm[2])

ax[0].set_yticks([-60., 0., 60])
ax[2].set_yticks([-60.5, -60.])
ax[1].set_yticks([-60., -45])

ax[1].set_ylabel("Membrane potential, mV", fontsize=19)
ax[0].set_title("Generator", fontsize=22)
ax[1].set_title("Depressive connection", fontsize=22)
ax[2].set_title("Facilitative connection", fontsize=22)
ax[1].set_xticks([])
ax[0].set_xticks([])

pl.xlabel("Time, ms")
pl.show()

#ax = []
#ax.append(pl.subplot(311))
#ax.append(pl.subplot(312))
#ax.append(pl.subplot(313))
#idx = 1
#ax[0].plot(t, x[idx], 'g')
#ax[1].plot(t, u[idx], 'k')
#ax[2].plot(t, y_exc[0], 'r')
#ax[0].set_yticks([0., 0.5, 1.])
#ax[1].set_yticks([0., 0.5, 1.])
#ax[2].set_yticks([0., 0.37, .75])
##ax[0].set_yticks([0.5, 0.75, 1.])
##ax[1].set_yticks([0., 0.1, .2])
##ax[2].set_yticks([0., 0.08, .16])
#
#ax[1].set_xticks([])
#ax[0].set_xticks([])
#
#ax[0].set_ylabel("x")
#ax[1].set_ylabel("u")
##ax[2].set_ylabel("y")
#pl.xlabel("Time, ms")
#pl.show()
