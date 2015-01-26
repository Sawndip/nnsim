# -*- coding: utf-8 -*-
"""
Example of using NNSim for modeling short term placticity
"""
# Импорт различных библиотек
from nnsim import *
import matplotlib
import matplotlib.pyplot as pl
import numpy as np

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

ax = []
ax.append(pl.subplot(311))
ax.append(pl.subplot(312))
ax.append(pl.subplot(313))

ax[0].plot(t, Vm[0], 'r')
ax[1].plot(t, Vm[1])
ax[2].plot(t, Vm[2])

ax[1].set_ylabel("Membrane potential, mV", fontsize=18)
ax[0].set_title("Generator", fontsize=18)
ax[1].set_title("Depressive connection", fontsize=18)
ax[2].set_title("Facilitative connection", fontsize=18)
ax[1].set_xticks([])
ax[0].set_xticks([])

pl.xlabel("Time, ms")
pl.show()

