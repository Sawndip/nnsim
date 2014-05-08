import numpy as np
import csv
import matplotlib.pylab as pl

f = open('oscill.csv', "r")
rdr = csv.reader(f, delimiter=";")
t = []
V_m_1 = []
V_m_2 = []

for l in rdr:
    t.append(l[0])
    V_m_1.append(l[1])
    V_m_2.append(l[2])

f.close()

pl.figure()
ax0 = pl.subplot(211)
ax1 = pl.subplot(212, sharex=ax0)

ax0.plot(t, V_m_1, 'b', label='1')
ax0.set_xlabel("Time, ms")

ax1.plot(t, V_m_2, 'r', label='2')

pl.show()
