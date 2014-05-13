# -*- coding: utf-8
'''
Created on 13 мая 2014 г.

@author: pavel
'''

import nnsim_pykernel
import numpy as np

h = 0.1
Nneur = 10000000
Ncon = 10
SimTime = 100.
for i in range(1):
    Vms = np.zeros(Nneur, dtype='float32')
    Ums = np.zeros(Nneur, dtype='float32')
    Ies = np.zeros(Nneur, dtype='float32')
    as_ = np.zeros(Nneur, dtype='float32')
    bs = np.zeros(Nneur, dtype='float32')
    cs = np.zeros(Nneur, dtype='float32')
    ds = np.zeros(Nneur, dtype='float32')
    ks = np.zeros(Nneur, dtype='float32')
    Cms = np.zeros(Nneur, dtype='float32')
    Vrs = np.zeros(Nneur, dtype='float32')
    Vts = np.zeros(Nneur, dtype='float32')
    Vpeaks = np.zeros(Nneur, dtype='float32')
    Isyns = np.zeros(Nneur, dtype='float32')
    Erev_exc = np.zeros(Nneur, dtype='float32')
    Erev_inh = np.zeros(Nneur, dtype='float32')
    
    for n in range(Nneur):
        Vms[n] = -60.0
        as_[n] = 0.02
        bs[n] = 0.5
        cs[n] = -40.0
        ds[n] = 100.0
        ks[n] = 0.5
        Cms[n] = 50.0
        Vrs[n] = -60.0
        Vts[n] = -45.0
        Vpeaks[n] = 35.0
        Erev_exc[n] = 0.0
        Erev_inh[n] = -70.0
    Ies[0] = 40.0
    
    nnsim_pykernel.init_network(h, Nneur, Ncon, SimTime) 
    nnsim_pykernel.init_neurs(Vms,  Ums,  Ies,  as_,
                bs,  cs,  ds,  ks,  Cms,
                Vrs,  Vts,  Vpeaks,  Isyns,
                Erev_exc,  Erev_inh)
