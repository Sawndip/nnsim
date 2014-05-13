# -*- coding: utf-8
'''
Created on 13 мая 2014 г.

@author: pavel
'''

import nnsim_pykernel
import numpy as np

h = 0.1
Nneur = 20
Ncon = 10
SimTime = 100.

class SpikingNNSimulator(object):
    '''
    Singltone class for simulator
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.exc_neur_param = {'a': 0.02, 'b': 0.5, 'c': -40., 'd': 100., 'k': 0.5, 'Cm': 50., 
                               'Vr': -60., 'Vt': -45., 'Vpeak': 40., 'Vm': -60., 'Um': 0., 
                               'Erev_AMPA': 0., 'Erev_GABBA': -70., 'Isyn': 0., 'Ie': 0.}

        self.inh_neur_param = {'a': 0.03, 'b': -2.0, 'c': -50., 'd': 100., 'k': 0.7, 'Cm': 100., 
                               'Vr': -60., 'Vt': -40., 'Vpeak': 35., 'Vm': -60., 'Um': 0., 
                               'Erev_AMPA': 0., 'Erev_GABBA': -70., 'Isyn': 0., 'Ie': 0.}
        
        self.exc_syn_param = {"tau_psc": 3., "tau_rec": 800., "tau_fac": 0.00001, 
                              "U": 0.5}
        
        self.inh_syn_param = {"tau_psc": 7., "tau_rec": 100., "tau_fac": 1000., 
                              "U": 0.04}
        
    def fill_neur_params(self, Nstart, Nstop, params={}):
        for i in xrange(Nstart, Nstop):
            self.as_[i] = params['a']
            self.bs[i] = params['b']
            self.cs[i] = params['c']
            self.ds[i] = params['d']
            self.ks[i] = params['k']
            self.Cms[i] = params['Cm']
            self.Vrs[i] = params['Vr']
            self.Vts[i] = params['Vt']
            self.Vpeaks[i] = params['Vpeak']
            self.Erev_exc[i] = params['Erev_AMPA']
            self.Erev_inh[i] = params['Erev_GABBA']
            self.Vms[i] = params['Vm']
            self.Ums[i] = params['Um']
            self.Isyns[i] = params['Isyn']
            self.Ies[i] = params['Ie']
        

    def init_network(self, h, Nneur, Ncon, SimTime):
        self.h = h
        self.Nneur = Nneur
        self.Ncon = Ncon
        self.SimTime = SimTime
        
        self.as_ = np.zeros(Nneur, dtype='float32')
        self.bs = np.zeros(Nneur, dtype='float32')
        self.cs = np.zeros(Nneur, dtype='float32')
        self.ds = np.zeros(Nneur, dtype='float32')
        self.ks = np.zeros(Nneur, dtype='float32')
        self.Cms = np.zeros(Nneur, dtype='float32')
        self.Vrs = np.zeros(Nneur, dtype='float32')
        self.Vts = np.zeros(Nneur, dtype='float32')
        self.Vpeaks = np.zeros(Nneur, dtype='float32')
        self.Erev_exc = np.zeros(Nneur, dtype='float32')
        self.Erev_inh = np.zeros(Nneur, dtype='float32')
        self.Vms = np.zeros(Nneur, dtype='float32')
        self.Ums = np.zeros(Nneur, dtype='float32')
        self.Isyns = np.zeros(Nneur, dtype='float32')
        self.Ies = np.zeros(Nneur, dtype='float32')
        
        
        nnsim_pykernel.init_network(h, Nneur, Ncon, SimTime)
    
    def new_exc_neurs(self, N, params={}):
        # must be called before new_inh_neurs
        self.Nexc = N
        n_params = self.exc_neur_param.copy()
        for key, value in params.items():
            n_params[key] = value
        self.fill_neur_params(0, self.Nexc, n_params)


    def new_inh_neurs(self, N, params={}):
        self.Ninh = N
        n_params = self.inh_neur_param.copy()
        for key, value in params.items():
            n_params[key] = value
        self.fill_neur_params(self.Nexc, self.Nexc + self.Ninh, n_params)
    
    def init_neurs(self):
        nnsim_pykernel.init_neurs(
                self.Vms,  self.Ums,  self.Ies,  self.as_,
                self.bs,  self.cs,  self.ds,  self.ks,  self.Cms,
                self.Vrs,  self.Vts,  self.Vpeaks,  self.Isyns,
                self.Erev_exc,  self.Erev_inh)
        
if __name__ == "__main__":
    nnsim = SpikingNNSimulator()
    nnsim.init_network(h, Nneur, Ncon, SimTime)
    nnsim.new_exc_neurs(10)
    nnsim.new_inh_neurs(10)
    nnsim.init_neurs()
