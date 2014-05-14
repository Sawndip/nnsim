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
        
        self.neur_params_arr = {'a': [], 'b': [], 'c': [], 'd': [], 'k': [], 'Cm': [], 
                               'Vr': [], 'Vt': [], 'Vpeak': [], 'Vm': [], 'Um': [], 
                               'Erev_AMPA': [], 'Erev_GABBA': [], 'Isyn': [], 'Ie': []}

        self.exc_syn_param = {"tau_psc": 3., "tau_rec": 800., "tau_fac": 0.00001, 
                              "U": 0.5}
        
        self.inh_syn_param = {"tau_psc": 7., "tau_rec": 100., "tau_fac": 1000., 
                              "U": 0.04}
        self.NumNodes = 0
              
        
    def fill_neurs(self, N, params={}, default_params=None):
        if (default_params == None):
            self.exc_neur_param
        n_params = default_params.copy()
        for key, value in params.items():
            n_params[key] = value
            
        for key, value in n_params.items():    
                self.neur_params_arr[key].extend([value]*N)
        self.NumNodes += N
        return [i for i in xrange(self.NumNodes - N, self.NumNodes)]

    def new_exc_neurs(self, N, params={}):
        return self.fill_neurs(N, params=params, default_params=self.exc_neur_param)

    def new_inh_neurs(self, N, params={}):
        return self.fill_neurs(N, params=params, default_params=self.inh_neur_param)
        
    def init_neurs(self):
        nnsim_pykernel.init_neurs(tuple(self.neur_params_arr.values()))


print "  --NNSIM--  "

if __name__ == "__main__":
    nnsim = SpikingNNSimulator()
    n_exc = nnsim.new_exc_neurs(10)
    n_inh = nnsim.new_inh_neurs(10)
    print n_exc
    print n_inh
    nnsim.init_neurs()
