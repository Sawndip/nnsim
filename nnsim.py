# -*- coding: utf-8
'''
Created on 13 мая 2014 г.

@author: pavel
'''

import nnsim_pykernel
import numpy as np
np.random.seed(seed=0)
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
        
        self.neur_arr = {'a': [], 'b': [], 'c': [], 'd': [], 'k': [], 'Cm': [], 
                               'Vr': [], 'Vt': [], 'Vpeak': [], 'Vm': [], 'Um': [], 
                               'Erev_AMPA': [], 'Erev_GABBA': [], 'Isyn': [], 'Ie': []}

        self.exc_syn_param = {'tau_psc': 3., 'tau_rec': 800., 'tau_fac': 0.00001, 
                              'U': 0.5}
        
        self.inh_syn_param = {'tau_psc': 7., 'tau_rec': 100., 'tau_fac': 1000., 
                              'U': 0.04}
        
        self.syn_arr = {'tau_psc': [], 'tau_rec': [], 'tau_fac': [], 'U': [], 
                            'y': [], 'x': [], 'u': [], 'weights': [], 'delays': [], 
                            'pre': [], 'post': [], 'receptor_type': []}
        
        self.NumNodes = 0
              
        self.Ncon = 0
        
    def fill_neurs(self, N, params={}, default_params=None, **kwargs):
        if (default_params == None):
            self.exc_neur_param
        n_params = default_params.copy()
                         
        for key, value in params.items():
            n_params[key] = value

        for key, value in kwargs.items():
            if (len(kwargs[key]) == N):
                self.neur_arr[key].extend(list(value))
                n_params.pop(key)
            
        for key, value in n_params.items():
                self.neur_arr[key].extend([value]*N)
        self.NumNodes += N
        return [i for i in xrange(self.NumNodes - N, self.NumNodes)]

    def new_exc_neurs(self, N, params={}, **kwargs):
        return self.fill_neurs(N, params=params, default_params=self.exc_neur_param, **kwargs)

    def new_inh_neurs(self, N, params={}, **kwargs):
        return self.fill_neurs(N, params=params, default_params=self.inh_neur_param, **kwargs)
    
    def connect(self, pre, post, weights, delays):
        if type(pre) == int:
            pre = [pre]
        if type(post) == int:
            post = [post]
        if type(weights) != list:
            weights = [weights]
        if type(delays) != list:
            delays = [delays]
            
        if (len(pre) != len(post)):
                raise RuntimeError("Lengths of pre and post must be equal")
        
        if (len(weights) != len(pre) or len(weights) != 1):
                raise RuntimeError("Lengths of weights must be 1 or equal to len of pre/post")

        if (len(delays) != len(pre) or len(delays) != 1):
                raise RuntimeError("Lengths of weights must be 1 or equal to len of pre/post")
        
        self.pre_conns.extend(pre)
        self.post_conns.extend(post)
        if (len(delays) == 1):
            self.delays.extend(delays*len(pre))
        else:
            self.delays.extend(delays)

        if (len(weights) == 1):
            self.weights.extend(weights*len(pre))
        else:
            self.weights.extend(weights)
        self.Ncon += len(pre)
    
        return [i for i in xrange(self.Ncon - len(pre), self.Ncon)]
    
    def simulate(self,h, SimTime):
        nnsim_pykernel.init_network(h, self.NumNodes, self.Ncon, SimTime)
        
        for key, val in self.neur_arr.items():
            self.neur_arr[key] = np.array(val, dtype='float32')
        nnsim_pykernel.init_neurs(**self.neur_arr)

print "  --NNSIM--  "

if __name__ == "__main__":
    nnsim = SpikingNNSimulator()
    
    n_exc = nnsim.new_exc_neurs(10)
    n_inh = nnsim.new_inh_neurs(10)
#     print n_exc
#     print n_inh
#     print nnsim.connect(0, 1, 10., 0.1)
    nnsim.simulate(0.1, 100.)
