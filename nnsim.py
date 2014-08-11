# -*- coding: utf-8
'''
Created on 13 мая 2014 г.

@author: pavel
'''

import nnsim_pykernel
import numpy as np
import matplotlib.pyplot as pl
from numpy.core.setup import check_types
np.random.seed(seed=0)
h = .2
Nneur = 20
Ncon = 10
SimTime = 240.

class SpikingNNSimulator(object):
    '''
    Singltone class for simulator
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.exc_neur_param = {'a': 0.02, 'b': 0.5, 'c': -40., 'd': 100., 'k': 0.5, 'Cm': 50., 
                               'Vr': -60., 'Vt': -45., 'Vpeak': 40., 'Vm': -45., 'Um': 0., 
                               'Erev_AMPA': 0., 'Erev_GABBA': -70., 'Isyn': 0., 'Ie': 0.}

        self.inh_neur_param = {'a': 0.03, 'b': -2.0, 'c': -50., 'd': 100., 'k': 0.7, 'Cm': 100., 
                               'Vr': -60., 'Vt': -40., 'Vpeak': 35., 'Vm': -60., 'Um': 0., 
                               'Erev_AMPA': 0., 'Erev_GABBA': -70., 'Isyn': 0., 'Ie': 0.}
        

        self.exc_syn_param = {'tau_psc': 3., 'tau_rec': 800., 'tau_fac': 0.00001, 
                              'U': 0.5, 'receptor_type': 1}
        
        self.inh_syn_param = {'tau_psc': 7., 'tau_rec': 100., 'tau_fac': 1000., 
                              'U': 0.04, 'receptor_type': 2}
        
        self.syn_arr = {'tau_psc': [], 'tau_rec': [], 'tau_fac': [], 'U': [], 
                            'y': [], 'x': [], 'u': [], 'weights': [], 'delays': [], 
                            'pre': [], 'post': [], 'receptor_type': []}

        self.neur_arr = {'a': [], 'b': [], 'c': [], 'd': [], 'k': [], 'Cm': [], 
                               'Vr': [], 'Vt': [], 'Vpeak': [], 'Vm': [], 'Um': [], 
                               'Erev_AMPA': [], 'Erev_GABBA': [], 'Isyn': [], 'Ie': []}
        
        self.spikes_arr = {'sps_times': [], 'neur_num_spk': [], 'syn_num_spk': []}
        
        self.NumNodes = 0
              
        self.Ncon = 0
        
        self.exc_syn = 1;
        self.inh_syn = 2;
        
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
    
    def check_type(self, arg, ar_type=int):
        if type(arg) == list:
            for i in arg:
                if type(i) != ar_type:
                    raise RuntimeError("Argument must be " + str(ar_type) + "or list of " + str(ar_type))
            return arg
        elif type(arg) != ar_type:
            raise RuntimeError("Argument must be " + str(ar_type) + "or list of " + str(ar_type))
        return [arg]
            
    
    def connect(self, pre, post, weights=0., delays=0., syn=None, **kwargs):
        pre = self.check_type(pre)
        post = self.check_type(post)
        weights = self.check_type(weights, ar_type=float)
        delays = self.check_type(delays, ar_type=float)
#        print ((len(weights) != 1))
        if (len(pre) != len(post)):
                raise RuntimeError("Lengths of pre and post must be equal")
        
        if (len(weights) != len(pre) and len(weights) != 1):
                raise RuntimeError("Lengths of weights must be 1 or equal to len of pre/post")

        if (len(delays) != len(pre) and len(delays) != 1):
                raise RuntimeError("Lengths of weights must be 1 or equal to len of pre/post")
        
        self.syn_arr['pre'].extend(pre)
        self.syn_arr['post'].extend(post)
        if (len(delays) == 1):
            self.syn_arr['delays'].extend(delays*len(pre))
        else:
            self.syn_arr['delays'].extend(delays)

        if (len(weights) == 1):
            self.syn_arr['weights'].extend(weights*len(pre))
        else:
            self.syn_arr['weights'].extend(weights)
        if (syn == None):
            syn = self.exc_syn
        if (syn == self.exc_syn):
            for key, value in self.exc_syn_param.items():
                self.syn_arr[key] = [value]*len(pre)
        elif (syn == self.inh_syn):
            for key, value in self.inh_syn_param.items():
                self.syn_arr[key] = [value]*len(pre)
#        self.syn_arr['x'] = [1.]*len(pre)
        self.syn_arr['x'] = self.syn_arr['weights']
        self.syn_arr['y'] = [0.]*len(pre)
        self.syn_arr['u'] = [0.]*len(pre)
        
        self.Ncon += len(pre)
#        print self.syn_arr
        return [i for i in xrange(self.Ncon - len(pre), self.Ncon)]
    
    def simulate(self, h, SimTime):
        nnsim_pykernel.init_network(h, self.NumNodes, self.Ncon, SimTime)
        args = {}
        for key, val in self.neur_arr.items():
            args[key] = np.array(val, dtype='float32')
        nnsim_pykernel.init_neurs(**args)
        
        args = {}
        for key, val in self.syn_arr.items():
            args[key] = np.array(val, dtype='float32')
        for key in ['pre', 'post', 'receptor_type']:
            args[key] = np.array(self.syn_arr[key], dtype='uint32')
        nnsim_pykernel.init_synapses(**args)
        
        args = {}
        args['sps_times'] = np.zeros(self.NumNodes*SimTime/25., dtype='uint32')
        args['neur_num_spk'] = np.zeros(self.NumNodes, dtype='uint32')
        args['syn_num_spk'] = np.zeros(self.Ncon, dtype='uint32')
        nnsim_pykernel.init_spikes(**args)
        
        self.rec_from_neur = [0, 1]
        self.rec_from_syn = [0]
        nnsim_pykernel.init_recorder(len(self.rec_from_neur), self.rec_from_neur, 
                                     len(self.rec_from_syn), self.rec_from_syn)

        nnsim_pykernel.simulate()
        
            
print "  --NNSIM--  "

if __name__ == "__main__":
    nnsim = SpikingNNSimulator()
    
    n_exc = nnsim.new_exc_neurs(1, params={'Ie':40.})
    n_inh = nnsim.new_inh_neurs(1)

    nnsim.connect(n_exc, n_inh, 10.0, 10.)

    
    nnsim.simulate(h, SimTime)

    (Vm_, Um_, Isyn_, x_, y_, u_) = nnsim_pykernel.get_results()
    Vm = []
    Um = []
    Isyn = []
    start = 0
    Tsim = len(Vm_)/len(nnsim.rec_from_neur)
    stop = Tsim
    for i in range(len(nnsim.rec_from_neur)):
        Vm.append(Vm_[start:stop])
        Um.append(Um_[start:stop])
        Isyn.append(Isyn_[start:stop])
        stop += Tsim
        start += Tsim
    t = np.linspace(0, SimTime, Tsim)
#    pl.plot(t, Vm[0])
    pl.plot(Vm[0])
    pl.plot(Vm[1])

