# -*- coding: utf-8
'''
Created on 13 мая 2014 г.

@author: pavel
'''

import nnsim_pykernel
import numpy as np
np.random.seed(seed=0)

MeanSpkPeriod = 5.

psn_tau = 3.

neur_param = {}

neur_param['exc'] = {'a': 0.02, 'b_1': 0.5, 'b_2': 0.5, 'c': -40., 'd': 100., 'k': 0.5, 'Cm': 50.,
                       'Vr': -60., 'Vt': -45., 'Vpeak': 40., 'p_1': 1., 'p_2': 1., 'Vm': -60., 'Um': 0.,
                       'Erev_AMPA': 0., 'Erev_GABA': -70., 'Isyn': 0., 'tau_psc_exc': 3., 'tau_psc_inh': 7., 'Ie': 0.,
                       'psn_seed': None, 'psn_rate': 0., 'psn_weight': 1.}

neur_param['inh'] = {'a': 0.03, 'b_1': -2.0, 'b_2': -2.0, 'c': -50., 'd': 100., 'k': 0.7, 'Cm': 100.,
                       'Vr': -60., 'Vt': -40., 'Vpeak': 35., 'p_1': 1., 'p_2': 1., 'Vm': -60., 'Um': 0.,
                       'Erev_AMPA': 0., 'Erev_GABA': -70., 'Isyn': 0., 'tau_psc_exc': 3., 'tau_psc_inh': 7., 'Ie': 0.,
                       'psn_seed': None, 'psn_rate': 0., 'psn_weight': 1.}

syn_param = {}

syn_param['exc'] = {'tau_rec': 800., 'tau_fac': 0.00001,
                      'U': 0.5, 'receptor_type': 1}

syn_param['inh'] = {'tau_rec': 100., 'tau_fac': 1000.,
                      'U': 0.04, 'receptor_type': 2}

syn_default = {'y': 0., 'x': 1., 'u': 0., 'weight': 1., 'delay': 0.}

neur_arr = {'a': [], 'b_1': [], 'b_2': [], 'c': [], 'd': [], 'k': [], 'Cm': [],
                       'Vr': [], 'Vt': [], 'Vpeak': [], 'p_1': [], 'p_2': [], 'Vm': [], 'Um': [],
                       'Erev_AMPA': [], 'Erev_GABA': [], 'Isyn': [], 'tau_psc_exc': [], 'tau_psc_inh': [], 'Ie': [],
                       'psn_seed': [], 'psn_rate': [], 'psn_weight': []}

syn_arr = {'tau_rec': [], 'tau_fac': [], 'U': [],
                    'y': [], 'x': [], 'u': [], 'weight': [], 'delay': [],
                    'pre': [], 'post': [], 'receptor_type': []}
rec_from_neur = []
rec_from_syn = []

NumNodes = 0

NumConns = 0

def init():
    global NumNodes, NumConns
    NumNodes, NumConns = 0, 0
    global neur_arr, syn_arr, rec_from_neur, rec_from_syn
    neur_arr = {'a': [], 'b_1': [], 'b_2': [], 'c': [], 'd': [], 'k': [], 'Cm': [],
                       'Vr': [], 'Vt': [], 'Vpeak': [], 'p_1': [], 'p_2': [], 'Vm': [], 'Um': [],
                       'Erev_AMPA': [], 'Erev_GABA': [], 'Isyn': [], 'tau_psc_exc': [], 'tau_psc_inh': [], 'Ie': [],
                       'psn_seed': [], 'psn_rate': [], 'psn_weight': []}

    syn_arr = {'tau_rec': [], 'tau_fac': [], 'U': [],
                    'y': [], 'x': [], 'u': [], 'weight': [], 'delay': [],
                    'pre': [], 'post': [], 'receptor_type': []}
    rec_from_neur = []
    rec_from_syn = []

def check_type(arg, ar_type=int):
    if type(arg) == list:
        for i in arg:
            if type(i) != ar_type:
                raise RuntimeError("Argument must be " + str(ar_type) + "or list of " + str(ar_type))
        return arg
    elif type(arg) == np.ndarray:
        if arg.dtype == np.int:
            return arg
    elif type(arg) != ar_type:
        raise RuntimeError("Argument must be " + str(ar_type) + "or list of " + str(ar_type))
    return [arg]

def create(N, n_type="exc", **kwargs):
    global neur_arr, NumNodes
    default_params=neur_param[n_type].copy()

    for key, value in kwargs.items():
        if type(value) in [list, tuple, np.ndarray]:
            neur_arr[key].extend(value[:N])
        elif type(value) not in [str, dict]:
            neur_arr[key].extend([value]*N)
        elif type(value) == dict:
            if value['distr'] == 'normal':
                std = value['std']
                mean = value['mean']
                if value.get('abs', True) == True:
                    neur_arr[key].extend(np.abs(mean + std*np.random.randn(N)))
                else:
                    neur_arr[key].extend(mean + std*np.random.randn(N))
            elif value['distr'] == 'uniform':
                low = value['low']
                high = value['high']
                neur_arr[key].extend(np.random.uniform(low, high, size=N))
            elif value['distr'] == 'gamma':
                shape = value['shape']
                scale = value['scale']
                loc = value['loc']
                neur_arr[key].extend(loc + np.random.gamma(shape, scale, size=N))
        else:
            raise RuntimeError("{0} must be a number or dict".format(key))
        default_params.pop(key)

    for key, value in default_params.items():
            neur_arr[key].extend([value]*N)
    NumNodes += N
    return [i for i in xrange(NumNodes - N, NumNodes)]

def set_nparam(n_idx, **kwargs):
    if type(n_idx) in [list, np.ndarray]:
        n_idx = n_idx[0]
    for key, value in kwargs.items():
        neur_arr[key][n_idx] = value

def connect(pre, post, conn_spec='one_to_one', syn='exc', **kwargs):
    global syn_arr, NumConns
    pre = check_type(pre)
    post = check_type(post)
    pre_ext = []
    post_ext = []
    syn_ext ={}
    if(conn_spec == 'one_to_one'):
        if (len(pre) != len(post)):
            raise RuntimeError("Lengths of pre and post must be equal")
        pre_ext = pre
        post_ext = post
    elif (conn_spec == 'all_to_all'):
        for i in pre:
            pre_ext.extend([i]*len(post))
            post_ext.extend(post)
    elif type(conn_spec) == dict:
        if conn_spec['rule'] == 'fixed_total_num':
            for i in xrange(conn_spec['N']):
                pre_ext.append(pre[np.random.randint(len(pre))])
                post_ext.append(post[np.random.randint(len(post))])
        if conn_spec['rule'] == 'fixed_outdegree':
            for i in pre:
                n_post = conn_spec['N']
                pre_ext.extend([i]*n_post)
                post_ext.extend(np.random.permutation(post)[:n_post])
        if conn_spec['rule'] == 'mean_outdegree':
            for i in pre:
                n_post = np.int(np.abs(conn_spec['N_mean'] + conn_spec['N_std']*np.random.randn()))
                pre_ext.extend([i]*n_post)
                post_ext.extend(np.random.permutation(post)[:n_post])
    else:
        raise RuntimeError("conn_spec must be one_to_one or all_to_all or dict")

    for key, value in syn_param[syn].items() + syn_default.items():
        syn_ext[key] = [value]*len(pre_ext)

    for key, value in kwargs.items():
        if type(value) not in [str, list, tuple, dict, np.ndarray]:
            syn_ext[key] = [value]*len(pre_ext)
        elif type(value) == dict:
            if value['distr'] == 'normal':
                std = value['std']
                mean = value['mean']
                if value.get('abs', True) == True:
                    syn_ext[key] = np.abs(mean + std*np.random.randn(len(pre_ext)))
                else:
                    syn_ext[key] = mean + std*np.random.randn(len(pre_ext))
            elif value['distr'] == 'uniform':
                low = value['low']
                high = value['high']
                syn_ext[key] = np.random.uniform(low, high, size=len(pre_ext))
            elif value['distr'] == 'gamma':
                shape = value['shape']
                scale = value['scale']
                loc = value['loc']
                syn_ext[key] = loc + np.random.gamma(shape, scale, size=len(pre_ext))
        else:
            raise RuntimeError("{0} must be a number or dict".format(key))
    syn_ext['pre'] = pre_ext
    syn_ext['post'] = post_ext
    for key, value in syn_ext.items():
        syn_arr[key].extend(value)

    NumConns += len(pre_ext)
    return [i for i in xrange(NumConns - len(pre_ext), NumConns)]

def record(nodes, node_type='neur'):
    global rec_from_neur, rec_from_syn
    if node_type == 'neur':
        rec_from_neur.extend(check_type(nodes))
    elif node_type == 'syn':
        rec_from_syn.extend(check_type(nodes))
    print rec_from_neur

pop_idx = {'neur': 0, 'syn': 0}
pop_nodes = {'neur': [], 'syn': []}
pop_names = {'neur': [], 'syn': []}

def mean_record(nodes, node_type='neur', name=None):
    pop_nodes[ntype].append(nodes)
    if name == None:
        name = pop_idx[ntype]
    pop_names[ntype].append(name)
    pop_idx[ntype] += 1
    pop_nodes[ntype].append(nodes)

def get_results(mean=False):
    if mean:
        num_neur_rec = pop_idx['neur']
        num_syn_rec = pop_idx['syn']
        mean = 1
    else:
        num_neur_rec = len(rec_from_neur)
        num_syn_rec = len(rec_from_syn)
        mean = 0
    (Vm_, Um_, Isyn_, y_exc_, y_inh_, x_, u_) = nnsim_pykernel.get_results(mean)
    Vm = []
    Um = []
    Isyn = []
    y_exc = []
    y_inh = []
    x = []
    u = []
    if len(Vm_) == 0:
        return (Vm, Um, Isyn, y_exc, y_inh, x, u)

    start = 0
    Tsim = len(Vm_)/num_neur_rec
    stop = Tsim
    for i in xrange(num_neur_rec):
        Vm.append(Vm_[start:stop])
        Um.append(Um_[start:stop])
        Isyn.append(Isyn_[start:stop])
        y_exc.append(y_exc_[start:stop])
        y_inh.append(y_inh_[start:stop])
        stop += Tsim
        start += Tsim

    if len(x_) == 0:
        return (Vm, Um, Isyn, y_exc, y_inh, x, u)

    start = 0
    Tsim = len(x_)/num_syn_rec
    stop = Tsim
    for i in xrange(num_syn_rec):
        x.append(x_[start:stop])
        u.append(u_[start:stop])
        stop += Tsim
        start += Tsim

    return (Vm, Um, Isyn, y_exc, y_inh, x, u)

def get_spk_times():
    global spk_times, n_spike
    (spk_times, n_spike) = nnsim_pykernel.get_spk_times()

    spikes = []
    for i in xrange(NumNodes):
        spikes.append([spk_times[NumNodes*sn + i]*tm_step for sn in xrange(n_spike[i])])
    return spikes

def get_ordered_spikes():
    return order_spikes(get_spk_times())

def order_spikes(spikes):
    times = []
    senders = []
    for i in xrange(NumNodes):
        times.extend(spikes[i])
        senders.extend([i]*len(spikes[i]))
    return (times, senders)

def simulate(h, SimTime, gpu=False):
    global tm_step
    tm_step = h
    nnsim_pykernel.init_network(h, NumNodes, NumConns, SimTime)
    psn_keys = ['psn_seed', 'psn_rate', 'psn_weight']

    args = {}
    for key, val in neur_arr.items():
        if key not in psn_keys:
            args[key] = np.array(val, dtype='float32')
    nnsim_pykernel.init_neurs(**args)

    psn_args = {}
    psn_args['psn_seed'] = np.array(np.random.randint(2147483647, size=NumNodes), dtype='uint32')
    for i in xrange(NumNodes):
        if neur_arr['psn_seed'][i] != None:
            psn_args['psn_seed'][i] = neur_arr['psn_seed'][i]

    psn_args['psn_rate'] = np.array(neur_arr['psn_rate'], dtype='float32')
    psn_args['psn_weight'] = np.array(neur_arr['psn_weight'], dtype='float32')
    psn_args['psn_tau'] = psn_tau
    nnsim_pykernel.init_poisson(**psn_args)

    args = {}
    for key, val in syn_arr.items():
        args[key] = np.array(val, dtype='float32')
    for key in ['pre', 'post', 'receptor_type']:
        args[key] = np.array(syn_arr[key], dtype='uint32')
    nnsim_pykernel.init_synapses(**args)

    args = {}
    args['sps_times'] = np.zeros(NumNodes*SimTime/MeanSpkPeriod, dtype='uint32')
    args['neur_num_spk'] = np.zeros(NumNodes, dtype='uint32')
    args['syn_num_spk'] = np.zeros(NumConns, dtype='uint32')
    nnsim_pykernel.init_spikes(**args)

    nnsim_pykernel.init_recorder(len(rec_from_neur), rec_from_neur,
                                 len(rec_from_syn), rec_from_syn)

    nnsim_pykernel.init_mean_recorder(pop_idx['neur'], pop_idx['syn'])
    for i in pop_nodes['neur']:
        nnsim_pykernel.add_neur_mean_record(np.array(i, dtype='uint32'))

    for i in pop_nodes['syn']:
        nnsim_pykernel.add_conn_mean_record(np.array(i, dtype='uint32'))
    gpu = [0, 1][gpu]
        #gpu = 1
    #else:
        #gpu = 0

    nnsim_pykernel.simulate(gpu)

print "  --NNSIM--  "
