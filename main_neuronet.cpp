/*
 * main_neuronet.cpp
 *
 *  Created on: 07 мая 2014 г.
 *      Author: pavel
 */
#include "kernel_api.h"
#include <cstring>
#include "main_neuronet.h"
#include <cstdio>

int main(int argc, char* argv[]){
	h = 0.1f;
	Ncon = 1;
	Nneur = 2;
	SimTime = 200.0f;
	printf("Success starting\n");

	size_t spk_times_sz = (size_t) Nneur*SimTime/25.0f;
	spk_times = new unsigned int[spk_times_sz];
	neur_num_spks = new unsigned int[Nneur]();
	syn_num_spks = new unsigned int[Ncon]();

	fill_neur_params();
	fill_conn_params();
	fill_syn_params();

	nnsim::init_network(h, Nneur, Ncon, SimTime);
	nnsim::init_neurs(Vms, Ums, Ies, as, bs, cs, ds, ks, Cms, Vrs, Vts, Vpeaks, Isyns, Erev_exc, Erev_inh);
	nnsim::init_conns(weights,delays, pre_conns, post_conns);
	nnsim::init_exc_synapses(y_exc, x_exc, u_exc, U_exc, tau_psc_exc, tau_rec_exc, tau_fac_exc);
	nnsim::init_inh_synapses(y_inh, x_inh, u_inh, U_inh, tau_psc_inh, tau_rec_inh, tau_fac_inh);
	nnsim::init_spikes(spk_times, neur_num_spks, syn_num_spks);

	nnsim::simulate();
	return 0;
}

void fill_neur_params(){
	Vms = new float[Nneur];
	Ums = new float[Nneur]();
	Ies = new float[Nneur]();
	as = new float[Nneur];
	bs = new float[Nneur];
	cs = new float[Nneur];
	ds = new float[Nneur];
	ks = new float[Nneur];
	Cms = new float[Nneur];
	Vrs = new float[Nneur];
	Vts = new float[Nneur];
	Vpeaks = new float[Nneur];
	Isyns = new float[Nneur]();
	Erev_exc = new float[Nneur];
	Erev_inh = new float[Nneur];

	for (int n = 0; n < Nneur; n++){
		Vms[n] = -60.0f;
		as[n] = 0.02f;
		bs[n] = 0.5f;
		cs[n] = -40.0f;
		ds[n] = 100.0f;
		ks[n] = 0.5f;
		Cms[n] = 50.0f;
		Vrs[n] = -60.0f;
		Vts[n] = -45.0f;
		Vpeaks[n] = 35.0f;
		Erev_exc[n] = 0.0f;
		Erev_inh[n] = -70.0f;
	}
	Ies[0] = 40.0f;
}

void fill_conn_params(){
	weights = new float[Ncon];
	delays = new unsigned int[Ncon];
	pre_conns = new int[Ncon];
	post_conns = new int[Ncon];
	for (int c = 0; c < Ncon; c++){
		weights[c] = -10.0f;
		delays[c] = 5.0f/h;
	}
	pre_conns[0] = 0;
	post_conns[0] = 1;
}

void fill_syn_params(){
	y_exc = new float[Nneur]();
	x_exc = new float[Nneur];
	u_exc = new float[Nneur]();
	U_exc = new float[Nneur];
	tau_psc_exc = new float[Nneur];
	tau_rec_exc = new float[Nneur];
	tau_fac_exc = new float[Nneur];

	y_inh = new float[Nneur]();
	x_inh = new float[Nneur];
	u_inh = new float[Nneur]();
	U_inh = new float[Nneur];
	tau_psc_inh = new float[Nneur];
	tau_rec_inh = new float[Nneur];
	tau_fac_inh = new float[Nneur];

	for (int c = 0; c < Ncon; c++){
		if (weights[c] > 0.0f){
			x_exc[post_conns[c]] += weights[c];
		} else {
			x_inh[post_conns[c]] -= weights[c];
		}
	}

	for (int n = 0; n < Nneur; n++){
		U_exc[n] = 0.5f;
		tau_psc_exc[n] = 3.0f;
		tau_rec_exc[n] = 800.0f;
		tau_fac_exc[n] = 0.0001f;

		U_inh[n] = 0.04f;
		tau_psc_inh[n] = 3.0f;
		tau_rec_inh[n] = 100.0f;
		tau_fac_inh[n] = 1000.0f;
	}
}
