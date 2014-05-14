/*
 * kernel_declatations.h
 *
 *  Created on: 14 апр. 2014 г.
 *      Author: pavel
 */

#ifndef KERNEL_DECLARATIONS_H_
#define KERNEL_DECLARATIONS_H_

namespace nnsim{
	// step size in ms
	float time_step;
	int Ncon;
	int Nneur;
	unsigned int Tsim;
	unsigned int t;

	// Neural variables and parameters
	float* Vms;
	float* Ums;
	float* Ies;
	float* as;
	float* bs;
	float* cs;
	float* ds;
	float* ks;
	float* Cms;
	float* Vrs;
	float* Vts;
	float* Vpeaks;
	float* Isyns;
	float* Erev_exc;
	float* Erev_inh;

	unsigned int* spk_times;
	unsigned int* neur_num_spks;
	unsigned int* syn_num_spks;

	// Synaptic parameters and variables

	// parameters for excitatory neurons
	float* y_exc;
	float* x_exc;
	float* u_exc;
	float* U_exc;
	float* tau_psc_exc;
	float* tau_rec_exc;
	float* tau_fac_exc;

	float* exp_psc_exc;
	float* exp_rec_exc;
	float* exp_fac_exc;
	float* exp_tau_exc;
	// initial amount of  neurotransmitter for excitatory neuron
	// it's equal to the convolution of input excitatory
	//connections weights the each neuron
	float* ina_exc;

	// parameters for inhibitory neurons
	float* y_inh;
	float* x_inh;
	float* u_inh;
	float* U_inh;
	float* tau_psc_inh;
	float* tau_rec_inh;
	float* tau_fac_inh;

	float* exp_psc_inh;
	float* exp_rec_inh;
	float* exp_fac_inh;
	float* exp_tau_inh;
	// initial amount of  neurotransmitter for inhibitory neuron
	// it's equal to the convolution of input inhibitory
	//connections weights the each neuron
	float* ina_inh;

	float* weights;
	int* delays;
	int* pre_conns;
	int* post_conns;
}

#endif /* KERNEL_DECLARATIONS_H_ */
