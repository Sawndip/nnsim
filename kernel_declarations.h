/*
 * kernel_declatations.h
 *
 *  Created on: 14 апр. 2014 г.
 *      Author: pavel
 */

#ifndef KERNEL_DECLARATIONS_H_
#define KERNEL_DECLARATIONS_H_

namespace nnsim{
	int AMPA_RECEPTOR = 1;
	int GABBA_RECEPTOR = 2;
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
	float* AMPA_Amuont;
	float* GABBA_Amuont;
	float* Erev_exc;
	float* Erev_inh;

	unsigned int* spk_times;
	unsigned int* neur_num_spks;
	unsigned int* syn_num_spks;


	// Synaptic parameters and variables
	float* ys;
	float* xs;
	float* us;
	float* Us;
	float* tau_pscs;
	float* tau_recs;
	float* tau_facs;

	float* exp_pscs;
	float* exp_recs;
	float* exp_facs;
	float* exp_taus;

	float* weights;
	int* delays;
	int* pre_syns;
	int* post_syns;
	int* receptor_type;
}

#endif /* KERNEL_DECLARATIONS_H_ */
