/*
 * kernel_declatations.h
 *
 *  Created on: 14 апр. 2014 г.
 *      Author: pavel
 */

#ifndef KERNEL_DECLARATIONS_H_
#define KERNEL_DECLARATIONS_H_

#include "kernel_api.h"

namespace nnsim{

	unsigned int AMPA_RECEPTOR = 1;
	unsigned int GABBA_RECEPTOR = 2;
	// step size in ms
	float time_step;
	int Ncon;
	int Nneur;
	unsigned int Tsim;

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
	unsigned int* delays;
	unsigned int* pre_syns;
	unsigned int* post_syns;
	unsigned int* receptor_type;


	unsigned int recorded_neur_num = 0;
	unsigned int* neurs_to_record;
	float* Vm_recorded;
	float* Um_recorded;
	float* Isyn_recorded;

	unsigned int recorded_con_num = 0;
	unsigned int* conns_to_record;
	float* x_recorded;
	float* y_recorded;
	float* u_recorded;
}

#endif /* KERNEL_DECLARATIONS_H_ */
