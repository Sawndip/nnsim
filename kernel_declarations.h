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

	myUint AMPA_RECEPTOR = 1;
	myUint GABBA_RECEPTOR = 2;
	// step size in ms
	float time_step;
	int Ncon;
	int Nneur;
	unsigned int Tsim;

	// Neural variables and parameters
	myFloat* Vms;
	myFloat* Ums;
	myFloat* Ies;
	myFloat* as;
	myFloat* bs;
	myFloat* cs;
	myFloat* ds;
	myFloat* ks;
	myFloat* Cms;
	myFloat* Vrs;
	myFloat* Vts;
	myFloat* Vpeaks;
	myFloat* Isyns;
	myFloat* AMPA_Amuont;
	myFloat* GABBA_Amuont;
	myFloat* Erev_exc;
	myFloat* Erev_inh;

	unsigned int* spk_times;
	unsigned int* neur_num_spks;
	unsigned int* syn_num_spks;


	// Synaptic parameters and variables
	myFloat* ys;
	myFloat* xs;
	myFloat* us;
	myFloat* Us;
	myFloat* tau_pscs;
	myFloat* tau_recs;
	myFloat* tau_facs;

	myFloat* exp_pscs;
	myFloat* exp_recs;
	myFloat* exp_facs;
	myFloat* exp_taus;

	myFloat* weights;
	myUint* delays;
	myUint* pre_syns;
	myUint* post_syns;
	myUint* receptor_type;


	unsigned int recorded_neur_num = 0;
	unsigned int* neurs_to_record;
	myFloat* Vm_recorded;
	myFloat* Um_recorded;
	myFloat* Isyn_recorded;

	unsigned int recorded_con_num = 0;
	unsigned int* conns_to_record;
	myFloat* x_recorded;
	myFloat* y_recorded;
	myFloat* u_recorded;
}

#endif /* KERNEL_DECLARATIONS_H_ */
