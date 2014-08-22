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
	float* b1_s;
	float* b2_s;
	float* cs;
	float* ds;
	float* ks;
	float* Cms;
	float* Vrs;
	float* Vts;
	float* Vpeaks;
	float* p1_s;
	float* p2_s;
	float* Isyns;
	float* AMPA_Amuont;
	float* GABBA_Amuont;
	float* Erev_exc;
	float* Erev_inh;

	unsigned int* psn_times;
	unsigned int* psn_seeds;
	float* psn_rates;
	float* y_psns;
	float* exp_psns;
	float* psn_weights;

	unsigned int* spk_times;
	unsigned int* neur_num_spks;
	unsigned int* syn_num_spks;
	unsigned int len_spk_tms;

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

	unsigned int NumPopNeur = 0; // number of populations (neurons)
	unsigned int NumPopConn = 0; // number of populations (synapses)
	unsigned int CurrentPopNeur = 0;
	unsigned int CurrentPopConn = 0;
	unsigned int* PopNeurSizes; 	// number of neuron in each population
	unsigned int* PopConnSizes; 	// number of synapses in each population
	unsigned int** PopNeurs;	     	// neuron indices of each population (size = max(PopNeurSizes)*NumPopNeur)
	unsigned int** PopConns;	     	// connection indices of each population (size = max(PopConnSizes)*NumPopConn)

	float* Vm_means;
	float* Um_means;
	float* Isyn_means;

	float* x_means;
	float* y_means;
	float* u_means;
}

#endif /* KERNEL_DECLARATIONS_H_ */
