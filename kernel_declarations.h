/*
 * kernel_declatations.h
 *
 *  Created on: 14 апр. 2014 г.
 *      Author: pavel
 */

#ifndef KERNEL_DECLARATIONS_H_
#define KERNEL_DECLARATIONS_H_

#include "kernel_api.h"
#include "nnsim_constants.h"

namespace nnsim{

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
	float* Erev_exc;
	float* Erev_inh;
	float* AMPA_Amuont;
	float* GABA_Amuont;
	float* exp_pscs_exc;
	float* exp_pscs_inh;

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
	float* xs;
	float* us;
	float* Us;
	float* exp_recs;
	float* exp_facs;

	float* weights;
	unsigned int* delays;
	unsigned int* pre_syns;
	unsigned int* post_syns;
	unsigned int* receptor_type;

	// Variables for recording
	unsigned int recorded_neur_num = 0;
	unsigned int* neurs_to_record;
	float* Vm_recorded;
	float* Um_recorded;
	float* Isyn_recorded;
	float* y_exc_recorded;
	float* y_inh_recorded;

	unsigned int recorded_con_num = 0;
	unsigned int* conns_to_record;
	float* x_recorded;
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
	float* y_exc_means;
	float* y_inh_means;

	float* x_means;
	float* u_means;
}

#endif /* KERNEL_DECLARATIONS_H_ */
