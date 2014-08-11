/*
 * kernel_api.h
 *
 *  Created on: 08 мая 2014 г.
 *      Author: pavel
 */

#ifndef KERNEL_API_H_
#define KERNEL_API_H_

namespace nnsim{

	typedef float myFloat;
	typedef unsigned int myUint;

	void init_network(float h, int NumNeur, int NumConns, float SimTime);

	void init_neurs(float* Cm_arr, float* Erev_exc_arr, float* Erev_inh_arr, float* Ie_arr, float* Isyn_arr,
			float* Um_arr, float* Vm_arr, float* Vpeak_arr, float* Vr_arr, float* Vt_arr,
			float* a_arr, float* b_arr, float* c_arr, float* d_arr, float* k_arr);

	void init_synapses(float* tau_rec_arr, float* tau_psc_arr, float* tau_fac_arr, float* U_arr,
			float* x_arr, float* y_arr, float* u_arr, float* weights_arr, float* delays_arr,
			myUint* pre_conns_arr, myUint* post_conns_arr, myUint* receptor_type_arr);

	void init_spikes(unsigned int* spike_times, unsigned int* neur_num_spikes, unsigned int* syn_num_spikes);

	int simulate();

	void init_recorder(unsigned int neur_num, unsigned int* neurs, unsigned int con_num, unsigned int* conns);

	void get_neur_results(myFloat* &Vm_res, myFloat* &Um_res, myFloat* &Isyn_res, unsigned int &N);

	void get_conn_results(myFloat* &x_res, myFloat* &y_res, myFloat* &u_res, unsigned int &N);

}

#endif /* KERNEL_API_H_ */
