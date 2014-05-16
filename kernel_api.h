/*
 * kernel_api.h
 *
 *  Created on: 08 мая 2014 г.
 *      Author: pavel
 */

#ifndef KERNEL_API_H_
#define KERNEL_API_H_

namespace nnsim{
	void init_network(float h, int NumNeur, int NumConns, float SimTime, unsigned int time=0);

	void init_neurs(float* Cm_arr, float* Erev_exc_arr, float* Erev_inh_arr, float* Ie_arr, float* Isyn_arr,
			float* Um_arr, float* Vm_arr, float* Vpeak_arr, float* Vr_arr, float* Vt_arr,
			float* a_arr, float* b_arr, float* c_arr, float* d_arr, float* k_arr);

	void init_synapses(float* U_arr, float* tau_fac_arr, float* tau_psc_arr,
			float* tau_rec_arr, float* u_arr, float* x_arr, float* y_arr,
			float* weights_arr, int* delays_arr, int* pre_conns_arr, int* post_conns_arr, int* receptor_type_arr);

	void init_spikes(unsigned int* spike_times, unsigned int* neur_num_spikes, unsigned int* syn_num_spikes);

	int simulate();
}

#endif /* KERNEL_API_H_ */
