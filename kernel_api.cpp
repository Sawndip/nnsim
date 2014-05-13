/*
 * kernel_api.cpp
 *
 *  Created on: 14 апр. 2014 г.
 *      Author: pavel
 */
#include "kernel_declarations.h"
#include <cmath>
#include <cstdio>

namespace nnsim{

float izhik_Vm(float Vm, float Um, float Isyn, int m){
	return (ks[m]*(Vm - Vrs[m])*(Vm - Vts[m]) - Um + Ies[m] + Isyn)*time_step/Cms[m];
}

float izhik_Um(float Vm, float Um, int m){
	return time_step*as[m]*(bs[m]*(Vm - Vrs[m]) - Um);
}

void init_network(float h, int NumNeur, int NumConns, float SimTime, unsigned int time=0){
	time_step = h;
	Ncon = NumConns;
	Nneur = NumNeur;
	Tsim = SimTime/time_step;
	t = time;
	printf("success initializing network!\n");
}

void init_neurs(float* Vm_arr, float* Um_arr, float* Ie_arr, float* a_arr,
		float* b_arr, float* c_arr, float* d_arr, float* k_arr, float* Cm_arr,
		float* Vr_arr, float* Vt_arr, float* Vpeak_arr, float* Isyn_arr,
		float* Erev_exc_arr, float* Erev_inh_arr){
	Vms = Vm_arr;
	Ums = Um_arr;
	Ies = Ie_arr;
	as = a_arr;
	bs = b_arr;
	cs = c_arr;
	ds = d_arr;
	ks = k_arr;
	Cms = Cm_arr;
	Vrs = Vr_arr;
	Vts = Vt_arr;
	Vpeaks = Vpeak_arr;
	Isyns = Isyn_arr;
	Erev_exc = Erev_exc_arr;
	Erev_inh = Erev_inh_arr;
	for (int i = 0; i < Nneur; i++){
		printf("%i: %g\n", i, as[i]);
	}
	printf("Success initializing neurons!\n");
}

void init_exc_synapses(float* y_exc_arr, float* x_exc_arr,
		float* u_exc_arr, float* U_exc_arr, float* tau_psc_exc_arr,
		float* tau_rec_exc_arr, float* tau_fac_exc_arr){
	y_exc = y_exc_arr;
	x_exc = x_exc_arr;
	u_exc = u_exc_arr;
	U_exc = U_exc_arr;
	tau_psc_exc = tau_psc_exc_arr;
	tau_rec_exc = tau_rec_exc_arr;
	tau_fac_exc = tau_fac_exc_arr;

	exp_psc_exc = new float[Nneur];
	exp_rec_exc = new float[Nneur];
	exp_fac_exc = new float[Nneur];
	exp_tau_exc = new float[Nneur];
	ina_exc = new float[Nneur];
	for (int n = 0; n < Nneur; n++){
		exp_psc_exc[n] = expf(-time_step/tau_psc_exc[n]);
//		printf("%f\n", exp_psc_exc[n]);
		exp_rec_exc[n] = expf(-time_step/tau_rec_exc[n]);
		exp_fac_exc[n] = expf(-time_step/tau_fac_exc[n]);
		exp_tau_exc[n] = (tau_psc_exc[n]*exp_psc_exc[n] - tau_rec_exc[n]*exp_rec_exc[n])/(tau_rec_exc[n] - tau_psc_exc[n]);
		ina_exc[n] = x_exc[n];
	}
	printf("Success initializing excitatory synapses!\n");
}

void init_inh_synapses(float* y_inh_arr, float* x_inh_arr,
		float* u_inh_arr, float* U_inh_arr, float* tau_psc_inh_arr,
		float* tau_rec_inh_arr, float* tau_fac_inh_arr){
	y_inh = y_inh_arr;
	x_inh = x_inh_arr;
	u_inh = u_inh_arr;
	U_inh = U_inh_arr;
	tau_psc_inh = tau_psc_inh_arr;
	tau_rec_inh = tau_rec_inh_arr;
	tau_fac_inh = tau_fac_inh_arr;

	exp_psc_inh = new float[Nneur];
	exp_rec_inh = new float[Nneur];
	exp_fac_inh = new float[Nneur];
	exp_tau_inh = new float[Nneur];
	ina_inh = new float[Nneur];
	for (int n = 0; n < Nneur; n++){
		printf("x_exc: %f\n", x_exc[n]);
		exp_psc_inh[n] = expf(-time_step/tau_psc_inh[n]);
		exp_rec_inh[n] = expf(-time_step/tau_rec_inh[n]);
		exp_fac_inh[n] = expf(-time_step/tau_fac_inh[n]);
		exp_tau_inh[n] = (tau_psc_inh[n]*exp_psc_inh[n] - tau_rec_inh[n]*exp_rec_inh[n])/(tau_rec_inh[n] - tau_psc_inh[n]);
		ina_inh[n] = x_inh[n];
	}
	printf("Success initializing inhibitory synapses!\n");
}

void init_conns(float* weights_arr, unsigned int* delays_arr, int* pre_conns_arr, int* post_conns_arr){
	weights = weights_arr;
	delays = delays_arr;
	pre_conns = pre_conns_arr;
	post_conns = post_conns_arr;
	printf("Success initializing connections!\n");
}

void init_spikes(unsigned int* spike_times, unsigned int* neur_num_spikes, unsigned int* syn_num_spikes){
	spk_times = spike_times;
	neur_num_spks = neur_num_spikes;
	syn_num_spks = syn_num_spikes;
}

int simulate(){
	float v1, u1, v2, u2, v3, u3, v4, u4;
	FILE* res_file;
	res_file = fopen("oscill.csv", "w");
	for (t = 1; t < Tsim; t++){
		fprintf(res_file, "%f;%f;%f\n", t*time_step, Vms[0], Vms[1]);
		for (int c = 0; c < Ncon; c++){
			if (syn_num_spks[c] < neur_num_spks[pre_conns[c]]){
//				printf("Where are unprocessed spikes at %i time: %i!\n", spk_times[Nneur*syn_num_spks[c] + pre_conns[c]], t);
				if (spk_times[Nneur*syn_num_spks[c] + pre_conns[c]] == t - delays[c]){
					int n = post_conns[c];
					if (weights[c] > 0.0f){
						u_exc[n] += U_exc[n]*(1.0f - u_exc[n]);
						float dx = x_exc[n]*u_exc[n]*weights[c]/ina_exc[n];
						y_exc[n] += dx;
						x_exc[n] -= dx;
					} else {
						u_inh[n] += U_inh[n]*(1.0f - u_inh[n]);
						float dx = x_inh[n]*u_inh[n]*weights[c]/ina_inh[n];
						y_inh[n] -= dx;
						x_inh[n] += dx;
					}
					syn_num_spks[c]++;
				}
			}
		}

		for (int n = 0; n < Nneur; n++){

			x_exc[n] = y_exc[n]*exp_tau_exc[n] - (ina_exc[n] - x_exc[n] - y_exc[n])*exp_rec_exc[n] + ina_exc[n];
			y_exc[n] = y_exc[n]*exp_psc_exc[n];
			u_exc[n] = u_exc[n]*exp_fac_exc[n];

			x_inh[n] = y_inh[n]*exp_tau_inh[n] - (ina_inh[n] - x_inh[n] - y_inh[n])*exp_rec_inh[n] + ina_inh[n];
			y_inh[n] = y_inh[n]*exp_psc_inh[n];
			u_inh[n] = u_inh[n]*exp_fac_inh[n];


			float Isyn_new = -y_exc[n]*(Vms[n] - Erev_exc[n]) - y_inh[n]*(Vms[n] - Erev_inh[n]);
			float Vm = Vms[n];
			float Um = Ums[n];
			if (Vm > Vpeaks[n]){
				spk_times[Nneur*neur_num_spks[n] + n] = t;
				neur_num_spks[n]++;

				Vms[n] = cs[n];
				Ums[n] = Um + ds[n];
			}else{
				v1 = izhik_Vm(Vm, Um, Isyns[n], n);
				u1 = izhik_Um(Vm, Um, n);
				Vms[n] = Vm + v1*0.5f;
				Ums[n] = Um + u1*0.5f;
				v2 = izhik_Vm(Vms[n], Ums[n], (Isyn_new + Isyns[n])*0.5f, n);
				u2 = izhik_Um(Vms[n], Ums[n], n);
				Vms[n] = Vm + v2*0.5f;
				Ums[n] = Um + u2*0.5f;
				v3 = izhik_Vm(Vms[n], Ums[n], (Isyn_new + Isyns[n])*0.5f, n);
				u3 = izhik_Um(Vms[n], Ums[n], n);
				Vms[n] = Vm + v3;
				Ums[n] = Um + u3;
				v4 = izhik_Vm(Vms[n], Ums[n], Isyns[n], n);
				u4 = izhik_Um(Vms[n], Ums[n], n);
				Vms[n] = Vm + (v1 + 2.0f*(v2 + v3) + v4)*0.16666666f;
				Ums[n] = Um + (u1 + 2.0f*(u2 + u3) + u4)*0.16666666f;
			}
			Isyns[n] = Isyn_new;
		}
	}
	fclose(res_file);
	return 0;
}

}
