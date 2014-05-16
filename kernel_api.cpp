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

void init_neurs(float* Cm_arr, float* Erev_exc_arr, float* Erev_inh_arr, float* Ie_arr, float* Isyn_arr,
		float* Um_arr, float* Vm_arr, float* Vpeak_arr, float* Vr_arr, float* Vt_arr,
		float* a_arr, float* b_arr, float* c_arr, float* d_arr, float* k_arr){
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
	AMPA_Amuont = new float[Nneur];
	GABBA_Amuont = new float[Nneur];
	for (int i = 0; i < Nneur; i++){
		printf("%i: %g\n", i, ks[i]);
	}
	printf("Success initializing neurons!\n");
}

void init_synapses(float* U_arr, float* tau_fac_arr, float* tau_psc_arr,
		float* tau_rec_arr, float* u_arr, float* x_arr, float* y_arr,
		float* weights_arr, int* delays_arr, int* pre_conns_arr, int* post_conns_arr, int* receptor_type_arr){
	ys = y_arr;
	xs = x_arr;
	us = u_arr;
	Us = U_arr;
	weights = weights_arr;
	delays = delays_arr;
	pre_syns = pre_conns_arr;
	post_syns = post_conns_arr;
	receptor_type = receptor_type_arr;
	tau_pscs = tau_psc_arr;
	tau_recs = tau_rec_arr;
	tau_facs = tau_fac_arr;

	exp_pscs = new float[Nneur];
	exp_recs = new float[Nneur];
	exp_facs = new float[Nneur];
	exp_taus = new float[Nneur];

	for (int c = 0; c < Ncon; c++){
		exp_pscs[c] = expf(-time_step/tau_psc_arr[c]);
		exp_recs[c] = expf(-time_step/tau_rec_arr[c]);
		exp_facs[c] = expf(-time_step/tau_fac_arr[c]);
		exp_taus[c] = (tau_psc_arr[c]*exp_pscs[c] - tau_rec_arr[c]*exp_recs[c])/(tau_rec_arr[c] - tau_psc_arr[c]);
		printf("%f\n", tau_pscs[c]);
//		printf("pre: %i post: %i\n", pre_syns[i], post_syns[i]);
	}

	printf("Synapses initialized successfully!\n");
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
			xs[c] = ys[c]*exp_taus[c] - (weights[c] - xs[c] - ys[c])*exp_recs[c] + weights[c];
			ys[c] = ys[c]*exp_pscs[c];
			us[c] = us[c]*exp_facs[c];

			if (syn_num_spks[c] < neur_num_spks[pre_syns[c]]){
//				printf("Where are unprocessed spikes at %i time: %i!\n", spk_times[Nneur*syn_num_spks[c] + pre_conns[c]], t);
				if (spk_times[Nneur*syn_num_spks[c] + pre_syns[c]] == t - delays[c]){
					us[c] += Us[c]*(1.0f - us[c]);
					float dx = xs[c]*us[c];
					ys[c] += dx;
					xs[c] -= dx;
					syn_num_spks[c]++;
				}
			}
			// When run parallel this incrementation should be atomic
			if (receptor_type[pre_syns[c]] == AMPA_RECEPTOR){
				AMPA_Amuont[post_syns[c]] += ys[c];
			} else if (receptor_type[pre_syns[c]] == GABBA_RECEPTOR){
				GABBA_Amuont[post_syns[c]] += ys[c];
			}
		}

		for (int n = 0; n < Nneur; n++){
			float Isyn_new = -AMPA_Amuont[n]*(Vms[n] - Erev_exc[n]) - GABBA_Amuont[n]*(Vms[n] - Erev_inh[n]);
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
