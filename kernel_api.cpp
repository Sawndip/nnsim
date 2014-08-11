/*
 * kernel_api.cpp
 *
 *  Created on: 14 апр. 2014 г.
 *      Author: pavel
 */
#include "kernel_declarations.h"
#include <cmath>
#include <cstdio>
#include <sstream>

namespace nnsim{

float izhik_Vm(float Vm, float Um, float Isyn, int m){
	return (ks[m]*(Vm - Vrs[m])*(Vm - Vts[m]) - Um + Ies[m] + Isyn)*time_step/Cms[m];
}

float izhik_Um(float Vm, float Um, int m){
	return time_step*as[m]*(bs[m]*(Vm - Vrs[m]) - Um);
}

void init_network(float h, int NumNeur, int NumConns, float SimTime){
	time_step = h;
	Ncon = NumConns;
	Nneur = NumNeur;
	Tsim = SimTime/time_step;
	printf("success initializing network!\n");
}

void init_neurs(float* a_arr, float* b_arr, float* c_arr, float* d_arr, float* k_arr,
		float* Cm_arr, float* Erev_exc_arr, float* Erev_inh_arr, float* Ie_arr, float* Isyn_arr,
		float* Um_arr, float* Vm_arr, float* Vpeak_arr, float* Vr_arr, float* Vt_arr){
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
	AMPA_Amuont = new float[Nneur]();
	GABBA_Amuont = new float[Nneur]();
//	for (int i = 0; i < Nneur; i++){
//		printf("%i: %g\n", i, as[i]);
//	}
	printf("Success initializing neurons!\n");
}

void init_synapses(float* tau_rec_arr, float* tau_psc_arr, float* tau_fac_arr, float* U_arr,
		float* x_arr, float* y_arr, float* u_arr, float* weights_arr, float* delays_arr,
		myUint* pre_conns_arr, myUint* post_conns_arr, myUint* receptor_type_arr){
	ys = y_arr;
	xs = x_arr;
	us = u_arr;
	Us = U_arr;
	weights = weights_arr;
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
	delays = new myUint[Nneur];

	for (int c = 0; c < Ncon; c++){
		exp_pscs[c] = expf(-time_step/tau_psc_arr[c]);
		exp_recs[c] = expf(-time_step/tau_rec_arr[c]);
		exp_facs[c] = expf(-time_step/tau_fac_arr[c]);
		exp_taus[c] = (tau_psc_arr[c]*exp_pscs[c] - tau_rec_arr[c]*exp_recs[c])/(tau_rec_arr[c] - tau_psc_arr[c]);
		delays[c] = delays_arr[c]/time_step;
		printf("pre: %i post: %i delay in step: %i\n", pre_syns[c], post_syns[c], delays[c]);
	}

	printf("Synapses initialized successfully!\n");
}

void init_spikes(unsigned int* spike_times, unsigned int* neur_num_spikes, unsigned int* syn_num_spikes){
	spk_times = spike_times;
	neur_num_spks = neur_num_spikes;
	syn_num_spks = syn_num_spikes;
	printf("Array for storing spikes initialized successfully!\n");
}

int simulate(){
	float v1, u1, v2, u2, v3, u3, v4, u4;
	for (unsigned int t = 0; t < Tsim; t++){
		for (int c = 0; c < Ncon; c++){
			xs[c] = ys[c]*exp_taus[c] - (weights[c] - xs[c] - ys[c])*exp_recs[c] + weights[c];
			ys[c] = ys[c]*exp_pscs[c];
			us[c] = us[c]*exp_facs[c];

			if (syn_num_spks[c] < neur_num_spks[pre_syns[c]]){
				printf("%i\n", t);
				if (t >= delays[c] && spk_times[Nneur*syn_num_spks[c] + pre_syns[c]] == t - delays[c]){
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
			AMPA_Amuont[n] = 0.0f;
			GABBA_Amuont[n] = 0.0f;
		}

		for (unsigned int rec_neur = 0; rec_neur < recorded_neur_num; rec_neur++){
			Vm_recorded[Tsim*rec_neur + t] = Vms[neurs_to_record[rec_neur]];
			Um_recorded[Tsim*rec_neur + t] = Ums[neurs_to_record[rec_neur]];
			Isyn_recorded[Tsim*rec_neur + t] = Isyns[neurs_to_record[rec_neur]];
		}

		for(unsigned int rec_con = 0; rec_con < recorded_con_num; rec_con++){
			x_recorded[Tsim*rec_con + t] = xs[conns_to_record[rec_con]];
			y_recorded[Tsim*rec_con + t] = ys[conns_to_record[rec_con]];
			u_recorded[Tsim*rec_con + t] = us[conns_to_record[rec_con]];
		}
	}

//	for (unsigned int rec_neur = 0; rec_neur < recorded_neur_num; rec_neur++){
//		std::stringstream s;
//		char* name = new char[500];
//		s << "res/" << neurs_to_record[rec_neur] << "_neur_oscill.csv" << std::endl;
//		s >> name;
//		FILE* rf;
//		rf = fopen(name, "w");
//		for (unsigned int t = 0; t < Tsim; t++){
//			unsigned int idx = Tsim*rec_neur + t;
//			fprintf(rf, "%.3f;%.3f;%.3f\n", t*time_step, Vm_recorded[idx], Um_recorded[idx]);
//		}
//		fclose(rf);
//		delete[] name;
//	}

	return 0;
}

void init_recorder(unsigned int neur_num, unsigned int* neurs, unsigned int con_num, unsigned int* conns){
	recorded_neur_num  = neur_num;
	neurs_to_record = neurs;
	Vm_recorded = new myFloat[recorded_neur_num*Tsim];
	Um_recorded = new myFloat[recorded_neur_num*Tsim];
	Isyn_recorded = new myFloat[recorded_neur_num*Tsim];
//	printf("Recorder initialized: [");
//	for (unsigned int i = 0; i < neur_num - 1; i++){
//		printf("%i,", neurs[i]);
//	}
//	printf("%i]\n", neurs[neur_num-1]);

	recorded_con_num = con_num;
	conns_to_record = conns;
	x_recorded = new myFloat[recorded_con_num*Tsim];
	y_recorded = new myFloat[recorded_con_num*Tsim];
	u_recorded = new myFloat[recorded_con_num*Tsim];
}

void get_neur_results(myFloat* &Vm_res, myFloat* &Um_res, myFloat* &Isyn_res, unsigned int &N){
	Vm_res = Vm_recorded;
	Um_res = Um_recorded;
	Isyn_res = Isyn_recorded;
	N = recorded_neur_num*Tsim;
}

void get_conn_results(myFloat* &x_res, myFloat* &y_res, myFloat* &u_res, unsigned int &N){
	x_res = x_recorded;
	y_res = y_recorded;
	u_res = u_recorded;
	N = recorded_con_num*Tsim;
}

void get_spike_times(unsigned int* &spike_times, unsigned int* &num_spikes_on_neur){
	spike_times = spk_times;
	num_spikes_on_neur = neur_num_spks;
}

}
