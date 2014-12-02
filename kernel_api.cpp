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
#include "kernel_api.h"

#ifdef USE_CUDA
#include "cuda_kernel_api.h"
#endif

namespace nnsim{

float izhik_Vm(float Vm, float Um, float Isyn, int m){
	return (ks[m]*(Vm - Vrs[m])*(Vm - Vts[m]) - Um + Ies[m] + Isyn)*time_step/Cms[m];
}

float izhik_Um(float Vm, float Um, int m){
	if (Vm < Vrs[m]){
		return time_step*as[m]*(b1_s[m]*powf((Vm - Vrs[m]), p1_s[m]) - Um);
	} else {
		return time_step*as[m]*(b2_s[m]*powf((Vm - Vrs[m]), p2_s[m]) - Um);
	}
}

float get_random(unsigned int *seed){
	// return random number homogeneously distributed in interval [0:1]
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	unsigned long x = (unsigned long) *seed;
	x = (a * x) % m;
	*seed = (unsigned int) x;
	return ((float)x)/m;
}

void init_poisson(unsigned int* seeds, float* rates, float* weights, float psn_tau){
	psn_seeds = seeds;
	psn_rates = rates;
	psn_weights = weights;
	psn_times = new unsigned int[Nneur];
	y_psns = new float[Nneur]();
	exp_psns = new float[Nneur];
	for (int i = 0; i < Nneur; i++){
		if (psn_rates[i] == 0.0f){
			psn_rates[i] = 1.e-6;
		}
		exp_psns[i] = expf(-time_step/psn_tau);
		psn_seeds[i] += 1000000;
		psn_times[i] = 1 -(1000.0f/(time_step*psn_rates[i]))*log(get_random(psn_seeds + i));
	}
}


void init_network(float h, int NumNeur, int NumConns, float SimTime){
	time_step = h;
	Ncon = NumConns;
	Nneur = NumNeur;
	Tsim = SimTime/time_step;
}

void init_neurs(float* a_arr, float* b1_arr, float* b2_arr, float* c_arr, float* d_arr, float* k_arr,
		float* Cm_arr, float* Erev_exc_arr, float* Erev_inh_arr, float* Ie_arr, float* Isyn_arr, float* tau_psc_exc_arr, float* tau_psc_inh_arr,
		float* Um_arr, float* Vm_arr, float* Vpeak_arr, float* Vr_arr, float* Vt_arr, float* p1_arr, float* p2_arr){
	Vms = Vm_arr;
	Ums = Um_arr;
	Ies = Ie_arr;
	as = a_arr;
	b1_s = b1_arr;
	b2_s = b2_arr;
	cs = c_arr;
	ds = d_arr;
	ks = k_arr;
	Cms = Cm_arr;
	Vrs = Vr_arr;
	Vts = Vt_arr;
	p1_s = p1_arr;
	p2_s = p2_arr;
	Vpeaks = Vpeak_arr;
	Isyns = Isyn_arr;
	Erev_exc = Erev_exc_arr;
	Erev_inh = Erev_inh_arr;
	AMPA_Amuont = new float[Nneur]();
	GABA_Amuont = new float[Nneur]();

	delete[] exp_pscs_exc;
	delete[] exp_pscs_inh;
	exp_pscs_exc = new float[Nneur];
	exp_pscs_inh = new float[Nneur];
	for (int n = 0; n < Nneur; n++){
		exp_pscs_exc[n] = expf(-time_step/tau_psc_exc_arr[n]);
		exp_pscs_inh[n] = expf(-time_step/tau_psc_inh_arr[n]);
	}
//	for (int i = 0; i < Nneur; i++){
////		printf("%i: %g %g\n", i, p1_s[i], p2_s[i]);
//		printf("%i: %g %g\n", i, tau_psc_exc_arr[i], tau_psc_inh_arr[i]);
//	}
}

void init_synapses(float* tau_rec_arr, float* tau_fac_arr, float* U_arr,
		float* x_arr, float* y_arr, float* u_arr, float* weights_arr, float* delays_arr,
		unsigned int* pre_conns_arr, unsigned int* post_conns_arr, unsigned int* receptor_type_arr){
	xs = x_arr;
	us = u_arr;
	Us = U_arr;
	weights = weights_arr;
	pre_syns = pre_conns_arr;
	post_syns = post_conns_arr;
	receptor_type = receptor_type_arr;

	delete[] exp_recs;
	delete[] exp_facs;
	delete[] delays;

	exp_recs = new float[Ncon];
	exp_facs = new float[Ncon];
	delays = new unsigned int[Ncon];
	
	for (int c = 0; c < Ncon; c++){
 		xs[c] = weights[c]*xs[c];
 		if (receptor_type[c] == AMPA_RECEPTOR){
			AMPA_Amuont[post_syns[c]] += weights[c]*y_arr[c];
 		} else if (receptor_type[c] == GABA_RECEPTOR){
			GABA_Amuont[post_syns[c]] += weights[c]*y_arr[c];
 		}
 		exp_recs[c] = expf(-time_step/tau_rec_arr[c]);
 		exp_facs[c] = expf(-time_step/tau_fac_arr[c]);
 		delays[c] = delays_arr[c]/time_step;
//		printf("pre: %i post: %i delay: %f\n", pre_syns[c], post_syns[c], delays_arr[c]);
	}
}

void init_spikes(unsigned int* spike_times, unsigned int* neur_num_spikes,
		unsigned int* syn_num_spikes, unsigned int spk_times_len){
	delete[] spk_times;
	delete[] neur_num_spks;
	delete[] syn_num_spks;
	spk_times = spike_times;
	neur_num_spks = neur_num_spikes;
	syn_num_spks = syn_num_spikes;
	len_spk_tms = spk_times_len;
}

void simulateOnCPU(){
	float v1, u1, v2, u2, v3, u3, v4, u4;
	float delta_x;
	for (unsigned int t = 0; t < Tsim; t++){

		for (int n = 0; n < Nneur; n++){
			y_psns[n] *= exp_psns[n];
			while (psn_times[n] == t){
				y_psns[n] += psn_weights[n];
				psn_times[n] -= -1 + (1000.0f/(time_step*psn_rates[n]))*log(get_random(psn_seeds + n));
			}
			AMPA_Amuont[n] *= exp_pscs_exc[n];
			GABA_Amuont[n] *= exp_pscs_inh[n];

			// y_psns is here because poisson noise is excitatory
			float Isyn_new = -(AMPA_Amuont[n] + y_psns[n])*(Vms[n] - Erev_exc[n]) - GABA_Amuont[n]*(Vms[n] - Erev_inh[n]);
			float Vm = Vms[n];
			float Um = Ums[n];
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

			if (Vm > Vpeaks[n]){
//				printf("Spike! neur: %i time: %f\n", n, t*time_step);
				spk_times[Nneur*neur_num_spks[n] + n] = t;
				neur_num_spks[n]++;
				Vms[n] = cs[n];
				Ums[n] = Um + ds[n];
			}

			Isyns[n] = Isyn_new;
		}
		for (int c = 0; c < Ncon; c++){
			xs[c] = (xs[c] - weights[c])*exp_recs[c] + weights[c];
			us[c] = us[c]*exp_facs[c];

			if (syn_num_spks[c] < neur_num_spks[pre_syns[c]]){
				if (t >= delays[c] && spk_times[Nneur*syn_num_spks[c] + pre_syns[c]] == t - delays[c]){
//					printf("Spike! neur: %i time: %f\n", post_syns[c], t*time_step);
					us[c] += Us[c]*(1.0f - us[c]);
					delta_x = xs[c]*us[c];
					xs[c] -= delta_x;
					syn_num_spks[c]++;
					// When run parallel this incrementation should be atomic
					if (receptor_type[c] == AMPA_RECEPTOR){
						AMPA_Amuont[post_syns[c]] += delta_x;
					} else if (receptor_type[c] == GABA_RECEPTOR){
						GABA_Amuont[post_syns[c]] += delta_x;
					}
				}
			}
		}


		// Saving variables
		for (unsigned int rec_neur = 0; rec_neur < recorded_neur_num; rec_neur++){
			Vm_recorded[Tsim*rec_neur + t] = Vms[neurs_to_record[rec_neur]];
			Um_recorded[Tsim*rec_neur + t] = Ums[neurs_to_record[rec_neur]];
			Isyn_recorded[Tsim*rec_neur + t] = Isyns[neurs_to_record[rec_neur]];
			y_exc_recorded[Tsim*rec_neur + t] = AMPA_Amuont[neurs_to_record[rec_neur]];
			y_inh_recorded[Tsim*rec_neur + t] = GABA_Amuont[neurs_to_record[rec_neur]];
		}

		for(unsigned int rec_con = 0; rec_con < recorded_con_num; rec_con++){
			x_recorded[Tsim*rec_con + t] = xs[conns_to_record[rec_con]];
			u_recorded[Tsim*rec_con + t] = us[conns_to_record[rec_con]];
		}

		// Saving mean variables
		for(unsigned int pn = 0; pn < NumPopNeur; pn++){
			for (unsigned int i = 0; i < PopNeurSizes[pn]; i++){
				Vm_means[Tsim*pn + t] += Vms[PopNeurs[pn][i]];
				Um_means[Tsim*pn + t] += Ums[PopNeurs[pn][i]];
				Isyn_means[Tsim*pn + t] += Isyns[PopNeurs[pn][i]];
				y_exc_means[Tsim*pn + t] += AMPA_Amuont[PopNeurs[pn][i]];
				y_inh_means[Tsim*pn + t] += GABA_Amuont[PopNeurs[pn][i]];
//				printf("t: %f, Vm %i: %f\n", t*time_step, PopNeurs[pn][i], Vms[PopNeurs[pn][i]]);
			}
			Vm_means[Tsim*pn + t] /= PopNeurSizes[pn];
			Um_means[Tsim*pn + t] /= PopNeurSizes[pn];
			Isyn_means[Tsim*pn + t] /= PopNeurSizes[pn];
			y_exc_means[Tsim*pn + t] /= PopNeurSizes[pn];
			y_inh_means[Tsim*pn + t] /= PopNeurSizes[pn];
		}

		for(unsigned int pn = 0; pn < NumPopConn; pn++){
			for (unsigned int i = 0; i < PopConnSizes[pn]; i++){
				x_means[Tsim*pn + t] += xs[PopConns[pn][i]];
				u_means[Tsim*pn + t] += us[PopConns[pn][i]];
			}
			x_means[Tsim*pn + t] /= PopConnSizes[pn];
			u_means[Tsim*pn + t] /= PopConnSizes[pn];
		}
	}
}

int simulate(int useGPU=0){
#ifdef USE_CUDA
	if (useGPU){
		simulateOnGpu();
		printf("Simulating on GPU by using CUDA!\n");
	} else{
#endif
		printf("Simulating on CPU!\n");
		simulateOnCPU();
#ifdef USE_CUDA
	}
#endif

	return 0;
}

void init_recorder(unsigned int neur_num, unsigned int* neurs, unsigned int con_num, unsigned int* conns){
	recorded_neur_num  = neur_num;
	neurs_to_record = neurs;
	delete[] Vm_recorded;
	delete[] Um_recorded;
	delete[] Isyn_recorded;
	delete[] y_exc_recorded;
	delete[] y_inh_recorded;
	Vm_recorded = new float[recorded_neur_num*Tsim];
	Um_recorded = new float[recorded_neur_num*Tsim];
	Isyn_recorded = new float[recorded_neur_num*Tsim];
	y_exc_recorded = new float[recorded_neur_num*Tsim];
	y_inh_recorded = new float[recorded_neur_num*Tsim];
//	printf("Recorder initialized: [");
//	for (unsigned int i = 0; i < neur_num - 1; i++){
//		printf("%i,", neurs[i]);
//	}
//	printf("%i]\n", neurs[neur_num-1]);

	recorded_con_num = con_num;
	conns_to_record = conns;
	delete[] x_recorded;
	delete[] u_recorded;
	x_recorded = new float[recorded_con_num*Tsim];
	u_recorded = new float[recorded_con_num*Tsim];
}

void get_neur_results(float* &Vm_res, float* &Um_res, float* &Isyn_res, float* &y_exc_res, float* &y_inh_res, unsigned int &N){
	Vm_res = Vm_recorded;
	Um_res = Um_recorded;
	Isyn_res = Isyn_recorded;
	y_exc_res = y_exc_recorded;
	y_inh_res = y_inh_recorded;
	N = recorded_neur_num*Tsim;
}

void get_conn_results(float* &x_res, float* &u_res, unsigned int &N){
	x_res = x_recorded;
	u_res = u_recorded;
	N = recorded_con_num*Tsim;
}

void get_mean_neur_results(float* &Vm_res, float* &Um_res, float* &Isyn_res, float* &y_exc_res, float* &y_inh_res, unsigned int &N){
	Vm_res = Vm_means;
	Um_res = Um_means;
	Isyn_res = Isyn_means;
	y_exc_res = y_exc_means;
	y_inh_res = y_inh_means;
	N = NumPopNeur*Tsim;
}

void get_mean_conn_results(float* &x_res, float* &u_res, unsigned int &N){
	x_res = x_means;
	u_res = u_means;
	N = NumPopConn*Tsim;
}

void get_spike_times(unsigned int* &spike_times, unsigned int* &num_spikes_on_neur){
	spike_times = spk_times;
	num_spikes_on_neur = neur_num_spks;
}

void init_mean_recorder(unsigned int num_pop_neur, unsigned int num_pop_conn){
	NumPopNeur = num_pop_neur;
	NumPopConn = num_pop_conn;
	CurrentPopConn = 0;
	CurrentPopNeur = 0;
	PopNeurSizes  = new unsigned int[NumPopNeur];
	PopConnSizes = new unsigned int[NumPopConn];
	PopNeurs = new unsigned int*[NumPopNeur];
	PopConns = new unsigned int*[NumPopConn];

	Vm_means = new float[Tsim*NumPopNeur]();
	Um_means = new float[Tsim*NumPopNeur]();
	Isyn_means = new float[Tsim*NumPopNeur]();
	y_exc_means = new float[Tsim*NumPopNeur]();
	y_inh_means = new float[Tsim*NumPopNeur]();

	x_means = new float[Tsim*NumPopConn]();
	u_means = new float[Tsim*NumPopConn]();
}

void add_neur_mean_record(unsigned int pop_size, unsigned int* pop_neurs){
	PopNeurSizes[CurrentPopNeur] = pop_size;
	PopNeurs[CurrentPopNeur] = pop_neurs;
//	printf("Record from neurs: [");
//	for (int i = 0; i < PopNeurSizes[CurrentPopNeur] - 1; i++){
//		printf("%i,", PopNeurs[CurrentPopNeur][i]);
//	}
//	printf("%i]\n", PopNeurs[CurrentPopNeur][PopNeurSizes[CurrentPopNeur] -1]);
	CurrentPopNeur ++;
}

void add_conn_mean_record(unsigned int pop_size, unsigned int* pop_conns){
	PopConnSizes[CurrentPopConn] = pop_size;
	PopConns[CurrentPopConn] = pop_conns;
	CurrentPopConn ++;
}

}
