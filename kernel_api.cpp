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

namespace nnsim{

float izhik_Vm(float Vm, float Um, float Isyn, int m){
	return (ks[m]*(Vm - Vrs[m])*(Vm - Vts[m]) - Um + Ies[m] + Isyn)*time_step/Cms[m];
}

float izhik_Um(float Vm, float Um, int m){
	return time_step*as[m]*(bs[m]*(Vm - Vrs[m]) - Um);
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
		psn_times[i] = -(1000.0f/(time_step*psn_rates[i]))*log(get_random(psn_seeds + i));
	}
}


void init_network(float h, int NumNeur, int NumConns, float SimTime){
	time_step = h;
	Ncon = NumConns;
	Nneur = NumNeur;
	Tsim = SimTime/time_step;
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
}

void init_synapses(float* tau_rec_arr, float* tau_psc_arr, float* tau_fac_arr, float* U_arr,
		float* x_arr, float* y_arr, float* u_arr, float* weights_arr, float* delays_arr,
		unsigned int* pre_conns_arr, unsigned int* post_conns_arr, unsigned int* receptor_type_arr){
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

	delete[] exp_pscs;
	delete[] exp_recs;
	delete[] exp_facs;
	delete[] exp_taus;
	delete[] delays;

	exp_pscs = new float[Ncon];
	exp_recs = new float[Ncon];
	exp_facs = new float[Ncon];
	exp_taus = new float[Ncon];
	delays = new unsigned int[Ncon];
	
	for (int c = 0; c < Ncon; c++){
 		xs[c] = weights[c]*xs[c];
 		ys[c] = weights[c]*ys[c];
 		exp_pscs[c] = expf(-time_step/tau_psc_arr[c]);
 		exp_recs[c] = expf(-time_step/tau_rec_arr[c]);
 		exp_facs[c] = expf(-time_step/tau_fac_arr[c]);
 		exp_taus[c] = (tau_psc_arr[c]*exp_pscs[c] - tau_rec_arr[c]*exp_recs[c])/(tau_rec_arr[c] - tau_psc_arr[c]);
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

int simulate(){
	float v1, u1, v2, u2, v3, u3, v4, u4;
	float delta_x;
	for (unsigned int t = 0; t < Tsim; t++){

		for (int n = 0; n < Nneur; n++){
			y_psns[n] *= exp_psns[n];
			while (psn_times[n] == t){
				y_psns[n] += psn_weights[n];
				psn_times[n] -= (1000.0f/(time_step*psn_rates[n]))*log(get_random(psn_seeds + n));
			}

			// y_psns is here because poisson noise is excitatory
			float Isyn_new = -(AMPA_Amuont[n] + y_psns[n])*(Vms[n] - Erev_exc[n]) - GABBA_Amuont[n]*(Vms[n] - Erev_inh[n]);
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
		for (int c = 0; c < Ncon; c++){
			xs[c] = ys[c]*exp_taus[c] - (weights[c] - xs[c] - ys[c])*exp_recs[c] + weights[c];
			ys[c] = ys[c]*exp_pscs[c];
			us[c] = us[c]*exp_facs[c];

			if (syn_num_spks[c] < neur_num_spks[pre_syns[c]]){
				if (t >= delays[c] && spk_times[Nneur*syn_num_spks[c] + pre_syns[c]] == t - delays[c]){
					us[c] += Us[c]*(1.0f - us[c]);
					delta_x = xs[c]*us[c];
					ys[c] += delta_x;
					xs[c] -= delta_x;
					syn_num_spks[c]++;
				}
			}
			// When run parallel this incrementation should be atomic
			if (receptor_type[c] == AMPA_RECEPTOR){
				AMPA_Amuont[post_syns[c]] += ys[c];
			} else if (receptor_type[c] == GABBA_RECEPTOR){
				GABBA_Amuont[post_syns[c]] += ys[c];
			}
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

		for(unsigned int pn = 0; pn < NumPopNeur; pn++){
			for (unsigned int i = 0; i < PopNeurSizes[pn]; i++){
				Vm_means[Tsim*pn + t] += Vms[PopNeurs[pn][i]];
				Um_means[Tsim*pn + t] += Ums[PopNeurs[pn][i]];
				Isyn_means[Tsim*pn + t] += Isyns[PopNeurs[pn][i]];
//				printf("t: %f, Vm %i: %f\n", t*time_step, PopNeurs[pn][i], Vms[PopNeurs[pn][i]]);
			}
			Vm_means[Tsim*pn + t] /= PopNeurSizes[pn];
			Um_means[Tsim*pn + t] /= PopNeurSizes[pn];
			Isyn_means[Tsim*pn + t] /= PopNeurSizes[pn];
		}

		for(unsigned int pn = 0; pn < NumPopConn; pn++){
			for (unsigned int i = 0; i < PopConnSizes[pn]; i++){
				x_means[Tsim*pn + t] += xs[PopConns[pn][i]];
				y_means[Tsim*pn + t] += ys[PopConns[pn][i]];
				u_means[Tsim*pn + t] += us[PopConns[pn][i]];
			}
			x_means[Tsim*pn + t] /= PopConnSizes[pn];
			y_means[Tsim*pn + t] /= PopConnSizes[pn];
			u_means[Tsim*pn + t] /= PopConnSizes[pn];
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
	delete[] Vm_recorded;
	delete[] Um_recorded;
	delete[] Isyn_recorded;
	Vm_recorded = new float[recorded_neur_num*Tsim];
	Um_recorded = new float[recorded_neur_num*Tsim];
	Isyn_recorded = new float[recorded_neur_num*Tsim];
//	printf("Recorder initialized: [");
//	for (unsigned int i = 0; i < neur_num - 1; i++){
//		printf("%i,", neurs[i]);
//	}
//	printf("%i]\n", neurs[neur_num-1]);

	recorded_con_num = con_num;
	conns_to_record = conns;
	delete[] x_recorded;
	delete[] y_recorded;
	delete[] u_recorded;
	x_recorded = new float[recorded_con_num*Tsim];
	y_recorded = new float[recorded_con_num*Tsim];
	u_recorded = new float[recorded_con_num*Tsim];
}

void get_neur_results(float* &Vm_res, float* &Um_res, float* &Isyn_res, unsigned int &N){
	Vm_res = Vm_recorded;
	Um_res = Um_recorded;
	Isyn_res = Isyn_recorded;
	N = recorded_neur_num*Tsim;
}

void get_conn_results(float* &x_res, float* &y_res, float* &u_res, unsigned int &N){
	x_res = x_recorded;
	y_res = y_recorded;
	u_res = u_recorded;
	N = recorded_con_num*Tsim;
}

void get_mean_neur_results(float* &Vm_res, float* &Um_res, float* &Isyn_res, unsigned int &N){
	Vm_res = Vm_means;
	Um_res = Um_means;
	Isyn_res = Isyn_means;
	N = NumPopNeur*Tsim;
}

void get_mean_conn_results(float* &x_res, float* &y_res, float* &u_res, unsigned int &N){
	x_res = x_means;
	y_res = y_means;
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

	x_means = new float[Tsim*NumPopConn]();
	y_means = new float[Tsim*NumPopConn]();
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
