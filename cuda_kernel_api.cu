/*
 * cuda_kernel_api.cpp
 *
 *  Created on: 01 нояб. 2014 г.
 *      Author: pavel
 */

#include "cuda_kernel_declarations.h"
#include "nnsim_constants.h"

__device__ float get_random(unsigned int *seed){
	// return random number homogeneously distributed in interval [0:1]
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	unsigned long x = (unsigned long) *seed;
	x = (a * x) % m;
	*seed = (unsigned int) x;
	return ((float)x)/m;
}

__global__ void integrate_synapses(float* x, float* u, float* exp_rec, float* exp_fac, float* U, 
								   float* weight, int* delay, int* pre_syn, int* post_syn, unsigned int* receptor_type,
								   unsigned int* syn_num_spk, unsigned int* neur_num_spk, unsigned int* spk_time,
								   float* AMPA_Amuont, float* GABA_Amuont,
								   unsigned int t, int Ncon, int Nneur){
	unsigned int c = blockDim.x*blockIdx.x + threadIdx.x;
	if (c < Ncon){
		x[c] = (x[c] - weight[c])*exp_rec[c] + weight[c];
		u[c] = u[c]*exp_fac[c];
	
		if (syn_num_spk[c] < neur_num_spk[pre_syn[c]]){
			if (t >= delay[c] && spk_time[Nneur*syn_num_spk[c] + pre_syn[c]] == t - delay[c]){
//				printf("Spike! neur: %i time: %f\n", post_syns[c], t*time_step);
				u[c] += U[c]*(1.0f - u[c]);
				float delta_x = x[c]*u[c];
				x[c] -= delta_x;
				syn_num_spk[c]++;
				
				// When run parallel this incrementation should be atomic
				if (receptor_type[c] == AMPA_RECEPTOR){
					atomicAdd(&AMPA_Amuont[post_syn[c]], delta_x);
				} else if (receptor_type[c] == GABA_RECEPTOR){
					atomicAdd(&GABA_Amuont[post_syn[c]], delta_x);
				}
			}
		}
	
	}
}

__device__ __inline__ float check_pow(float x, float degr){
	if (degr == 1.0f){
		return x;
	} else {
		return powf(x, degr);
	}
}

__global__ void integrate_neurons(float* Vms, float* Ums,
		float* a, float* b1, float* b2, float* c, float* d, float* k, float* p1, float* p2,
		float* Vpeak, float* Vr, float* Vt, float* Cm, float* Ie, float* Isyn,
		float* AMPA_Amuont, float* GABA_Amuont, float* exp_pscs_exc, float* exp_pscs_inh,
		float* Erev_exc, float* Erev_inh, 
		float* y_psn, float* psn_weight, float* exp_psn, unsigned int* psn_time, float* psn_rate, unsigned int* psn_seed,
		unsigned int* spk_time, unsigned int* neur_num_spk, 
		unsigned int t, float time_step, int Nneur){
	unsigned int n = blockIdx.x*blockDim.x + threadIdx.x;
	float v1, u1, v2, u2, v3, u3, v4, u4;
	if (n < Nneur){
		y_psn[n] *= exp_psn[n];
		while (psn_time[n] == t){
			y_psn[n] += psn_weight[n];
			psn_time[n] -= -1 + (1000.0f/(time_step*psn_rate[n]))*log(get_random(psn_seed + n));
		}

		AMPA_Amuont[n] *= exp_pscs_exc[n];
		GABA_Amuont[n] *= exp_pscs_inh[n];
	
		float Vm = Vms[n];
		float Um = Ums[n];
		// y_psns is here because poisson noise is excitatory
		float Isyn_new = -(AMPA_Amuont[n] + y_psn[n])*(Vm - Erev_exc[n]) - GABA_Amuont[n]*(Vm - Erev_inh[n]);

		v1 = (k[n]*(Vm - Vr[n])*(Vm - Vt[n]) - Um + Ie[n] + Isyn[n])*time_step/Cm[n];
		u1 = time_step*a[n]*(Vms[n] < Vr[n] ? b1[n]*check_pow((Vm - Vr[n]), p1[n]) - Um : b2[n]*check_pow((Vm - Vr[n]), p2[n]) - Um);
		Vms[n] = Vm + v1*0.5f;
		Ums[n] = Um + u1*0.5f;
		v2 = (k[n]*(Vms[n] - Vr[n])*(Vms[n] - Vt[n]) - Ums[n] + Ie[n] + (Isyn_new + Isyn[n])*0.5f)*time_step/Cm[n];
		u2 = time_step*a[n]*(Vms[n] < Vr[n] ? b1[n]*check_pow((Vms[n] - Vr[n]), p1[n]) - Ums[n] : b2[n]*check_pow((Vms[n] - Vr[n]), p2[n]) - Ums[n]);
		Vms[n] = Vm + v2*0.5f;
		Ums[n] = Um + u2*0.5f;
		v3 = (k[n]*(Vms[n] - Vr[n])*(Vms[n] - Vt[n]) - Ums[n] + Ie[n] + (Isyn_new + Isyn[n])*0.5f)*time_step/Cm[n];
		u3 = time_step*a[n]*(Vms[n] < Vr[n] ? b1[n]*check_pow((Vms[n] - Vr[n]), p1[n]) - Ums[n] : b2[n]*check_pow((Vms[n] - Vr[n]), p2[n]) - Ums[n]);
		Vms[n] = Vm + v3;
		Ums[n] = Um + u3;
		v4 = (k[n]*(Vms[n] - Vr[n])*(Vms[n] - Vt[n]) - Ums[n] + Ie[n] + Isyn_new)*time_step/Cm[n];
		u4 = time_step*a[n]*(Vms[n] < Vr[n] ? b1[n]*check_pow((Vms[n] - Vr[n]), p1[n]) - Ums[n] : b2[n]*check_pow((Vms[n] - Vr[n]), p2[n]) - Ums[n]);
		Vms[n] = Vm + (v1 + 2.0f*(v2 + v3) + v4)*0.16666666f;
		Ums[n] = Um + (u1 + 2.0f*(u2 + u3) + u4)*0.16666666f;

		if (Vm > Vpeak[n]){
//			printf("Spike! neur: %i time: %f\n", n, t*time_step);
			spk_time[Nneur*neur_num_spk[n] + n] = t;
			neur_num_spk[n]++;
			Vms[n] = c[n];
			Ums[n] = Um + d[n];
		}
		Isyn[n] = Isyn_new;
	}
}

void simulateOnGpu(){
	init_mem();
	copy2device();
	for (unsigned int t = 0; t < Tsim; t++){
		integrate_neurons<<<Nneur/NEUR_BLOCK_SZ + 1, NEUR_BLOCK_SZ>>>(
				Vms_dev, Ums_dev, as_dev, b1_s_dev, b2_s_dev, cs_dev, ds_dev, ks_dev, p1_s_dev, p2_s_dev,
				Vpeaks_dev, Vrs_dev, Vts_dev, Cms_dev, Ies_dev, Isyns_dev,
				AMPA_Amuont_dev, GABA_Amuont_dev, exp_pscs_exc_dev, exp_pscs_inh_dev,
				Erev_exc_dev, Erev_inh_dev,
				y_psns_dev, psn_weights_dev, exp_psns_dev, psn_times_dev, psn_rates_dev, psn_seeds_dev,
				spk_times_dev, neur_num_spks_dev, t, time_step, Nneur);
//		cudaDeviceSynchronize();
//		integrate_synapses<<<Ncon/SYN_BLOCK_SZ + 1, SYN_BLOCK_SZ>>>(
//				);
//		cudaDeviceSynchronize();
	}
	const char* error = cudaGetErrorString(cudaPeekAtLastError());
	printf("%s\n", error);
	error = cudaGetErrorString(cudaThreadSynchronize());
	printf("%s\n", error);
	CUDA_CHECK_RETURN(
		cudaMemcpy(spk_times, spk_times_dev, sizeof(unsigned int)*len_spk_tms, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(
		cudaMemcpy(neur_num_spks, neur_num_spks_dev, sizeof(unsigned int)*Nneur, cudaMemcpyDeviceToHost));
}
