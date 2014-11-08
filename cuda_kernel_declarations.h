/*
 * cuda_kernel_api.h
 *
 *  Created on: 05.11.2014
 *      Author: pavel
 */

#ifndef CUDA_KERNEL_API_H_
#define CUDA_KERNEL_API_H_

#include <stdio.h>

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

namespace nnsim {

	extern float time_step;
	extern int Ncon;
	extern int Nneur;
	extern unsigned int Tsim;

	// Neural variables and parameters
	extern float* Vms;
	extern float* Ums;
	extern float* Ies;
	extern float* as;
	extern float* b1_s;
	extern float* b2_s;
	extern float* cs;
	extern float* ds;
	extern float* ks;
	extern float* Cms;
	extern float* Vrs;
	extern float* Vts;
	extern float* Vpeaks;
	extern float* p1_s;
	extern float* p2_s;
	extern float* Isyns;
	extern float* Erev_exc;
	extern float* Erev_inh;
	extern float* AMPA_Amuont;
	extern float* GABA_Amuont;
	extern float* exp_pscs_exc;
	extern float* exp_pscs_inh;
	extern unsigned int* psn_times;
	extern unsigned int* psn_seeds;
	extern float* psn_rates;
	extern float* y_psns;
	extern float* exp_psns;
	extern float* psn_weights;
	extern unsigned int* spk_times;
	extern unsigned int* neur_num_spks;
	extern unsigned int* syn_num_spks;
	extern unsigned int len_spk_tms;

	// Synaptic parameters and variables
	extern float* xs;
	extern float* us;
	extern float* Us;
	extern float* exp_recs;
	extern float* exp_facs;
	extern float* weights;
	extern unsigned int* delays;
	extern unsigned int* pre_syns;
	extern unsigned int* post_syns;
	extern unsigned int* receptor_type;
}
	using namespace nnsim;

	// Neural variables and parameters
	float* Vms_dev;
	float* Ums_dev;
	float* Ies_dev;
	float* as_dev;
	float* b1_s_dev;
	float* b2_s_dev;
	float* cs_dev;
	float* ds_dev;
	float* ks_dev;
	float* Cms_dev;
	float* Vrs_dev;
	float* Vts_dev;
	float* Vpeaks_dev;
	float* p1_s_dev;
	float* p2_s_dev;
	float* Isyns_dev;
	float* Erev_exc_dev;
	float* Erev_inh_dev;
	float* AMPA_Amuont_dev;
	float* GABA_Amuont_dev;
	float* exp_pscs_exc_dev;
	float* exp_pscs_inh_dev;
	unsigned int* psn_times_dev;
	unsigned int* psn_seeds_dev;
	float* psn_rates_dev;
	float* y_psns_dev;
	float* exp_psns_dev;
	float* psn_weights_dev;
	unsigned int* spk_times_dev;
	unsigned int* neur_num_spks_dev;
	unsigned int* syn_num_spks_dev;

	// Synaptic parameters and variables
	float* xs_dev;
	float* us_dev;
	float* Us_dev;
	float* exp_recs_dev;
	float* exp_facs_dev;
	float* weights_dev;
	unsigned int* delays_dev;
	unsigned int* pre_syns_dev;
	unsigned int* post_syns_dev;
	unsigned int* receptor_type_dev;

	__host__ void init_mem(){
		CUDA_CHECK_RETURN(cudaMalloc((void**) &Vms_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &Ums_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &Ies_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &as_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &b1_s_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &b2_s_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &cs_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &ds_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &ks_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &Cms_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &Vrs_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &Vts_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &Vpeaks_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &p1_s_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &p2_s_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &Isyns_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &Erev_exc_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &Erev_inh_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &AMPA_Amuont_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &GABA_Amuont_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &exp_pscs_exc_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &exp_pscs_inh_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &psn_rates_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &y_psns_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &exp_psns_dev, sizeof(float)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &psn_weights_dev, sizeof(float)*Nneur));

		CUDA_CHECK_RETURN(cudaMalloc((void**) &psn_times_dev, sizeof(unsigned int)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &psn_seeds_dev, sizeof(unsigned int)*Nneur));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &neur_num_spks_dev, sizeof(unsigned int)*Nneur));

		CUDA_CHECK_RETURN(cudaMalloc((void**) &spk_times_dev, sizeof(unsigned int)*len_spk_tms));

		CUDA_CHECK_RETURN(cudaMalloc((void**) &syn_num_spks_dev, sizeof(unsigned int)*Ncon));

		CUDA_CHECK_RETURN(cudaMalloc((void**) &xs_dev, sizeof(float)*Ncon));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &us_dev, sizeof(float)*Ncon));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &Us_dev, sizeof(float)*Ncon));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &exp_recs_dev, sizeof(float)*Ncon));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &exp_facs_dev, sizeof(float)*Ncon));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &weights_dev, sizeof(float)*Ncon));

		CUDA_CHECK_RETURN(cudaMalloc((void**) &delays_dev, sizeof(unsigned int)*Ncon));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &pre_syns_dev, sizeof(unsigned int)*Ncon));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &post_syns_dev, sizeof(unsigned int)*Ncon));
		CUDA_CHECK_RETURN(cudaMalloc((void**) &receptor_type_dev, sizeof(unsigned int)*Ncon));
	}

	__host__ void copy2device(){
		CUDA_CHECK_RETURN(cudaMemcpy(Vms_dev, Vms, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(Ums_dev, Ums, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(Ies_dev, Ies, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(as_dev, as, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(b1_s_dev, b1_s, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(b2_s_dev, b2_s, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(cs_dev, cs, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(ds_dev, ds, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(ks_dev, ks, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(Cms_dev, Cms, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(Vrs_dev, Vrs, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(Vts_dev, Vts, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(Vpeaks_dev, Vpeaks, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(p1_s_dev, p1_s, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(p2_s_dev, p2_s, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(Isyns_dev, Isyns, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(Erev_exc_dev, Erev_exc, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(Erev_inh_dev, Erev_inh, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(AMPA_Amuont_dev, AMPA_Amuont, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(GABA_Amuont_dev, GABA_Amuont, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(exp_pscs_exc_dev, exp_pscs_exc, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(exp_pscs_inh_dev, exp_pscs_inh, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(psn_rates_dev, psn_rates, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(y_psns_dev, y_psns, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(exp_psns_dev, exp_psns, sizeof(float)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(psn_weights_dev, psn_weights, sizeof(float)*Nneur, cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(psn_times_dev, psn_times, sizeof(unsigned int)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(psn_seeds_dev, psn_seeds, sizeof(unsigned int)*Nneur, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(neur_num_spks_dev, neur_num_spks, sizeof(unsigned int)*Nneur, cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(spk_times_dev, spk_times, sizeof(unsigned int)*len_spk_tms, cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(syn_num_spks_dev, syn_num_spks, sizeof(unsigned int)*Ncon, cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(xs_dev, xs, sizeof(float)*Ncon, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(us_dev, us, sizeof(float)*Ncon, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(Us_dev, Us, sizeof(float)*Ncon, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(exp_recs_dev, exp_recs, sizeof(float)*Ncon, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(exp_facs_dev, exp_facs, sizeof(float)*Ncon, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(weights_dev, weights, sizeof(float)*Ncon, cudaMemcpyHostToDevice));

		CUDA_CHECK_RETURN(cudaMemcpy(delays_dev, delays, sizeof(unsigned int)*Ncon, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(pre_syns_dev, pre_syns, sizeof(unsigned int)*Ncon, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(post_syns_dev, post_syns, sizeof(unsigned int)*Ncon, cudaMemcpyHostToDevice));
		CUDA_CHECK_RETURN(cudaMemcpy(receptor_type_dev, receptor_type, sizeof(unsigned int)*Ncon, cudaMemcpyHostToDevice));
	}


#endif /* CUDA_KERNEL_API_H_ */
