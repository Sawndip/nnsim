/*
 * main_neuronet.h
 *
 *  Created on: 08 мая 2014 г.
 *      Author: pavel
 */

#ifndef MAIN_NEURONET_H_
#define MAIN_NEURONET_H_

float h;
int Ncon;
int Nneur;
float SimTime;

float* Vms;
float* Ums;
float* Ies;
float* as;
float* bs;
float* cs;
float* ds;
float* ks;
float* Cms;
float* Vrs;
float* Vts;
float* Vpeaks;
float* Isyns;
float* Erev_exc;
float* Erev_inh;

unsigned int* spk_times;
unsigned int* neur_num_spks;
unsigned int* syn_num_spks;

float* y_exc;
float* x_exc;
float* u_exc;
float* U_exc;
float* tau_psc_exc;
float* tau_rec_exc;
float* tau_fac_exc;

float* y_inh;
float* x_inh;
float* u_inh;
float* U_inh;
float* tau_psc_inh;
float* tau_rec_inh;
float* tau_fac_inh;

float* weights;
unsigned int* delays;
int* pre_conns;
int* post_conns;


void fill_syn_params();

void fill_conn_params();

void fill_neur_params();


#endif /* MAIN_NEURONET_H_ */
