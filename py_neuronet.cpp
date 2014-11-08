/*
 * py_neuronet.cpp
 *
 *  Created on: 07 мая 2014 г.
 *      Author: pavel
 */

#include <python2.7/Python.h>
#include <numpy/arrayobject.h>
#include "kernel_api.h"
#include <cstdio>

static const char module_docstring[] = "Python interface for NeuralNetworkSIMulator (NNSIM)";
static const char simulate_docstring[] = "simulate";
static const char init_network_docstring[] = "init_network";
static const char init_neurs_docstring[] = "init_neurs";
static const char init_synapses_docstring[] = "init_synapses";
static const char init_spikes_docstring[] = "init_spikes";
static const char init_recorder_docstring[] = "init_recorder";
static const char get_results_docstring[] = "get_results";
static const char get_spk_times_docstring[] = "get_spk_times";
static const char init_poisson_docstring[] = "init_poisson";
static const char init_mean_recorder_docstring[] = "init_mean_recorder";
static const char add_neur_mean_record_docstring[] = "add_neur_mean_record";
static const char add_conn_mean_record_docstring[] = "add_conn_mean_record";

static PyObject* simulate(PyObject *self, PyObject* args);

static PyObject* init_network(PyObject *self, PyObject* args);

static PyObject* init_neurs(PyObject *self, PyObject* args, PyObject* keywds);

static PyObject* init_synapses(PyObject *self, PyObject* args, PyObject* keywds);

static PyObject* init_spikes(PyObject *self, PyObject* args, PyObject* keywds);

static PyObject* init_recorder(PyObject *self, PyObject* args);

static PyObject* get_results(PyObject *self, PyObject* args);

static PyObject* get_spk_times(PyObject* self, PyObject* args);

static PyObject* init_poisson(PyObject* self, PyObject* args, PyObject* keywds);

static PyObject* init_mean_recorder(PyObject* self, PyObject* args);

static PyObject* add_neur_mean_record(PyObject* self, PyObject* args);

static PyObject* add_conn_mean_record(PyObject* self, PyObject* args);

unsigned int NumNeur;
unsigned int SpkTimesSz;

static PyMethodDef module_methods[] = {
		{"simulate", simulate, METH_VARARGS, simulate_docstring},
		{"init_network", init_network, METH_VARARGS, init_network_docstring},
		{"init_neurs", (PyCFunction) init_neurs, METH_VARARGS | METH_KEYWORDS, init_neurs_docstring},
		{"init_synapses", (PyCFunction) init_synapses, METH_VARARGS | METH_KEYWORDS, init_synapses_docstring},
		{"init_spikes", (PyCFunction) init_spikes, METH_VARARGS | METH_KEYWORDS, init_spikes_docstring},
		{"init_recorder", init_recorder, METH_VARARGS, init_recorder_docstring},
		{"get_results", get_results, METH_VARARGS, get_results_docstring},
		{"get_spk_times", get_spk_times, METH_VARARGS, get_spk_times_docstring},
		{"init_poisson", (PyCFunction) init_poisson, METH_VARARGS | METH_KEYWORDS, init_poisson_docstring},
		{"init_mean_recorder", init_mean_recorder, METH_VARARGS, init_mean_recorder_docstring},
		{"add_neur_mean_record", add_neur_mean_record, METH_VARARGS, add_neur_mean_record_docstring},
		{"add_conn_mean_record", add_conn_mean_record, METH_VARARGS, add_conn_mean_record_docstring},
		{NULL, NULL, 0, NULL}
	};

PyMODINIT_FUNC initnnsim_pykernel(){
	PyObject *m = Py_InitModule3("nnsim_pykernel", module_methods, module_docstring);
	if (m == NULL){
		return ;
	}
	import_array();
}

static PyObject* simulate(PyObject *self, PyObject* args){
	int useGPU;
	if (!PyArg_ParseTuple(args, "i", &useGPU)){
	  return NULL;
	}
	nnsim::simulate(useGPU);
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* init_network(PyObject *self, PyObject* args){
	float SimulationTime, h;
	int Nneur, Ncon;
	if (!PyArg_ParseTuple(args, "fiif", &h, &Nneur, &Ncon, &SimulationTime)){
		 return NULL;
	 }
	nnsim::init_network(h, Nneur, Ncon, SimulationTime);
	NumNeur = Nneur;
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* init_neurs(PyObject *self, PyObject* args, PyObject* keywds){
	int Nparam = 20;
	PyObject** args_pyobj_arr = new PyObject*[Nparam];
	 static char * kwlist[] = {"a", "b_1", "b_2", "c", "d", "k", "Cm", "Erev_AMPA", "Erev_GABA",
			 "Ie", "Isyn", "tau_psc_exc", "tau_psc_inh", "Um", "Vm", "Vpeak", "Vr", "Vt", "p_1", "p_2", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOOOOOOOOOOOOOOOOO", kwlist,
			&args_pyobj_arr[0], &args_pyobj_arr[1], &args_pyobj_arr[2], &args_pyobj_arr[3],
			&args_pyobj_arr[4], &args_pyobj_arr[5], &args_pyobj_arr[6], &args_pyobj_arr[7],
			&args_pyobj_arr[8], &args_pyobj_arr[9], &args_pyobj_arr[10], &args_pyobj_arr[11],
			&args_pyobj_arr[12], &args_pyobj_arr[13], &args_pyobj_arr[14], &args_pyobj_arr[15],
			&args_pyobj_arr[16], &args_pyobj_arr[17], &args_pyobj_arr[18], &args_pyobj_arr[19])){
		return NULL;
	}
	float** args_arr = new float*[Nparam];
	PyObject* arg_npa;
	for (int i = 0; i < Nparam; i++){
		arg_npa = PyArray_FROM_OTF(args_pyobj_arr[i], NPY_FLOAT32, NPY_IN_ARRAY);
		if (arg_npa != NULL){
			args_arr[i] = (float*) PyArray_DATA(arg_npa);
//			Py_DECREF(arg_npa);
		} else{
			Py_XDECREF(arg_npa);
			return NULL;
		}
	}
	nnsim::init_neurs(args_arr[0], args_arr[1], args_arr[2], args_arr[3],
			args_arr[4], args_arr[5], args_arr[6], args_arr[7],
			args_arr[8], args_arr[9], args_arr[10], args_arr[11],
			args_arr[12], args_arr[13], args_arr[14], args_arr[15],
			args_arr[16], args_arr[17], args_arr[18], args_arr[19]);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* init_synapses(PyObject *self, PyObject* args, PyObject* keywds){
	int Nparam = 8;
	int Nparam_int = 3;
	PyObject** args_pyobj_arr = new PyObject*[Nparam + Nparam_int];

	static char *kwlist[] = {"tau_rec", "tau_fac", "U", "x", "y", "u",
							"weight", "delay", "pre", "post", "receptor_type", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOOOOOOOO", kwlist,
			&args_pyobj_arr[0], &args_pyobj_arr[1], &args_pyobj_arr[2], &args_pyobj_arr[3],
			&args_pyobj_arr[4], &args_pyobj_arr[5], &args_pyobj_arr[6], &args_pyobj_arr[7],
			&args_pyobj_arr[8], &args_pyobj_arr[9], &args_pyobj_arr[10])){
		return NULL;
	}

	float** args_arr = new float*[Nparam];
	unsigned int** args_arr_int = new unsigned int*[Nparam_int];
	PyObject* arg_npa;
	for (int i = 0; i < Nparam; i++){
		arg_npa = PyArray_FROM_OTF(args_pyobj_arr[i], NPY_FLOAT32, NPY_IN_ARRAY);
		if (arg_npa != NULL){
			args_arr[i] = (float*) PyArray_DATA(arg_npa);
//			Py_DECREF(arg_npa);
		} else{
			Py_XDECREF(arg_npa);
			return NULL;
		}
	}
	for (int i = Nparam; i < Nparam + Nparam_int; i++){
		arg_npa = PyArray_FROM_OTF(args_pyobj_arr[i], NPY_UINT32, NPY_IN_ARRAY);
		if (arg_npa != NULL){
			args_arr_int[i - Nparam] = (unsigned int*) PyArray_DATA(arg_npa);
//			Py_DECREF(arg_npa);
		} else{
			Py_XDECREF(arg_npa);
			return NULL;
		}
	}
	nnsim::init_synapses(args_arr[0], args_arr[1], args_arr[2], args_arr[3],
			args_arr[4], args_arr[5], args_arr[6], args_arr[7],
			args_arr_int[0], args_arr_int[1], args_arr_int[2]);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* init_spikes(PyObject *self, PyObject* args, PyObject* keywds){
	int Nparam = 3;
	PyObject** args_pyobj_arr = new PyObject*[Nparam];
	 static char *kwlist[] = {"sps_times", "neur_num_spk", "syn_num_spk", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOO", kwlist,
			&args_pyobj_arr[0], &args_pyobj_arr[1], &args_pyobj_arr[2])){
		return NULL;
	}
	unsigned int **args_arr = new unsigned int*[Nparam];
	PyObject* arg_npa;
	for (int i = 0; i < Nparam; i++){
		arg_npa = PyArray_FROM_OTF(args_pyobj_arr[i], NPY_UINT32, NPY_IN_ARRAY);
		if (arg_npa != NULL){
			args_arr[i] = (unsigned int*) PyArray_DATA(arg_npa);
//			Py_DECREF(arg_npa);
		} else{
			Py_XDECREF(arg_npa);
			return NULL;
		}
	}
	SpkTimesSz = PyArray_DIM(args_pyobj_arr[0], 0);

	nnsim::init_spikes(args_arr[0], args_arr[1], args_arr[2], SpkTimesSz);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* init_recorder(PyObject *self, PyObject* args){
	unsigned int neur_num, con_num;
	unsigned int* neurs;
	unsigned int* conns;
	PyObject* neurs_nparr;
	PyObject* conns_nparr;

	if (!PyArg_ParseTuple(args, "IOIO", &neur_num, &neurs_nparr, &con_num, &conns_nparr)){
		 return NULL;
	}
	neurs = (unsigned int*) PyArray_DATA(PyArray_FROM_OTF(neurs_nparr, NPY_UINT32, NPY_IN_ARRAY));
	conns = (unsigned int*) PyArray_DATA(PyArray_FROM_OTF(conns_nparr, NPY_UINT32, NPY_IN_ARRAY));

	nnsim::init_recorder(neur_num, neurs, con_num, conns);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* get_results(PyObject *self, PyObject* args){
	float* Vm_res;
	float* Um_res;
	float* Isyn_res;
	float* y_exc_res;
	float* y_inh_res;
	unsigned int N;

	int meanFlg;
	if (!PyArg_ParseTuple(args, "i", &meanFlg)){
		return NULL;
	}
	if (meanFlg == 1){
		nnsim::get_mean_neur_results(Vm_res, Um_res, Isyn_res, y_exc_res, y_inh_res, N);
	} else{
		nnsim::get_neur_results(Vm_res, Um_res, Isyn_res, y_exc_res, y_inh_res, N);
	}

	npy_intp res_dims[] = {N};
	PyObject* Vm_obj_arr = PyArray_SimpleNewFromData(1, res_dims, NPY_FLOAT32, Vm_res);
	PyObject* Um_obj_arr = PyArray_SimpleNewFromData(1, res_dims, NPY_FLOAT32, Um_res);
	PyObject* Isyn_obj_arr = PyArray_SimpleNewFromData(1, res_dims, NPY_FLOAT32, Isyn_res);
	PyObject* y_exc_obj_arr = PyArray_SimpleNewFromData(1, res_dims, NPY_FLOAT32, y_exc_res);
	PyObject* y_inh_obj_arr = PyArray_SimpleNewFromData(1, res_dims, NPY_FLOAT32, y_inh_res);
	PyArray_ENABLEFLAGS((PyArrayObject *) Vm_obj_arr, NPY_ARRAY_OWNDATA);
	PyArray_ENABLEFLAGS((PyArrayObject *) Um_obj_arr, NPY_ARRAY_OWNDATA);
	PyArray_ENABLEFLAGS((PyArrayObject *) Isyn_obj_arr, NPY_ARRAY_OWNDATA);
	PyArray_ENABLEFLAGS((PyArrayObject *) y_exc_obj_arr, NPY_ARRAY_OWNDATA);
	PyArray_ENABLEFLAGS((PyArrayObject *) y_inh_obj_arr, NPY_ARRAY_OWNDATA);

	float* x_res;
	float* u_res;
	if (meanFlg == 1){
		nnsim::get_mean_conn_results(x_res, u_res, N);
	} else{
		nnsim::get_conn_results(x_res, u_res, N);
	}
	res_dims[0] = N;
	PyObject* x_obj_arr = PyArray_SimpleNewFromData(1, res_dims, NPY_FLOAT32, x_res);
	PyObject* u_obj_arr = PyArray_SimpleNewFromData(1, res_dims, NPY_FLOAT32, u_res);
	PyArray_ENABLEFLAGS((PyArrayObject *) x_obj_arr, NPY_ARRAY_OWNDATA);
	PyArray_ENABLEFLAGS((PyArrayObject *) u_obj_arr, NPY_ARRAY_OWNDATA);

	PyObject* result = Py_BuildValue("(OOOOOOO)", Vm_obj_arr, Um_obj_arr, Isyn_obj_arr, y_exc_obj_arr, y_inh_obj_arr, x_obj_arr, u_obj_arr);
	return result;
}

static PyObject* get_spk_times(PyObject* self, PyObject* args){
	unsigned int* spk_times;
	unsigned int* n_spk;

	nnsim::get_spike_times(spk_times, n_spk);
	npy_intp s_dim[] = {SpkTimesSz};
	npy_intp n_dim[] = {NumNeur};
	PyObject* spk_times_obj = PyArray_SimpleNewFromData(1, s_dim, NPY_UINT32, spk_times);
	PyObject* n_spk_obj = PyArray_SimpleNewFromData(1, n_dim, NPY_UINT32, n_spk);

	PyObject* result = Py_BuildValue("(OO)", spk_times_obj, n_spk_obj);
	return result;
}

static PyObject* init_poisson(PyObject* self, PyObject* args, PyObject* keywds){
	unsigned int* seeds;
	float* rates;
	float* weights;
	float psn_tau;

	PyObject* seeds_obj;
	PyObject* rates_obj;
	PyObject* psn_weights_obj;
	PyObject* arg_npa;

	static char *kwlist[] = {"psn_seed", "psn_rate", "psn_weight", "psn_tau", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOf", kwlist,
			&seeds_obj, &rates_obj, &psn_weights_obj, &psn_tau)){
		return NULL;
	}
	arg_npa = PyArray_FROM_OTF(seeds_obj, NPY_UINT32, NPY_IN_ARRAY);
	seeds = (unsigned int*) PyArray_DATA(arg_npa);

	arg_npa = PyArray_FROM_OTF(rates_obj, NPY_FLOAT32, NPY_IN_ARRAY);
	rates = (float*) PyArray_DATA(arg_npa);

	arg_npa = PyArray_FROM_OTF(psn_weights_obj, NPY_FLOAT32, NPY_IN_ARRAY);
	weights = (float*) PyArray_DATA(arg_npa);

	nnsim::init_poisson(seeds, rates, weights, psn_tau);

	Py_INCREF(Py_None);
	return Py_None;
}


static PyObject* init_mean_recorder(PyObject* self, PyObject* args){
	unsigned int num_pop_neur;
	unsigned int num_pop_conn;
	if (!PyArg_ParseTuple(args, "II", &num_pop_neur, &num_pop_conn)){
		 return NULL;
	}
	nnsim::init_mean_recorder(num_pop_neur, num_pop_conn);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* add_neur_mean_record(PyObject* self, PyObject* args){
	PyObject* neurs_obj;
	unsigned int* neurs;
	if (!PyArg_ParseTuple(args, "O", &neurs_obj)){
		 return NULL;
	}
	neurs = (unsigned int*) PyArray_DATA(PyArray_FROM_OTF(neurs_obj, NPY_UINT32, NPY_IN_ARRAY));

	npy_intp* p = PyArray_SHAPE((PyArrayObject*) neurs_obj);

	nnsim::add_neur_mean_record(p[0], neurs);
	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* add_conn_mean_record(PyObject* self, PyObject* args){
	PyObject* conns_obj;
	unsigned int* conns;
	if (!PyArg_ParseTuple(args, "O", &conns_obj)){
		 return NULL;
	}
	conns = (unsigned int*) PyArray_DATA(PyArray_FROM_OTF(conns_obj, NPY_UINT32, NPY_IN_ARRAY));

	npy_intp* p = PyArray_SHAPE((PyArrayObject*) conns_obj);

	nnsim::add_conn_mean_record(p[0], conns);

	Py_INCREF(Py_None);
	return Py_None;
}
