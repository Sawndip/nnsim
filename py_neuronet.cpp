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
static const char init_network_docstring[] = "init_network";
static const char init_neurs_docstring[] = "init_neurs";
static const char init_synapses_docstring[] = "init_synapses";
static const char init_spikes_docstring[] = "init_spikes";
static const char init_simulate_docstring[] = "simulate";

static PyObject* init_network(PyObject *self, PyObject* args);

static PyObject* init_neurs(PyObject *self, PyObject* args, PyObject* keywds);

static PyObject* init_synapses(PyObject *self, PyObject* args, PyObject* keywds);

static PyObject* init_spikes(PyObject *self, PyObject* args, PyObject* keywds);

static PyObject* simulate(PyObject *self, PyObject* args);

static PyMethodDef module_methods[] = {
		{"init_network", init_network, METH_VARARGS, init_network_docstring},
		{"init_neurs", (PyCFunction) init_neurs, METH_VARARGS | METH_KEYWORDS, init_neurs_docstring},
		{"init_synapses", (PyCFunction) init_synapses, METH_VARARGS | METH_KEYWORDS, init_synapses_docstring},
		{"init_spikes", (PyCFunction) init_spikes, METH_VARARGS | METH_KEYWORDS, init_spikes_docstring},
		{"simulate", simulate, METH_VARARGS, init_simulate_docstring},
		{NULL, NULL, 0, NULL}
	};

PyMODINIT_FUNC initnnsim_pykernel(){
	PyObject *m = Py_InitModule3("nnsim_pykernel", module_methods, module_docstring);
	if (m == NULL){
		return ;
	}
	import_array();
}

static PyObject* init_network(PyObject *self, PyObject* args){
	float SimulationTime, h;
	int Nneur, Ncon;

	if (!PyArg_ParseTuple(args, "fiif", &h, &Nneur, &Ncon, &SimulationTime)){
		 return NULL;
	 }
	nnsim::init_network(h, Nneur, Ncon, SimulationTime, 0);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* init_neurs(PyObject *self, PyObject* args, PyObject* keywds){
	int Nparam = 15;
	PyObject** args_pyobj_arr = new PyObject*[Nparam];
	 static char * kwlist[] = {"a", "b", "c", "d", "k", "Cm", "Erev_AMPA", "Erev_GABBA",
			 "Ie", "Isyn", "Um", "Vm", "Vpeak", "Vr", "Vt", NULL};

	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOOOOOOOOOOOO", kwlist,
			&args_pyobj_arr[0], &args_pyobj_arr[1], &args_pyobj_arr[2], &args_pyobj_arr[3],
			&args_pyobj_arr[4], &args_pyobj_arr[5], &args_pyobj_arr[6], &args_pyobj_arr[7],
			&args_pyobj_arr[8], &args_pyobj_arr[9], &args_pyobj_arr[10], &args_pyobj_arr[11],
			&args_pyobj_arr[12], &args_pyobj_arr[13], &args_pyobj_arr[14])){
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
			args_arr[12], args_arr[13], args_arr[14]);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* init_synapses(PyObject *self, PyObject* args, PyObject* keywds){
	int Nparam = 9;
	int Nparam_int = 3;
	PyObject** args_pyobj_arr = new PyObject*[Nparam + Nparam_int];

	static char *kwlist[] = {"tau_rec", "tau_psc", "tau_fac", "U", "x", "y", "u",
							"weights", "delays", "pre", "post", "receptor_type", NULL};
	if (!PyArg_ParseTupleAndKeywords(args, keywds, "OOOOOOOOOOOO", kwlist,
			&args_pyobj_arr[0], &args_pyobj_arr[1], &args_pyobj_arr[2], &args_pyobj_arr[3],
			&args_pyobj_arr[4], &args_pyobj_arr[5], &args_pyobj_arr[6], &args_pyobj_arr[7],
			&args_pyobj_arr[8], &args_pyobj_arr[9], &args_pyobj_arr[10], &args_pyobj_arr[11])){
		return NULL;
	}

	float** args_arr = new float*[Nparam];
	int** args_arr_int = new int*[Nparam_int];
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
		arg_npa = PyArray_FROM_OTF(args_pyobj_arr[i], NPY_INT32, NPY_IN_ARRAY);
		if (arg_npa != NULL){
			args_arr_int[i - Nparam] = (int*) PyArray_DATA(arg_npa);
//			Py_DECREF(arg_npa);
		} else{
			Py_XDECREF(arg_npa);
			return NULL;
		}
	}
	nnsim::init_synapses(args_arr[0], args_arr[1], args_arr[2], args_arr[3],
			args_arr[4], args_arr[5], args_arr[6], args_arr[7],
			args_arr[8], args_arr_int[0], args_arr_int[1], args_arr_int[2]);

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
	nnsim::init_spikes(args_arr[0], args_arr[1], args_arr[2]);

	Py_INCREF(Py_None);
	return Py_None;
}

static PyObject* simulate(PyObject *self, PyObject* args){
	nnsim::simulate();
	Py_INCREF(Py_None);
	return Py_None;
}
