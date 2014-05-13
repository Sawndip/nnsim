'''
Created on 09.03.2014

@author: pavel
'''
from distutils.core import setup, Extension
import numpy.distutils.misc_util

extentions = Extension(name="nnsim_pykernel",
                       extra_objects=["Debug/kernel_api.o"],
                       sources=["py_neuronet.cpp"])
setup(
    ext_modules=[extentions],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
