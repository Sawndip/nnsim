'''
Created on 09.03.2014

@author: pavel
'''
from distutils.core import setup, Extension
import numpy.distutils.misc_util

import os
from distutils.sysconfig import get_config_vars
(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

extentions = Extension(name="nnsim_pykernel",
#                        extra_objects=["Debug/kernel_api.o"],
                       sources=["py_neuronet.cpp"],
                       library_dirs=[".", "./Release"],
                       libraries=['kernel_api'],
                       extra_link_args=['-Wl,-rpath,. -Wl,-rpath,Release'],
                       extra_compile_args=['-Wwrite-strings']
                       )
setup(
    ext_modules=[extentions],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
