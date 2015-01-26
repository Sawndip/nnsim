'''
Created on 09.03.2014

@author: pavel
'''
from distutils.core import setup

setup(
    name='NNSim',
    version='0.1',
    description='Module for simulating dynamics of spiking neural network',
    long_description="Module for simulating dynamics of spiking neural network with Izhikevich neuron model and Tsodyks-Markram synapse model",
    author='Pavel Esir',
    author_email='esirpavel@gmail.com',
    url='https://github.com/esirpavel',
    packages=['nnsim'],
    package_dir={'nnsim': 'build'},
    package_data={'nnsim': ['nnsim_pykernel.pyd']}
)
