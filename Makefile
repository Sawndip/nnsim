.PHONY: all clean 

CXX=g++
NUMPY_INC_DIR=/usr/lib/python2.7/dist-packages/numpy/core/include
PY_INC_DIR=/usr/include/python2.7
BDIR=./build/
FLAGS=-Wall -O3
prefix=/usr/local

all: $(BDIR)nnsim_pykernel.so $(BDIR)__init__.py

$(BDIR)nnsim_pykernel.so: $(BDIR) $(BDIR)py_neuronet.o $(BDIR)kernel_api.o $(BDIR)cuda_kernel_api.o
	nvcc -shared $(BDIR)py_neuronet.o $(BDIR)kernel_api.o $(BDIR)cuda_kernel_api.o -lpython2.7 -o $@

$(BDIR)py_neuronet.o: py_neuronet.cpp kernel_api.h
	$(CXX) -c -fPIC $(FLAGS) py_neuronet.cpp -Wwrite-strings -o $@

$(BDIR)kernel_api.o: kernel_api.cpp kernel_api.h cuda_kernel_api.h
	$(CXX) -DUSE_CUDA -c -fPIC $(FLAGS) kernel_api.cpp -o $@

$(BDIR)cuda_kernel_api.o: cuda_kernel_api.cu cuda_kernel_declarations.h 
	nvcc -c --use_fast_math -O3 -Xcompiler -fPIC cuda_kernel_api.cu -arch sm_21 -o $@

$(BDIR)__init__.py: $(BDIR)
	cp nnsim.py $@

$(BDIR): 
	mkdir $(BDIR)
	
clean:
	rm -R $(BDIR)

install:
	mkdir $(prefix)/lib/python2.7/dist-packages/nnsim
	install -m 644 $(BDIR)/nnsim_pykernel.so  $(prefix)/lib/python2.7/dist-packages/nnsim
	install -m 755 $(BDIR)/__init__.py $(prefix)/lib/python2.7/dist-packages/nnsim

.PHONY: install
