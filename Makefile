CC=g++-4.6
NUMPY_INC_DIR=/usr/lib/python2.7/dist-packages/numpy/core/include
PY_INC_DIR=/usr/include/python2.7
SRC_DIR=./
BUILD_DIR=./build/
FLAGS=-Wall -O3

all: nnsim_pykernel.so

nnsim_pykernel.so: $(BUILD_DIR) $(BUILD_DIR)py_neuronet.o $(BUILD_DIR)libkernel_api.so $(BUILD_DIR)cuda_kernel_api.o
	nvcc --use_fast_math -O3 -shared $(BUILD_DIR)py_neuronet.o $(BUILD_DIR)cuda_kernel_api.o -L. -L$(BUILD_DIR) -lkernel_api -lpython2.7 -o nnsim_pykernel.so -Xlinker -rpath,.,-rpath,$(BUILD_DIR)
# 	$(CC) -shared $(FLAGS) -pthread $(BUILD_DIR)py_neuronet.o -L. -L$(BUILD_DIR) -lkernel_api -o nnsim_pykernel.so -Wl,-rpath,.,-rpath,$(BUILD_DIR)

$(BUILD_DIR)py_neuronet.o:
	$(CC) -c -fPIC $(FLAGS) $(SRC_DIR)py_neuronet.cpp -Wwrite-strings -o $(BUILD_DIR)py_neuronet.o

$(BUILD_DIR)libkernel_api.so: $(BUILD_DIR)kernel_api.o
	$(CC) -shared $(FLAGS) $(BUILD_DIR)kernel_api.o -o $(BUILD_DIR)libkernel_api.so

$(BUILD_DIR)kernel_api.o:
	$(CC) -c -fPIC $(FLAGS) $(SRC_DIR)kernel_api.cpp -o $(BUILD_DIR)kernel_api.o

$(BUILD_DIR)cuda_kernel_api.o:
	nvcc -c --use_fast_math -O3 -Xcompiler -fPIC cuda_kernel_api.cu -arch sm_21 -o $(BUILD_DIR)cuda_kernel_api.o

$(BUILD_DIR): 
	mkdir $(BUILD_DIR)
	
clean:
	rm $(BUILD_DIR)kernel_api.o $(BUILD_DIR)libkernel_api.so $(BUILD_DIR)py_neuronet.o nnsim_pykernel.so -r build
