CC=g++
NUMPY_INC_DIR=/usr/lib/python2.7/dist-packages/numpy/core/include
PY_INC_DIR=/usr/include/python2.7
SRC_DIR=./
BUILD_DIR=./Release/
# LFLAGS=-Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing \
# -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g \
# -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security

all: nnsim_pykernel.so

nnsim_pykernel.so: $(BUILD_DIR)py_neuronet.o $(BUILD_DIR)libkernel_api.so
	$(CC) -shared -pthread $(BUILD_DIR)py_neuronet.o -L. -L$(BUILD_DIR) -lkernel_api -o nnsim_pykernel.so -Wl,-rpath,.,-rpath,$(BUILD_DIR)

$(BUILD_DIR)py_neuronet.o:
	$(CC) -c -fPIC  -I$(NUMPY_INC_DIR) -I$(PY_INC_DIR) $(SRC_DIR)py_neuronet.cpp -Wwrite-strings -o $(BUILD_DIR)py_neuronet.o

$(BUILD_DIR)libkernel_api.so: $(BUILD_DIR)kernel_api.o
	$(CC) -shared $(BUILD_DIR)kernel_api.o -o $(BUILD_DIR)libkernel_api.so

$(BUILD_DIR)kernel_api.o:
	$(CC) -c -fPIC -O3 $(SRC_DIR)kernel_api.cpp -o $(BUILD_DIR)kernel_api.o

clean:
	rm $(BUILD_DIR)kernel_api.o $(BUILD_DIR)libkernel_api.so $(BUILD_DIR)py_neuronet.o nnsim_pykernel.so
