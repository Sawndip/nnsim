CC=g++
NUMPY_INC_DIR=/usr/lib/python2.7/dist-packages/numpy/core/include
PY_INC_DIR=/usr/include/python2.7
SRC_DIR=../
# LFLAGS=-Wl,-O1 -Wl,-Bsymbolic-functions -Wl,-z,relro -fno-strict-aliasing \
# -DNDEBUG -g -fwrapv -O2 -Wall -Wstrict-prototypes -D_FORTIFY_SOURCE=2 -g \
# -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security

all: nnsim_pykernel.so

nnsim_pykernel.so: py_neuronet.o libkernel_api.so
	$(CC) -shared -pthread py_neuronet.o -L. -lkernel_api -o nnsim_pykernel.so -Wl,-rpath,.,-rpath,./Release

py_neuronet.o:
	$(CC) -c -fPIC -I$(NUMPY_INC_DIR) -I$(PY_INC_DIR) $(SRC_DIR)py_neuronet.cpp -o py_neuronet.o

libkernel_api.so: kernel_api.o
	$(CC) -shared kernel_api.o -o libkernel_api.so

kernel_api.o:
	$(CC) -c -fPIC -O3 $(SRC_DIR)kernel_api.cpp -o kernel_api.o

clean:
	rm kernel_api.o libkernel_api.so py_neuronet.o nnsim_pykernel.so
