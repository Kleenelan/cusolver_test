
SW_HOME 	?= /usr/
CUDADIR 	:= $(SW_HOME)/local/cuda
GPU_ARCH 	:= -arch=sm_75
DEVCC 		:= /usr/local/cuda/bin/nvcc
CXX   		:= g++
DEVCC_FLAGS := -Xcompiler -fPIC -DNDEBUG -DADD_
INC 	:= -I/usr/local/cuda/include -I../control -I../include


include makefile.blas.src
OBJ_ICLABLAS_CU := $(SRC_ICLABLAS_CU:.cu=.cu.o)

all: $(OBJ_ICLABLAS_CU)

%.cu.o: %.cu
	$(DEVCC) $(DEVCC_FLAGS) $< -c -o $@ $(INC)

.PHONEY: clean
clean:
	-rm -rf $(OBJ_ICLABLAS_CU)