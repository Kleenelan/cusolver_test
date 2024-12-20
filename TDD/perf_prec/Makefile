
SW_HOME ?= /usr
### interface variables which would be got from sw_home/Makefile #########################################
DEVCC 			?= $(SW_HOME)/local/cuda/bin/nvcc
TARGET_CUDA_DIR ?= $(SW_HOME)/local/cuda
USR_DIR			?= $(SW_HOME)
##########################################################################################################

CUDADIR := $(TARGET_CUDA_DIR)

include makefile.src

PROJ_DIR := $(shell pwd)


SRC_TESTING_EXE := $(SRC_TESTING)

SRC_FRAME := \
$(SRC_LIB)    \
$(SRC_UTILS)   \
$(SRC_CONTROL)  \
$(SRC_MATH_TOOL) \
$(SRC_INTERFACE_CUDA)

OBJ_TESTING_LIN_FORT := $(SRC_TESTING_LIN_FORT:.f=.f.o)
OBJ_FRAME := $(SRC_FRAME:.cpp=.cpp.o) $(OBJ_TESTING_LIN_FORT)
OBJ_TESTING := $(SRC_TESTING:.cpp=.cpp.o)

EXE_TESTING := $(SRC_TESTING_EXE:.cpp=.out)

LIB_SHARED_TDD   := libtdd.so
LIB_SHARED_FRAME := libframe.so

all: $(LIB_SHARED_FRAME) $(LIB_SHARED_TDD)
app: $(EXE_TESTING)

#################################################################################

GPU_ARCH 	:= -arch=sm_75
DEVCC 		:= /usr/local/cuda/bin/nvcc
CXX   		:= g++
DEVCC_FLAGS := -Xcompiler -fPIC -DNDEBUG -DADD_
INC_CU 	:= -I/usr/local/cuda/include -I$(PROJ_DIR)/control -I$(PROJ_DIR)/include


#include makefile.blas.src
OBJ_ICLABLAS_CU := $(SRC_ICLABLAS_CU:.cu=.cu.o)

OBJ_FRAME += $(OBJ_ICLABLAS_CU)

#all: $(OBJ_ICLABLAS_CU)

%.cu.o: %.cu
	$(DEVCC) $(DEVCC_FLAGS) $< -c -o $@ $(INC_CU)


# -fsanitize=leak -fsanitize=address -fsanitize=undefined  -fsanitize=thread
OPT_FLAGS :=  -g -O0 -Wall -ggdb -fno-inline
#OPT_FLAGS := -O3
CPP_FLAGS := $(OPT_FLAGS) -fPIC -DNDEBUG -DADD_ -Wall -fopenmp -std=c++11 #-fsanitize=undefined# -fsanitize=thread
FORT        ?= gfortran
FFLAGS       = $(OPT_FLAGS) -fPIC -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument
INC := \
-I$(CUDADIR)/include \
-I$(PROJ_DIR)/include \
-I$(PROJ_DIR)/testing/utils \
-I$(PROJ_DIR)/control \
-I$(PROJ_DIR)/testing \
-I$(PROJ_DIR)/interface_cuda

%.cpp.o: %.cpp
	g++ $(CPP_FLAGS)  $(INC) -c -o $@  $<

%.f.o: %.f
	gfortran $(FFLAGS) -c -o $@ $<


LD_FLAGS := -L$(CUDADIR)/lib64 -lcudart -lcublas -lcublasLt -lcusparse -lcusolver -lgfortran \
			-L../third_party/openblas/OpenBLAS-0.3.27 -lopenblas -Wl,-rpath,$(PROJ_DIR)/../third_party/openblas/OpenBLAS-0.3.27

$(LIB_SHARED_TDD): $(OBJ_FRAME) $(OBJ_TESTING)
	make -C $(PROJ_DIR)/../third_party/openblas
	g++ -shared -fPIC $^ -o $@ $(LD_FLAGS)

$(LIB_SHARED_FRAME): $(OBJ_FRAME)
	g++ -shared -fPIC $^ -o $@ $(LD_FLAGS)

INC_EXE := \
$(INC) \
$(PROJ_DIR)/../third_party/googletest/googletest-release-1.10.0/googletest/include

LD_FLAGS_EXE := $(LD_FLAGS) \
			-L$(PROJ_DIR) -lframe -Wl,-rpath,$(PROJ_DIR) \
			-L$(PROJ_DIR)/../third_party/googletest/build/lib  -lgtest -lgtest_main

%.out: %.cpp
	g++ $(CPP_FLAGS) $< -o $@ $(INC) $(LD_FLAGS_EXE)
# -DBUILD_MAIN_





#########################################################################################
#########################################################################################


.PHONY: clean
clean:
	-rm -rf $(EXE_TEST)  $(OBJ_TESTING) $(OBJ_FRAME) $(LIB_SHARED_TDD) $(LIB_SHARED_FRAME)

.PHONY: clean-app
clean-app:
	-rm -rf $(EXE_TESTING)

.PHONY: clean-openblas
clean-openblas:
	-rm -rf $(PROJ_DIR)/../third_party/openblas/OpenBLAS-0.3.27
