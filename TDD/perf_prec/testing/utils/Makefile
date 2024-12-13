
#SW_HOME ?= $(HOME)/sw_home
#CUDADIR := $(SW_HOME)/local/corex
SW_HOME ?= /usr/
CUDADIR := $(SW_HOME)/local/cuda



include makefile.utl.src
LIB_STATIC := libutils.a
OBJ_LIB := $(SRC_UTILS:.cpp=.o)

all:  $(LIB_STATIC)

PROJ_DIR ?= $(shell pwd)/../..
#OPT_FLAGS := -O3
OPT_FLAGS := -g -O0 -Wall -ggdb -fno-inline
C_FLAGS := $(OPT_FLAGS) -fPIC -DNDEBUG -DADD_ -Wall -fopenmp
CPP_FLAGS := $(OPT_FLAGS) -fPIC -DNDEBUG -DADD_ -Wall -fopenmp -std=c++11

INC += \
-I$(CUDADIR)/include \
-I$(PROJ_DIR)/control \
-I$(PROJ_DIR)/include \
-I$(PROJ_DIR)/testing/utils \
-I$(PROJ_DIR)/interface_cuda \
-I$(PROJ_DIR)/math_tool

%.o: %.cpp
	g++ $(CPP_FLAGS)  $(INC) -c -o $@  $<

$(LIB_STATIC): $(OBJ_LIB)
	ar cr $@ $^
	ranlib $@

.PHONY: clean
clean:
	-rm -rf $(LIB_STATIC) $(OBJ_LIB)
