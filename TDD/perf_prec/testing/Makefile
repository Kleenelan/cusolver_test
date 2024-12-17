#SW_HOME ?= $(HOME)/sw_home
#CUDADIR := $(SW_HOME)/local/corex
SW_HOME ?= /usr/
CUDADIR := $(SW_HOME)/local/cuda


include makefile.test.src
LIB_STATIC := libtest.a
OBJ_LIB := $(SRC_TEST:.cpp=.o)
EXE := $(SRC_TEST:.cpp=.out)

all:  $(LIB_STATIC) $(EXE)

#OPT_FLAGS := -O3
PROJ_DIR ?= $(shell pwd)/..
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

LD_FLAGS :=


%.out: %.o
	g++ $(C_FLAGS) $< -o $@ $(LD_FLAGS)

.PHONY: clean
clean:
	-rm -rf $(LIB_STATIC) $(OBJ_LIB) $(EXE)







