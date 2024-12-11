
#SW_HOME ?= $(HOME)/sw_home
#CUDADIR := $(SW_HOME)/local/corex
SW_HOME ?= /usr/
CUDADIR := $(SW_HOME)/local/cuda

include makefile.src

OBJ_LIB := $(SRC_LIB:.cpp=.lib.o)

OBJ_TEST := $(SRC_TEST:.cpp=.cpp.o)
OBJ_INTERFACE_CUDA := $(SRC_INTERFACE_CUDA:.cpp=.o)

EXE_TEST := $(OBJ_TEST:.cpp.o=.out)
#EXT_TEST := testing/testing_zgemm
all:  $(EXE_TEST)
LIBS := \
	libtest.a \
	libinterface_cuda.a \
	libwrapper.a

#OPENBLAS :
#	make -C ../third_party/openblas/

OBJ_TOOL := $(SRC_TOOL:.cpp=.o)

OPT_FLAGS1 :=  -g -O0 -Wall -ggdb -fno-inline
C_FLAGS1 := $(OPT_FLAGS1) -fPIC -DNDEBUG -DADD_ -Wall -fopenmp

#OPT_FLAGS1 :=  -g -O0 -Wall -ggdb -fno-inline
CPP_FLAGS1 := $(OPT_FLAGS1) -fPIC -DNDEBUG -DADD_ -Wall -fopenmp -std=c++11
INC1 := -I$(CUDADIR)/include -I./include -I./testing/utils -I./control

%.o: %.cpp
	g++ $(CPP_FLAGS1)  $(INC1) -c -o $@  $<

%.o: %.c
	gcc $(CPP_FLAGS1)  $(INC1) -c -o $@  $<

libtest.a: $(OBJ_TOOL)
	ar cr $@ $^
	ranlib $@

libinterface_cuda.a: $(OBJ_INTERFACE_CUDA)
	ar cr $@ $^
	ranlib $@

#########################################################################################
#########################################################################################

OPT_FLAGS_LIB := -g -O0 -Wall -ggdb -fno-inline
CPP_FLAGS_LIB := $(OPT_FLAGS_LIB) -fPIC -DNDEBUG -DADD_ -Wall -fopenmp -std=c++11
INC_LIB :=  -I$(CUDADIR)/include -I./include -I./testing/utils -I./control/

#LD_FLAGS_LIB := -Wl,-rpath,//OpenBLAS/local/lib \
#-L//OpenBLAS/local/lib -lopenblas \
#-L$(CUDADIR)/lib64 -lcublas -lcusparse -lcusolver -lcudart -lcudadevrt

LD_FLAGS_LIB := -lopenblas \
	-L$(CUDADIR)/lib64 -lcublas -lcusparse -lcusolver -lcudart# -lcudadevrt

%.lib.o: %.cpp
	g++ $(CPP_FLAGS_TEST) $(INC_TEST) -DICLA_HAVE_CUDA -c -o $@ $<

libwrapper.a: $(OBJ_LIB)
	ar cr $@ $^
	ranlib $@

#$(SRC_LIB:.cpp=.lib.o)

#########################################################################################
#########################################################################################

OPT_FLAGS_TEST := -g -O0 -Wall -ggdb -fno-inline
CPP_FLAGS_TEST := $(OPT_FLAGS_TEST) -fPIC -DNDEBUG -DADD_ -Wall -fopenmp -std=c++11
INC_TEST :=  -I$(CUDADIR)/include -I./include -I./testing/utils -I./control/

#LD_FLAGS_TEST := -Wl,-rpath,/home/hipper/ex_icla/testSys_icla/tmp1/icla/lib
LD_FLAGS_TEST := -Wl,-rpath,../third_party/openblas/OpenBLAS-0.3.27  \
-L../third_party/openblas/OpenBLAS-0.3.27 -lopenblas \
-L$(CUDADIR)/lib64 -lcublas -lcusparse -lcusolver -lcudart -lgfortran -lgtest -lgtest_main# -lcudadevrt
# \-L/home/hipper/ex_icla/testSys_icla/tmp1/icla/lib  -licla

%.cpp.o: %.cpp
	g++ $(CPP_FLAGS_TEST) $(INC_TEST) -c -o $@ $<

%.out: %.cpp.o $(LIBS)
	make -C ../third_party/openblas/
	g++ -fPIC -fopenmp -o $@ $^ $(LD_FLAGS_TEST)

#########################################################################################
#########################################################################################

.PHONY: clean
clean:
	-rm -rf $(OBJ_TOOL) $(LIBS) $(EXE_TEST) $(OBJ_TEST) $(OBJ_INTERFACE_CUDA) $(OBJ_LIB)