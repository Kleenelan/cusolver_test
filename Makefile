
include makefile.src

OBJ_TEST := $(SRC_TEST:.cpp=.cpp.o)
OBJ_INTERFACE_CUDA := $(SRC_INTERFACE_CUDA:.cpp=.o)

EXE_TEST := $(OBJ_TEST:.cpp.o=.out)
#EXT_TEST := testing/testing_zgemm
all: libtest.a $(EXE_TEST) libinterface_cuda.a


OBJ_TOOL := $(SRC_TOOL:.cpp=.o)

OPT_FLAGS1 :=  -g -O0 -Wall -ggdb -fno-inline
C_FLAGS1 := $(OPT_FLAGS1) -fPIC -DNDEBUG -DADD_ -Wall -fopenmp

#OPT_FLAGS1 :=  -g -O0 -Wall -ggdb -fno-inline
CPP_FLAGS1 := $(OPT_FLAGS1) -fPIC -DNDEBUG -DADD_ -Wall -fopenmp -std=c++11
INC1 := -I/usr/local/cuda/include -I./include -I./testing/utils -I./control

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




OPT_FLAGS_TEST := -g -O0 -Wall -ggdb -fno-inline
CPP_FLAGS_TEST := $(OPT_FLAGS_TEST) -fPIC -DNDEBUG -DADD_ -Wall -fopenmp -std=c++11
INC_TEST :=  -I/usr/local/cuda/include -I./include -I./testing/utils
#LD_FLAGS_TEST := -Wl,-rpath,/home/hipper/ex_icla/testSys_icla/tmp1/icla/lib
LD_FLAGS_TEST := -Wl,-rpath,/home/hipper/ex_icla/testSys_icla/OpenBLAS/local/lib \
-L/home/hipper/ex_icla/testSys_icla/OpenBLAS/local/lib -lopenblas \
-L/usr/local/cuda/lib64 -lcublas -lcusparse -lcusolver -lcudart -lcudadevrt
# \-L/home/hipper/ex_icla/testSys_icla/tmp1/icla/lib  -licla

%.cpp.o: %.cpp
	g++ $(CPP_FLAGS_TEST) $(INC_TEST) -c -o $@ $<

%.out: %.cpp.o libtest.a libinterface_cuda.a
	g++ -fPIC -fopenmp -o $@ $^ $(LD_FLAGS_TEST)







.PHONY: clean
clean:
	-rm -rf $(OBJ_TOOL) libinterface_cuda.a libtest.a $(OBJ_FORT) $(EXE_TEST) $(OBJ_TEST) $(OBJ_INTERFACE_CUDA)

