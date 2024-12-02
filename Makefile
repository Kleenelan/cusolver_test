
include makefile.src





OBJ_TEST := $(SRC_TEST:.cpp=.cpp.o)

EXE_TEST := $(OBJ_TEST:.cpp.o=.out)
#EXT_TEST := testing/testing_zgemm
all: libtest.a liblapacktest.a $(EXE_TEST)


OBJ_TOOL := $(SRC_TOOL:.cpp=.o)

OPT_FLAGS1 :=  -g -O0 -Wall -ggdb -fno-inline
CPP_FLAGS1 := $(OPT_FLAGS1) -fPIC -DNDEBUG -DADD_ -Wall -fopenmp -std=c++11
INC1 := -I/usr/local/cuda/include -I./include -I./testing -I./testing

%.o: %.cpp
	g++ $(CPP_FLAGS1)  $(INC1) -c -o $@  $<

libtest.a: $(OBJ_TOOL)
	ar cr $@ $^
	ranlib $@



#########################################################################################

OBJ_FORT := $(SRC_FORT:.f=.f.o)
OPT_FORT_FLAGS := -g -O0 -Wall -ggdb -fno-inline
FORT_FLAGS := $(OPT_FORT_FLAGS) -fPIC -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument

%.f.o: %.f
	gfortran -c -o $@ $<

liblapacktest.a: $(OBJ_FORT)
	ar cr $@ $^
	ranlib $@

#################




OPT_FLAGS_TEST := -g -O0 -Wall -ggdb -fno-inline
CPP_FLAGS_TEST := $(OPT_FLAGS_TEST) -fPIC -DNDEBUG -DADD_ -Wall -fopenmp -std=c++11
INC_TEST :=  -I/usr/local/cuda/include -I./include -I./testing
LD_FLAGS_TEST := -Wl,-rpath,/home/hipper/ex_magma/testSys_magma/tmp1/magma/lib -L. \
-ltest -L/home/hipper/ex_magma/testSys_magma/tmp1/magma/lib -lmagma -L./testing/lin -llapacktest \
-L/home/hipper/ex_magma/testSys_magma/OpenBLAS/local/lib -L/usr/local/cuda/lib64 -lopenblas -lcublas -lcusparse -lcudart -lcudadevrt -lcublas -lcudart

%.cpp.o: %.cpp
	g++ $(CPP_FLAGS_TEST) $(INC_TEST) -c -o $@ $<

%.out: %.cpp.o libtest.a liblapacktest.a
	g++ -fPIC -fopenmp -o $@ $^ $(LD_FLAGS_TEST)







.PHONY: clean
clean:
	-rm -rf libtest.a $(OBJ_TOOL) liblapacktest.a $(OBJ_FORT) $(EXE_TEST) $(OBJ_TEST)

