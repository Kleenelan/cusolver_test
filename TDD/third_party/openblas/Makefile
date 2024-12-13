all: OpenBLAS-0.3.27/libopenblas.so

v0.3.27.tar.gz:
	wget https://github.com/OpenMathLib/OpenBLAS/archive/refs/tags/v0.3.27.tar.gz

OpenBLAS-0.3.27/version.h: v0.3.27.tar.gz
	tar zxf v0.3.27.tar.gz 

OpenBLAS-0.3.27/libopenblas.so: OpenBLAS-0.3.27/version.h
	make -C OpenBLAS-0.3.27 FC=gfortran -j


install:
	make -C OpenBLAS-0.3.27 install PREFIX=../local/

clean:
	-rm -rf ./local/ ./OpenBLAS-0.3.27
