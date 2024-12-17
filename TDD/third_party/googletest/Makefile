GTEST := googletest-release-1.10.0
#https://github.com/google/googletest/archive/refs/tags/release-1.10.0.tar.gz

export CXXFLAGS="-Wno-error=maybe-uninitialized"

all:
	tar zxf googletest-release-1.10.0.tar.gz 
	cmake -S $(GTEST)/ -B ./build/
	make -j -C build/

.PHONY: clean
clean:
	-rm -rf ./build/ ./googletest-release-1.10.0/
