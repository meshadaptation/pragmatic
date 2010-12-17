SHELL = /bin/bash

CXX = g++

CPPFLAGS = -I/usr/include/vtk-5.4 -I../eigen-2.0/ -I../metric_tensor/include/

CXXFLAGS = $(CPPFLAGS) -g -O2 -Wno-deprecated

LIBS = -lvtkIO

TEST_BINARIES = test_hessian

.SUFFIXES: .cpp .o

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<

default: $(TEST_BINARIES)

test_hessian: test_hessian.cpp
	$(CXX) $(CXXFLAGS) -o test_hessian test_hessian.cpp $(LIBS)

clean:
	rm -rf *.o $(TEST_BINARIES)