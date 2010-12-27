SHELL = /bin/bash

CXX = g++

CPPFLAGS = -I/usr/include/vtk-5.4 -I../eigen-2.0 -I../metric_tensor/include -I../include

CXXFLAGS = $(CPPFLAGS) -g -O2 -Wno-deprecated -fopenmp

LIBS = -lvtkIO -lmetis

TEST_BINARIES = test_hessian_2d test_hessian_3d

.SUFFIXES: .cpp .o

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<

default: $(TEST_BINARIES)

test_hessian_2d: test_hessian_2d.cpp
	$(CXX) $(CXXFLAGS) -o test_hessian_2d test_hessian_2d.cpp $(LIBS)

test_hessian_3d: test_hessian_3d.cpp
	$(CXX) $(CXXFLAGS) -o test_hessian_3d test_hessian_3d.cpp $(LIBS)

clean:
	rm -rf *.o $(TEST_BINARIES)
