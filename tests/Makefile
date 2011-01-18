CXX = g++
CPPFLAGS = -I/usr/include/vtk-5.4 -I../eigen-2.0 -I../metric_tensor/include -I../include -I../smooth/include -I../functionals/include

CXXFLAGS = $(CPPFLAGS) -O3 -Wall -mtune=native -ffast-math -fexpensive-optimizations -Wno-deprecated -fopenmp

LIBS = -lvtkIO -lmetis

TEST_SRC = $(wildcard *.cpp)
TEST_BIN = $(patsubst %.cpp,bin/%,$(TEST_SRC))

.SUFFIXES: .cpp .o

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<

default: $(TEST_BIN)

bin/%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf *.o $(TEST_BIN)
