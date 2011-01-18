CXX = g++
CPPFLAGS = -I/usr/include/vtk-5.4 -I../eigen-2.0 -I../metric_tensor/include -I../include -I../smooth/include -I../functionals/include

CXXFLAGS = $(CPPFLAGS) -g -O3 -Wall -mtune=native -ffast-math -fexpensive-optimizations -Wno-deprecated -fopenmp

LIBS = -lvtkIO -lmetis

TEST_SRC = $(wildcard src/*.cpp)
TEST_BIN = $(patsubst src/%.cpp,bin/%,$(TEST_SRC))

.SUFFIXES: .cpp .o

.cpp.o:
	$(CXX) $(CXXFLAGS) -c $<

default: test

test: $(TEST_BIN)
	python ./unittest bin

bin/%: src/%.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf *.o $(TEST_BIN)
