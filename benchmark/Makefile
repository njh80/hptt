CXX_FLAGS=-O3 -std=c++11

ifeq ($(CXX),icpc)
CXX_FLAGS += -qopenmp -xhost 
else
ifeq ($(CXX),g++)
CXX_FLAGS += -fopenmp -march=native 
else
ifeq ($(CXX),clang++)
CXX_FLAGS += -fopenmp
endif
endif
endif

scalar: 
	${MAKE} clean 
	${MAKE} scalar2

scalar2: all

SRC=benchmark.cpp reference.cpp
OBJ=$(SRC:.cpp=.o)

INCLUDE_PATH=-I../src/ 
LIB_PATH=-L../lib/
LIBS=-lhptt

all: ${OBJ}
	${CXX} ${OBJ} ${LIB_PATH} ${LIBS} ${CXX_FLAGS} -o benchmark.exe

LIB_PATH+= ${OPENMP_LIB_PATH}
LIBS+= -lomp

%.o: %.cpp
	${CXX} ${LIB_PATH} ${LIBS} ${CXX_FLAGS} ${INCLUDE_PATH} -c $< -o $@

clean:
	rm -rf *.o benchmark.exe
