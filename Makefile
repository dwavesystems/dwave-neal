
CXX=g++
SSE_EXT=-msse4.2
CXX_FLAGS=$(SSE_TEXT) -Wall -O3 -g -funroll-loops --std=c++11 -fPIC

SRC=$(wildcard src/*.cc)
OBJ=$(SRC:.cc=.o)

PYTHON_HEADERS=/usr/include/python2.7/
PYTHON_LIB=/usr/lib/libpython2.7.so
PYTHON_FLAGS=-Isrc -I$(PYTHON_HEADERS) -L$(PYTHON_LIB)

PYSRC=pysrc/python_bindings.cc
PYOBJ=$(PYSRC:.cc=.o)

TARGET=dwave_sa_chi

all:	$(TARGET)

python: python_wrapper python_bindings

$(PYOBJ): $(PYSRC)
	$(CXX) $(CXX_FLAGS) $(PYTHON_FLAGS) -c $< -o $@

%.o:	%.cc
	$(CXX) $(CXX_FLAGS) -c $< -o $@

python_bindings: $(OBJ) $(PYOBJ)
	@(mkdir -p site-packages)
	$(CXX) -shared $(PYTHON_FLAGS) $(OBJ) $(PYOBJ) -o site-packages/_$(TARGET).so -lpython2.7
	@(rm -f $(OBJ) $(PYOBJ))
	@(cp pysrc/$(TARGET).py site-packages/)

python_wrapper:
	mkdir -p pysrc
	swig -c++ -python -outdir pysrc -Isrc/ -o pysrc/python_bindings.cc interfaces/$(TARGET).i

dwave_sa_chi: $(OBJ)
	$(CXX) $(CXX_FLAGS) $(OBJ) -o $(TARGET)

clean:
	@(rm -f $(OBJ) $(TARGET))
	@(rm -rf pysrc site-packages)
