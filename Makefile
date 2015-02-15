CPU_STACKER_PATH=/home/lindroos/stacker
CPU_STACKER_LIBS=-L$(CPU_STACKER_PATH)/stacker_clib -lstacker -lcuda -lcudart

CASACORE_INCLUDE=-I/astro/include/casacore
CASACORE_LIBPATH=/astro/lib
CASALIBS=-L$(CASACORE_LIBPATH) -lcasa_coordinates -lcasa_tables -lcasa_ms  -lcasa_measures -lcasa_images -lcasa_meas -lcasa_casa -gfortran -llapack -lblas -lwcs -lcfitsio -ldl
LDFLAGS=$(CPU_STACKER_LIBS) $(CASALIBS)

CPPFLAGS=-I$(CPU_STACKER_PATH)/stacker_clib $(CASACORE_INCLUDE) -Xcompiler -fPIC

all: grid.o
	g++ -o grid grid.o $(LDFLAGS)

shared: grid.o
	g++ -o libgrid.so -shared -Wl,-soname,libgrid.so grid.o $(LDFLAGS)


grid.o: grid.cu
	nvcc -arch sm_20 -c $(CPPFLAGS) $(DEBUG) -g grid.cu

# stack.o: stack.cu
# 	nvcc -c $(CPPFLAGS) $(DEBUG) -g stack.cu

clean:
	rm -f *.o app

