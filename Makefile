CPP=g++
CPPFLAGS=-std=c++17 -O3 -march=native -fopenmp -I inc/eigen3
SONAME=-soname

ifeq ($(shell uname -s),Darwin)
	SONAME=-install_name
	CPP=g++-15
endif

all: SISN_mp simulation

SISN_mp:
	$(CPP) $(CPPFLAGS) -c -fPIC src/SISN_mp.cpp -o out/SISN_mp.o 
	$(CPP) $(CPPFLAGS) -lm -shared -Wl,$(SONAME),SISN_mp.so -o out/SISN_mp.so out/SISN_mp.o

simulation:
	$(CPP) $(CPPFLAGS) -c -fPIC src/simulation.cpp -o out/simulation.o 
	$(CPP) $(CPPFLAGS) -lm -shared -Wl,$(SONAME),simulation.so -o out/simulation.so out/simulation.o
