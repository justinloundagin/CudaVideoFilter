CC = g++
NVCC = nvcc
CFLAGS = `pkg-config opencv --cflags --libs`
NVFLAGS= -arch=sm_30

all:
	make cudafilter

cudafilter: cudafilter.cu main.cpp
	$(NVCC)  cudafilter.cu main.cpp -o $@ $(NVFLAGS) $(CFLAGS)

clean:
	rm -rf *.o cudafilter
