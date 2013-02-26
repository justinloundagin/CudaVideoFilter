CC = nvcc
CFLAGS = `pkg-config opencv --cflags --libs`
NVFLAGS= -arch=sm_30
SRC = src/filter.cpp src/cudafilter.cu src/main.cpp

cudafilter: $(SRC)
	$(CC)  $^ -o $@ $(NVFLAGS) $(CFLAGS)

clean:
	rm -rf *.o cudafilter
