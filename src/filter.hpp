#ifndef FILTER_H
#define FILTER_H

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

class Filter {
public:
   float factor;
   float bias;
   float *data;
   int rows; 
   int cols;

   HOST DEVICE Filter(char *path, float factor, float bias);
   HOST DEVICE Filter() {}
   HOST DEVICE float &at(int row, int col) {
   	return data[cols * row + col];
   }

   HOST DEVICE float *operator[](int row) {
   	return data +row * cols;
   }
};
#endif
