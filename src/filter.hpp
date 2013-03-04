#ifndef FILTER_H
#define FILTER_H

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

struct Filter {
   float factor;
   float bias;
   float *data;
   int rows; 
   int cols;
};

inline HOST DEVICE float &filterElement(Filter filter, int row, int col) {
   return filter.data[filter.cols * row + col];
}

Filter *createFilter(int rows, int cols, float factor, float bias, float val = 0.0);
Filter *createFilterFromFile(char *path, float factor, float bias);
#endif
