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
   double factor;
   double bias;
   double *data;
   int rows; 
   int cols;
};

inline HOST DEVICE double &filterElement(Filter *filter, int row, int col) {
   return filter->data[filter->cols * row + col];
}

Filter *createFilter(int rows, int cols, double factor, double bias, double val = 0.0);
Filter *createFilterFromFile(char *path, double factor, double bias);
Filter **createFiltersFromFiles(char **paths, int size);
void freeFilter(Filter *filter);
void freeFilters(Filter **filters, int size);

#endif
