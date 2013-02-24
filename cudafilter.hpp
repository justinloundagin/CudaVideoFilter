#ifndef CUDAFILTER_H
#define CUDAFILTER_H

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

#include <cv.h>

struct Filter {
   double *data;
   int rows; 
   int cols;
};

inline HOST DEVICE double &filterElement(Filter *filter, int row, int col) {
   return filter->data[filter->cols * row + col];
}

inline HOST DEVICE uchar &imageElement(IplImage *image, int row, int col, int ndx) {
   return CV_IMAGE_ELEM(image, uchar, row, col * image->nChannels + ndx);
}

void createFilter(Filter *filter, int rows, int cols, double val = 0.0);
void cudaFilter(IplImage *image, Filter *filter);

#endif
