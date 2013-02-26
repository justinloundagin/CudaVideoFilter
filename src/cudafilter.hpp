#ifndef CUDAFILTER_H
#define CUDAFILTER_H

#include <cv.h>
#include "filter.hpp"

void cudaFilter(IplImage *image, Filter *filter);

#endif
