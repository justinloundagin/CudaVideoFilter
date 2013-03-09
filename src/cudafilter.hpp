#ifndef CUDAFILTER_H
#define CUDAFILTER_H

#include <cv.h>
#include <vector>
#include "filter.hpp"
#include "image.hpp"

class CudaFilter {
	Image image, devImage, devResult;
	Filter devFilter;

   void applyFilter();

public:
	CudaFilter(Image image, Filter filter);
   ~CudaFilter();

	float operator()();
   void toDevice(void **dev, void *host, int bytes);
   void toHost(void *host, void *dev, int bytes);
};

#endif
