#ifndef CUDAFILTER_H
#define CUDAFILTER_H

#include <cv.h>
#include "filter.hpp"
#include "image.hpp"


class CudaFilter {
	Image image;
	Filter filter;

   void applyFilter();

public:
	CudaFilter(cv::Mat image, Filter filter);
	float operator() ();

   static Filter filterToDevice(Filter filter);
   static Image imageToDevice(Image image);
};

#endif
