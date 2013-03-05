#ifndef CUDAFILTER_H
#define CUDAFILTER_H

#include <cv.h>
#include "filter.hpp"

struct Image {
   char *data;
   int width; 
   int height;
   int widthStep;
   int nChannels;
   int size;
};

class CudaFilter {
	Image image;
	Filter filter;

   Filter filterToDevice(Filter filter);
   Image imageToDevice(Image image);

public:
	CudaFilter(cv::Mat image, Filter filter);
	void operator() ();
};

#endif
