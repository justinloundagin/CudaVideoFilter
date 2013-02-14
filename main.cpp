#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "cudafilter.hpp"

int main(int argc, char **argv) {
   cv::VideoCapture capture(argv[1]);
   cv::Mat frame;
   Filter filter;
   Image<unsigned char>image;

   initFilter(&filter, 3, 3);
   for(int i=0; i<filter.width * filter.height; i++)
      filter.data[i] = -5;
   filter.data[4] = 8;

   
    for(; cv::waitKey(10) != 27;) {
       capture >> frame;
       image.data = frame.data;
       image.width = frame.cols;
       image.height = frame.rows;

       cudaFilter<unsigned char>(filter, image);

       cv::imshow("Cuda Video Filter", frame);
    }
    return EXIT_SUCCESS;
}
