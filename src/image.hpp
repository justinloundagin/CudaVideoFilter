#ifndef IMAGE_H
#define IMAGE_H
#include <cv.h>

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#else
#define HOST
#define DEVICE
#endif

enum BGR {BLUE, GREEN, RED};

class Image {
   
public:
   char *data;
   int width; 
   int height;
   int widthStep;
   int nChannels;
   int size;

   HOST DEVICE Image() {}
   HOST DEVICE Image(cv::Mat image) {
      IplImage tmp = image;
      width = tmp.width;
      height = tmp.height;
      nChannels = tmp.nChannels;
      widthStep = tmp.widthStep;
      size = tmp.imageSize;
      data = tmp.imageData;
   }

   HOST DEVICE Image(const Image &image) {
      width = image.width;
      height = image.height;
      nChannels = image.nChannels;
      widthStep = image.widthStep;
      size = image.size;
      data = image.data;
   }

   HOST DEVICE uchar &at(int row, int col, BGR color) {
      return ((uchar*)(data + widthStep * (row)))[col * nChannels + color];
   }  
};

#endif