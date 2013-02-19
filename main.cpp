#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include "cudafilter.hpp"

#define WINDOW_TITLE "Cuda Video Filter"

int main(int argc, char **argv)
{
   cvNamedWindow(WINDOW_TITLE, CV_WINDOW_AUTOSIZE);
   CvCapture* capture = cvCaptureFromCAM(0);
   IplImage* frame;
   for(;;) {
      frame = cvQueryFrame(capture);
      if(!frame) 
         break;
      cudaFilter(frame);
      cvShowImage(WINDOW_TITLE, frame);
      char c = cvWaitKey(10);
      if(c == 27) 
         break;
   }
   cvReleaseCapture(&capture);
   cvDestroyWindow(WINDOW_TITLE);
   return 0;
}
