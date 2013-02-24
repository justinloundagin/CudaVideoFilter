#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include "cudafilter.hpp"

#define WINDOW_TITLE "Cuda Video Filter"
#define FPS_LIMIT 30

double difftimeval(const timeval start, const timeval end) {
   return (end.tv_sec - start.tv_sec) * 1000.0 + 
          (end.tv_usec - start.tv_usec) / 1000.0;
}

void computeFps(const timeval start, const timeval end, const char *fmt, char *fps) {
   static unsigned count = 0;

   if(++count % FPS_LIMIT == 0) {
      sprintf(fps, fmt, (int)(count / (difftimeval(start, end) / 1000.0)));
   }
}

int main(int argc, char **argv) {
   cvNamedWindow(WINDOW_TITLE, CV_WINDOW_AUTOSIZE);
   CvCapture* capture = cvCaptureFromCAM(0);
   IplImage* frame;
   Filter filter;
   CvFont font;
   timeval start, end;
   char fps[255];

   createFilter(&filter, 5, 5, 0.0);
   filterElement(&filter, 2, 2) = 2;
   filterElement(&filter, 2, 1) = -1;
   filterElement(&filter, 2, 0) = -1;

   for(int i=0; i<filter.rows; i++) {
      for(int j=0; j<filter.cols; j++) {
         printf("%lf ", filterElement(&filter, i, j));
      }
      printf("\n");
   }

   cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.5, .5, 0, 1);

   gettimeofday(&start, NULL);
   while(cvWaitKey(10) != 27 && (frame = cvQueryFrame(capture)) != NULL) {
      cudaFilter(frame, &filter);
      gettimeofday(&end, NULL);
      computeFps(start, end, "FPS: %d", fps);
      cvPutText(frame, fps, cvPoint(5, 15), &font, cvScalar(255, 255, 0));
      cvShowImage(WINDOW_TITLE, frame);
   }
   cvReleaseCapture(&capture);
   cvDestroyWindow(WINDOW_TITLE);
   return 0;
}
