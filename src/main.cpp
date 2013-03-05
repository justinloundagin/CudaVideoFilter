#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include "cudafilter.hpp"
#include "imageutils.hpp"

#define WINDOW_TITLE "Cuda Video Filter"
#define FPS_LIMIT 15

using namespace cv;
using namespace std;


float difftimeval(const timeval *start, const timeval *end) {
   float ms = (end->tv_sec - start->tv_sec) * 1000.0 + 
               (end->tv_usec - start->tv_usec) / 1000.0;

   return ms < 0.0 ? 0.0 : ms;
}

char *computeFps(const char *fmt) {
   static char fps[256] = {0};
   static unsigned count = 0;
   static float elapsed = 0.0;
   static timeval start;
   static timeval end;

   gettimeofday(&end, NULL);
   elapsed += difftimeval(&start, &end);

   if(++count % FPS_LIMIT == 0) {
      sprintf(fps, fmt, (int)(count / (elapsed / 1000.0)));
      count = elapsed = 0;
   }
   gettimeofday(&start, NULL);
   return fps;
}


int main(int argc, char **argv) {
   if(argc < 2) {
      cerr<<"usage: cudafitler <filter> ...\n";
      exit(1);
   }


   VideoCapture capture(0);
   Mat frame;
   Filter filter(argv[1], 1.0, 0.0);

   namedWindow(WINDOW_TITLE, CV_WINDOW_AUTOSIZE);

   while(waitKey(10) != 27) { 
      capture >> frame;
      CudaFilter(frame, filter)();
      imshow(WINDOW_TITLE, frame);
   }

   return 0;
}
