#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <iostream>
#include <cstdlib>
#include "cudafilter.hpp"

#define WINDOW_TITLE "Cuda Video Filter"
#define FPS_LIMIT 15

using namespace cv;
using namespace std;

char *computeFps(float ms) {
   static char fps[256] = {0};
   static unsigned count = 0;
   static float elapsed = 0.0;

   elapsed += ms;

   if(++count % FPS_LIMIT == 0) {
      sprintf(fps, "FPS (%.2f)", 1000.0 * FPS_LIMIT / (float) elapsed);
      count = elapsed = 0;
   }
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
      int ms = CudaFilter(frame, filter)();
      putText(frame, computeFps(ms), cvPoint(30,30), 
              FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(20,20,250), 1, CV_AA);
      imshow(WINDOW_TITLE, frame);
   }

   return 0;
}
