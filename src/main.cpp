#include <iostream>
#include <limits>
#include <cstdlib>
#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include "cudafilter.hpp"

#define WINDOW_TITLE "Cuda Video Filter"
#define FPS_LIMIT 30

using namespace cv;
using namespace std;

int kerdim;
int limit = 5;
char *computeFps(float ms) {
   static char fps[256] = {0};
   static unsigned count = 0;
   static float elapsed = 0.0;

   elapsed += ms;

   if(++count % FPS_LIMIT == 0) {
      sprintf(fps, "FPS (%.2f)", 1000.0 * FPS_LIMIT / (float) elapsed);
      printf("%d\t%d\t%.2f\n", FPS_LIMIT, kerdim, elapsed);
      count = elapsed = 0;
      if(--limit == 0)
         exit(0);
   }
   return fps;
}


unsigned parseArgs(int argc, char **argv) {
   unsigned frameLimit = std::numeric_limits<unsigned>::max();
   if(argc < 2) {
      cerr<<"usage: cudafitler <filter> ...\n";
      exit(1);
   }
   else if(argc > 2)
      frameLimit = atoi(argv[2]);

   return frameLimit;
}


int main(int argc, char **argv) {
   parseArgs(argc, argv);
   VideoCapture capture(0);
   Mat frame;
   Filter filter(argv[1], 1.0, 0.0);
   kerdim = filter.rows;

   namedWindow(WINDOW_TITLE, CV_WINDOW_AUTOSIZE);

   while(waitKey(10) != 27) { 
      capture >> frame;
      int ms = CudaFilter(Image(frame), filter)();
      putText(frame, computeFps(ms), cvPoint(30,30), 
              FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(20,20,250), 1, CV_AA);
      imshow(WINDOW_TITLE, frame);
   }

   return 0;
}
