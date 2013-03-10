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

 int getOptions(int argc, char **argv) {
   int options = 0x00;

   for(int i=4; i<argc; i++) {
      if(!strcmp(argv[i], "-gray")) {
         options |= CV_RGB2GRAY;
      }
   }
   return options;
}

Mat applyOptions(Mat &image, int options) {
   Mat cvtImg = image.clone();
   cvtColor(image, cvtImg, options);
   return cvtImg;
}

int main(int argc, char **argv) {
   if(argc < 4) {
      cerr<<"usage: cudafitler <filter> <factor> <bias>\n";
      exit(1);
   }

   VideoCapture capture(0);
   Mat frame;
   Filter filter(argv[1], atof(argv[2]), atof(argv[3]));
   int options = getOptions(argc, argv);

   namedWindow(WINDOW_TITLE, CV_WINDOW_AUTOSIZE);

   while(waitKey(10) != 27) { 
      capture >> frame;
      int ms = CudaFilter(Image(frame), filter)();
      if(options)
         frame = applyOptions(frame, options);

      putText(frame, computeFps(ms), cvPoint(30,30), 
              FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(20,20,250), 1, CV_AA);
      imshow(WINDOW_TITLE, frame);
   }

   return 0;
}
