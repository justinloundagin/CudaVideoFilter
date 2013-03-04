#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <cstdlib>
#include <cmath>

#include <sys/time.h>
#include "cudafilter.hpp"
#include "imageutils.hpp"

#define WINDOW_TITLE "Cuda Video Filter"
#define FPS_LIMIT 15


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

void beginProcessLoop(CvCapture *capture, IplImage **frames, Filter **filters, int size) {
   IplImage *origin;
   CvFont font;

   cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.5, .5, 0, 1);

   while(cvWaitKey(5) != 27 && (origin = cvQueryFrame(capture)) != NULL) {
      for(int i=0; i<size; i++)
         cudaFilter(frames[i] = cvCloneImage(origin), filters[i]);

     // IplImage *result = stitchImages(frames, size);
   //   cvPutText(result, computeFps("FPS: %d"), cvPoint(5, 15), &font, cvScalar(255, 255, 0));
     // cvShowImage(WINDOW_TITLE, result);
      
     // IplImage *result = stitchImages(frames, argc - 1); 
      cvPutText(frames[0], computeFps("FPS: %d"), cvPoint(5, 15), &font, cvScalar(255, 255, 0));
      cvShowImage(WINDOW_TITLE, frames[0]);
      //cvReleaseImage(&result);
   }
}


int main(int argc, char **argv) {
   if(argc < 2) {
      fprintf(stderr, "usage: cudafitler <filter> ...\n");
      exit(1);
   }

   cvNamedWindow(WINDOW_TITLE, CV_WINDOW_AUTOSIZE);
   CvCapture* capture = cvCaptureFromCAM(0);
   Filter **filters = (Filter**)calloc(argc - 1, sizeof(Filter*)); //createFiltersFromFiles(argv + 1, argc - 1);
   IplImage **frames = (IplImage **) calloc(argc - 1, sizeof(IplImage*));

   for(int i=0; i < argc - 1; i++)
      filters[i] = createFilterFromFile(argv[i+1], 1.0, 0.0);

   //start video processing
   beginProcessLoop(capture, frames, filters, argc - 1);

   //clean up
   cvReleaseCapture(&capture);
   cvDestroyWindow(WINDOW_TITLE);

   for(int i=0; i<argc - 1; i++) {
      free(filters[i]);
      cvReleaseImage(&frames[i]);
   }

   free(filters);
   free(frames);

   return 0;
}
