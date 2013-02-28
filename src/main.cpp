#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <cstdlib>
#include <stdarg.h>
#include <sys/time.h>
#include "cudafilter.hpp"

#define WINDOW_TITLE "Cuda Video Filter"
#define FPS_LIMIT 30

double difftimeval(const timeval *start, const timeval *end) {
   double ms = (end->tv_sec - start->tv_sec) * 1000.0 + 
               (end->tv_usec - start->tv_usec) / 1000.0;

   return ms < 0.0 ? 0.0 : ms;
}


IplImage *stitchImages(IplImage *images[], int numImages) {
    int w = ceil(numImages / 2.0);
    int h = 2.0;
    int size = 300;

    // Create a new 3 channel image
    IplImage *stitched = cvCreateImage( cvSize(100 + size*w, 60 + size*h), 8, 3 );

    // Loop for nArgs number of arguments
    for (int i = 0, m = 20, n = 20; i < numImages; i++, m += (20 + size)) {
        IplImage *img = images[i];
        if(img == 0) {
            cvReleaseImage(&stitched);
            break;
        }
        // Find whether height or width is greater in order to resize the image
        int max = (img->width > img->height)? img->width: img->height;

        // Find the scaling factor to resize the image
        float scale = (float) ( (float) max / size );

        // Used to Align the images
        if( i % w == 0 && m!= 20) {
            m = 20;
            n += 20 + size;
        }
        cvSetImageROI(stitched, cvRect(m, n, (int)( img->width/scale ), (int)( img->height/scale )));
        cvResize(img, stitched);
        cvResetImageROI(stitched);
    }
    return stitched;
}

char *computeFps(const char *fmt) {
   static char fps[256] = {0};
   static unsigned count = 0;
   static double elapsed = 0.0;
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

IplImage **createImages(int size, int width, int height, int depth, int channels) {
   IplImage **frames = (IplImage**)calloc(size, sizeof(IplImage*));
   
   for(int i=0; i<size; i++)
      frames[i] = cvCreateImage(cvSize(width, height), depth, channels);
   return frames;
}

void releaseImages(IplImage **images, int size) {
   while(size--) 
      cvReleaseImage(images++);
}

int main(int argc, char **argv) {
   if(argc < 2) {
      fprintf(stderr, "usage: cudafitler <filter> ...\n");
      exit(1);
   }

   cvNamedWindow(WINDOW_TITLE, CV_WINDOW_AUTOSIZE);
   CvCapture* capture = cvCaptureFromCAM(0);
   CvFont font;

   int width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
   int height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);

   Filter **filters = createFiltersFromFiles(argv + 1, argc - 1);
   IplImage *origin, **frames = createImages(argc - 1, width, height, 8 , 3); 

   cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.5, .5, 0, 1);

   while(cvWaitKey(5) != 27 && (origin = cvQueryFrame(capture)) != NULL) {
      for(int i=0; i<argc - 1; i++)
         cudaFilter(frames[i] = cvCloneImage(origin), filters[i]);

      /*
      IplImage *result = stitchImages(frames, argc - 1);
      cvPutText(result, computeFps("FPS: %d"), cvPoint(5, 15), &font, cvScalar(255, 255, 0));
      cvShowImage(WINDOW_TITLE, result);
      */
      IplImage *result = stitchImages(frames, argc - 1); 
      cvPutText(result, computeFps("FPS: %d"), cvPoint(5, 15), &font, cvScalar(255, 255, 0));
      cvShowImage(WINDOW_TITLE, result);
       cvReleaseImage(&result);
   }

   cvReleaseCapture(&capture);
   cvDestroyWindow(WINDOW_TITLE);
   releaseImages(frames, argc - 1);
   freeFilters(filters, argc - 1);
   return 0;
}
