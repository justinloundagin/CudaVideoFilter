#include <opencv2/highgui/highgui.hpp>
#include <cv.h>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "cudafilter.hpp"

#define WINDOW_TITLE "Cuda Video Filter"
#define FPS_LIMIT 50

char *mapFile(const char *path, int *size, int flags) {
   int fd;
   struct stat fileStat;
   char *ret = NULL;

   if((fd = open(path, flags)) == -1);
   else if(fstat(fd, &fileStat) < 0);
   else ret = (char*)mmap(0, *size = fileStat.st_size, 
               PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
   return ret;
}

double difftimeval(const timeval *start, const timeval *end) {
   double ms = (end->tv_sec - start->tv_sec) * 1000.0 + 
               (end->tv_usec - start->tv_usec) / 1000.0;
   return ms < 0.0 ? 0.0 : ms;
}

IplImage *stitchImages(IplImage *images[], int size) {
   IplImage *image = cvCreateImage(cvSize(images[0]->width, size * images[0]->height), images[0]->depth, images[0]->nChannels);
   char *data = image->imageData;

   for(int i=0; i<size; i++) {
      memcpy(data, images[i]->imageData, images[i]->imageSize);
      data += images[i]->imageSize;
   }
   return image;
}

Filter *initFilters(int argc, char **argv) {
   Filter *filters = (Filter*)calloc(argc - 1, sizeof(Filter));

   int size = 0;
   for(int i=1; i<argc; i++) {
      char *buff = mapFile(argv[i], &size, O_RDONLY);
      if(buff == NULL) {
         fprintf(stderr, "Map Failed\n");
         break;
      }
      createFilterFromData(&filters[i-1], 1.0, 0.0, buff, size);
      munmap(buff, size);
   }
   return filters;
}


void computeFps(const char *fmt, char *fps) {
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
}

int main(int argc, char **argv) {
   cvNamedWindow(WINDOW_TITLE, CV_WINDOW_AUTOSIZE);
   CvCapture* capture = cvCaptureFromCAM(0);
   CvFont font;
   char fps[255];

   int numFilters = argc - 1;
   Filter *filters = initFilters(argc, argv);

   int camWidth = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
   int camHeight = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);
   IplImage **frames = (IplImage**)calloc(numFilters, sizeof(IplImage*));
   for(int i=0; i<numFilters; i++)
      frames[i] = cvCreateImage(cvSize(camWidth, camHeight), 8, 3);
   
   cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, 0.5, .5, 0, 1);

   while(cvWaitKey(5) != 27) {
      computeFps("FPS: %d", fps);
      IplImage *frame = cvQueryFrame(capture);
      if(frame == NULL)
         break;

      for(int i=0; i<numFilters; i++) {
         memcpy(frames[i]->imageData, frame->imageData, frame->imageSize);
         cudaFilter(frames[i], &filters[i]);
      }

      IplImage *result = stitchImages(frames, numFilters);
      cvPutText(result, fps, cvPoint(5, 15), &font, cvScalar(255, 255, 0));
      cvShowImage(WINDOW_TITLE, result);
      cvReleaseImage(&result);
   }
   cvReleaseCapture(&capture);
   cvDestroyWindow(WINDOW_TITLE);
   while(numFilters) 
      freeFilter(&filters[--numFilters]);

   return 0;
}
