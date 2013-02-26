#include "cudafilter.hpp"
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cmath>
#include <stdio.h>
#include <assert.h>

#define THREAD_DIM 32
#define CUDA_ERR_HANDLER(err) cudaErrorHandler(err, __FILE__, __LINE__)

#define BLUE 0
#define GREEN 1
#define RED 2

void cudaErrorHandler(cudaError_t err, const char *file, int line) {
   if(err != cudaSuccess) {
      fprintf(stderr, "%s on line %d: %s\n", file, line, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }
}

__device__ uchar &imageElement(IplImage *image, int row, int col, int ndx) {
   return CV_IMAGE_ELEM(image, uchar, row, col * image->nChannels + ndx);
}

__device__ void convolutionFilter(IplImage *image, IplImage *result, Filter *filter, int x, int y) {
   double red = 0.0, green = 0.0, blue = 0.0; 
   double factor = 1.0; 
   double bias = 0.0;

   //multiply every value of the filter with corresponding image pixel 
   for(int filterX = 0; filterX < filter->cols; filterX++) {
      for(int filterY = 0; filterY < filter->rows; filterY++) {
         int imageX = x - filter->cols / 2 + filterX;
         int imageY = y - filter->rows / 2 + filterY;
         if(imageX < 0 || imageX >= image->height ||
            imageY < 0 || imageY >= image->width)
            continue;
         
         double filterVal = filterElement(filter, filterX, filterY);
         blue  += imageElement(image, imageX, imageY, BLUE) * filterVal;
         green += imageElement(image, imageX, imageY, GREEN) * filterVal;
         red   += imageElement(image, imageX, imageY, RED) * filterVal;
      } 
   }

   //truncate values smaller than zero and larger than 255 
   imageElement(result, x, y, BLUE) = min(max(int(filter->factor * blue + filter->bias), 0), 255); 
   imageElement(result, x, y, RED) = min(max(int(filter->factor * red + filter->bias), 0), 255);
}

__device__ void grayFilter(IplImage *image, int row, int col) {
   double avg = 0.0;

   for(int i=0; i<3; i++)
      avg += imageElement(image, row, col, i);
   avg /= 3.0;

   for(int i=0; i<3; i++) 
      imageElement(image, row, col, i) = avg;
}


__global__ void filterKernal(IplImage image, IplImage result, Filter filter) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if(row < image.height && col < image.width) {
         convolutionFilter(&image, &result, &filter, row, col);
   }

}

void cudaFilter(IplImage *image, Filter *filter) {
   IplImage devImage, devResult;
   Filter devFilter;

   int blkDimX = ceil((double)image->width / (double) THREAD_DIM);
   int blkDimY = ceil((double)image->height / (double) THREAD_DIM);

   dim3 blkDim(blkDimX, blkDimY);
   dim3 thrDim(THREAD_DIM, THREAD_DIM);

   memcpy(&devFilter, filter, sizeof(Filter));
   CUDA_ERR_HANDLER(cudaMalloc(&devFilter.data, filter->rows * filter->cols * sizeof(double)));
   CUDA_ERR_HANDLER(cudaMemcpy(devFilter.data, filter->data, filter->rows * filter->cols * sizeof(double), cudaMemcpyHostToDevice));

   memcpy(&devImage, image, sizeof(IplImage)); 
   CUDA_ERR_HANDLER(cudaMalloc(&devImage.imageData, image->imageSize));
   CUDA_ERR_HANDLER(cudaMemcpy(devImage.imageData, image->imageData, image->imageSize, cudaMemcpyHostToDevice));

   memcpy(&devResult, image, sizeof(IplImage));
   CUDA_ERR_HANDLER(cudaMalloc(&devResult.imageData, image->imageSize));

   filterKernal<<<blkDim, thrDim>>>(devImage, devResult, devFilter);
   CUDA_ERR_HANDLER(cudaGetLastError());

   CUDA_ERR_HANDLER(cudaMemcpy(image->imageData, devResult.imageData, devResult.imageSize, cudaMemcpyDeviceToHost));
   CUDA_ERR_HANDLER(cudaFree(devImage.imageData));
   CUDA_ERR_HANDLER(cudaFree(devResult.imageData));
   CUDA_ERR_HANDLER(cudaFree(devFilter.data));
}
