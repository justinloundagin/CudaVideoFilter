#include "cudafilter.hpp"
#include <stdlib.h>
#include <cuda.h>
#include <cmath>
#include <stdio.h>

#define THREAD_DIM 32
#define CUDA_ERR_HANDLER(err) cudaErrorHandler(err, __FILE__, __LINE__)

#define BGR_ELEM(image, row, col, off) CV_IMAGE_ELEM(image, uchar, row, col * image->nChannels + off)
#define B_ELEM(image, row, col) BGR_ELEM(image, row, col, 0)
#define G_ELEM(image, row, col) BGR_ELEM(image, row, col, 1)
#define R_ELEM(image, row, col) BGR_ELEM(image, row, col, 2)

#define FILTER_WIDTH 3
#define FILTER_HEIGHT 3

__constant__ double filter[FILTER_HEIGHT][FILTER_WIDTH] = 
{
   0.0, 0.2, 0.0,
   0.2, 0.2, 0.2,
   0.0, 0.2, 0.0
};

void cudaErrorHandler(cudaError_t err, const char *file, int line) {
   if(err != cudaSuccess) {
      fprintf(stderr, "%s on line %d: %s\n", file, line, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }
}

__device__ void convolutionFilter(IplImage *image, int x, int y) {
   double red = 0.0, green = 0.0, blue = 0.0; 
   double factor = 1.0; 
   double bias = 0.0;

   //multiply every value of the filter with corresponding image pixel 
   for(int filterX = 0; filterX < FILTER_WIDTH; filterX++) {
      for(int filterY = 0; filterY < FILTER_HEIGHT; filterY++) {
         int imageX = x - FILTER_WIDTH / 2 + filterX;
         int imageY = y - FILTER_HEIGHT / 2 + filterY;
         if(imageX < 0 || imageX >= image->height ||
            imageY < 0 || imageY >= image->width)
            continue;

         blue  += B_ELEM(image, imageX, imageY) * filter[filterX][filterY]; 
         green += G_ELEM(image, imageX, imageY) * filter[filterX][filterY]; 
         red   += R_ELEM(image, imageX, imageY) * filter[filterX][filterY]; 
      } 
   }

   //truncate values smaller than zero and larger than 255 
   B_ELEM(image, x, y) = min(max(int(factor * blue + bias), 0), 255); 
   G_ELEM(image, x, y) = min(max(int(factor * green + bias), 0), 255); 
   R_ELEM(image, x, y) = min(max(int(factor * red + bias), 0), 255);
}

__device__ void grayFilter(IplImage *image, int row, int col) {
   double avg = 0.0;

   for(int i=0; i<3; i++)
      avg += BGR_ELEM(image, row, col, i);
   avg /= 3.0;

   for(int i=0; i<3; i++) 
      BGR_ELEM(image, row, col, i) = avg;
}


__global__ void filterKernal(IplImage image) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if(row < image.height && col < image.width) {
      for(int i=0; i<10; i++)
         convolutionFilter(&image, row, col);
   }

}

void cudaFilter(IplImage *image) {
   IplImage devImage;

   int blkDimX = ceil((double)image->width / (double) THREAD_DIM);
   int blkDimY = ceil((double)image->height / (double) THREAD_DIM);

   dim3 blkDim(blkDimX, blkDimY);
   dim3 thrDim(THREAD_DIM, THREAD_DIM);

   memcpy(&devImage, image, sizeof(IplImage)); 
   CUDA_ERR_HANDLER(cudaMalloc(&devImage.imageData, image->imageSize));
   CUDA_ERR_HANDLER(cudaMemcpy(devImage.imageData, image->imageData, image->imageSize, cudaMemcpyHostToDevice));

   filterKernal<<<blkDim, thrDim>>>(devImage);
   CUDA_ERR_HANDLER(cudaGetLastError());

   CUDA_ERR_HANDLER(cudaMemcpy(image->imageData, devImage.imageData, image->imageSize, cudaMemcpyDeviceToHost));
   CUDA_ERR_HANDLER(cudaFree(devImage.imageData));
}
