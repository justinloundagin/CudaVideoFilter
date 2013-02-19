#include "cudafilter.hpp"
#include <stdlib.h>
#include <cuda.h>
#include <cmath>
#include <stdio.h>

#define THREAD_DIM 32
#define CUDA_ERR_HANDLER(err) cudaErrorHandler(err, __FILE__, __LINE__)

void cudaErrorHandler(cudaError_t err, const char *file, int line) {
   if(err != cudaSuccess) {
      fprintf(stderr, "%s on line %d: %s\n", file, line, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }
}

__device__ void grayFilter(IplImage *image, int row, int col) {
   double avg = 0.0;

   for(int i=0; i<3; i++)
      avg += CV_IMAGE_ELEM(image, uchar, row, col * image->nChannels + i);
   avg /= 3.0;

   for(int i=0; i<3; i++) 
      CV_IMAGE_ELEM(image, uchar, row, col * image->nChannels + i) = avg;
}


__global__ void filterKernal(IplImage image) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if(row < image.height && col < image.width) {
      grayFilter(&image, row, col);
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
   //invoke kernal

   //  cudaDeviceSynchronize();

   CUDA_ERR_HANDLER(cudaMemcpy(image->imageData, devImage.imageData, image->imageSize, cudaMemcpyDeviceToHost));
   CUDA_ERR_HANDLER(cudaFree(devImage.imageData));
}
