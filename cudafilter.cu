#include "cudafilter.hpp"
#include <stdlib.h>
#include <cuda.h>
#include <cmath>
#include <stdio.h>

#define THREAD_DIM 32
#define CUDA_ERR_HANDLER(err) cudaErrorHandler(err, __FILE__, __LINE__)

template void cudaFilter(Filter, Image<unsigned char>);

void cudaErrorHandler(cudaError_t err, const char *file, int line) {
   if(err != cudaSuccess) {
      fprintf(stderr, "%s in line %d: %s\n", file, line, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }
}
void initFilter(Filter *filter, short width, short height) {
   filter->data = (short*)calloc(width * height, sizeof(short));
   filter->width = width;
   filter->height = height;
}

template<typename T>
__device__ T *pixelAt(Image<T> image, int x, int y) {
   return image.data + image.width * y + x;
}

__device__ short filterAt(Filter filter, int x, int y) {
   return filter.data[y * filter.width + x];
}

template<typename T>
__device__ void applyFilter(Filter filter, Image<T> result, Image<T> image, int x, int y) {
   double thresh = 0.0;

   for(int fx = 0; fx < filter.width; fx++) {
      for(int fy = 0; fy < filter.height; fy++) {
         int imgx = (x - filter.width / 2 + fx + image.width) % image.width;
         int imgy = (y - filter.height / 2 + fy + image.height) % image.height;

         thresh += *pixelAt(image, imgx, imgy) * filterAt(filter, fx, fy);
      
      }
      result.data[image.width * y + x] = thresh;
   }
}

template<typename T>
__global__ void filterKernal(Filter filter, Image<T> result, Image<T> image) {
   int px = blockIdx.x * blockDim.x + threadIdx.x;
   int py = blockIdx.y * blockDim.y + threadIdx.y;

   if(px < image.width && py < image.height) {
      applyFilter(filter, result, image, px, py);
   }
}

template<typename T>
void cudaFilter(Filter filter, Image<T> image) {
   Filter devFilter;
   Image<T> devImage;
   Image<T> devResult;

   int blkDimX = ceil((double)image.width / (double) THREAD_DIM);
   int blkDimY = ceil((double)image.height / (double) THREAD_DIM);

   dim3 blkDim(blkDimX, blkDimY);
   dim3 thrDim(THREAD_DIM, THREAD_DIM);

   int imageSize = sizeof(T) * image.width * image.height;
   int filterSize = sizeof(short) * filter.width * filter.height;

   CUDA_ERR_HANDLER(cudaMalloc(&devImage.data, imageSize));
   CUDA_ERR_HANDLER(cudaMalloc(&devResult.data, imageSize));
   CUDA_ERR_HANDLER(cudaMalloc(&devFilter.data, filterSize));


   devImage.width = devResult.width = image.width;
   devImage.height = devResult.height = image.height;
   devFilter.width = filter.width;
   devFilter.height = filter.height;

   CUDA_ERR_HANDLER(cudaMemcpy(devImage.data, image.data, imageSize, cudaMemcpyHostToDevice));
   CUDA_ERR_HANDLER(cudaMemcpy(devFilter.data, filter.data, filterSize, cudaMemcpyHostToDevice));


 //  printf("starting kernal: blkdimx = %d\nblkdimy = %d\nimage.width = %d\nimage.height=%d\n", blkDimX, blkDimY, devImage.width, devImage.height);
   filterKernal<<<blkDim, thrDim>>>(devFilter, devResult, devImage);
   CUDA_ERR_HANDLER(cudaGetLastError());

   //copy back resultant image
   CUDA_ERR_HANDLER(cudaMemcpy(image.data, devImage.data, imageSize, cudaMemcpyDeviceToHost));

   CUDA_ERR_HANDLER(cudaFree(devImage.data));
   CUDA_ERR_HANDLER(cudaFree(devResult.data));
   CUDA_ERR_HANDLER(cudaFree(devFilter.data));
}
