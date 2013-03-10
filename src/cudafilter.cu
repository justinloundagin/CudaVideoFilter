#include "cudafilter.hpp"
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <iostream>

#define THREADS_PER_DIM 32
#define CUDA_ERR_HANDLER(err) cudaErrorHandler(err, __FILE__, __LINE__)


static void cudaErrorHandler(cudaError_t err, const char *file, int line) {
   if(err != cudaSuccess) {
      std::cerr<<cudaGetErrorString(err)<<" on line "<<line<<" : "<<file<<std::endl;
      exit(EXIT_FAILURE);
   }
}

__device__ void convolutionFilter(Image image, Image result, Filter filter, int x, int y) {
   float3 pixel;

   //multiply every value of the filter with corresponding image pixel 
   for(int filterX = 0; filterX < filter.cols; filterX++) {
      for(int filterY = 0; filterY < filter.rows; filterY++) {
         int imageX = x - filter.cols / 2 + filterX;
         int imageY = y - filter.rows / 2 + filterY;
         if(imageX < 0 || imageX >= image.width ||
            imageY < 0 || imageY >= image.height)
            continue;
         
         float filterVal = filter[filterY][filterX];
         pixel.x += image.at(imageY, imageX, BLUE) * filterVal;
         pixel.y += image.at(imageY, imageX, GREEN) * filterVal;
         pixel.z += image.at(imageY, imageX, RED) * filterVal;
      } 
   }

   //truncate values smaller than zero and larger than 255 
   result.at(y, x, BLUE) = min(max(int(filter.factor * pixel.x + filter.bias), 0), 255); 
   result.at(y, x, GREEN) = min(max(int(filter.factor * pixel.y + filter.bias), 0), 255); 
   result.at(y, x, RED) = min(max(int(filter.factor * pixel.z + filter.bias), 0), 255); 
}

__global__ void filterKernal(Image image, Image result, Filter filter) {
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int x = blockIdx.x * blockDim.x + threadIdx.x;

   if(y < image.height && x < image.width) {
      convolutionFilter(image, result, filter, x, y);
   }
}

CudaFilter::CudaFilter(Image image, Filter filter)
   :image(image), devImage(image), devResult(image), devFilter(filter) {
      toDevice((void**)&devImage.data, image.data, image.size);
      toDevice((void**)&devResult.data, image.data, image.size);
      toDevice((void**)&devFilter.data, filter.data, filter.rows * filter.cols * sizeof(float));
}

CudaFilter::~CudaFilter() {
   assert(image.size == devResult.size);
   toHost(image.data, devResult.data, image.size);
   CUDA_ERR_HANDLER(cudaFree(devImage.data));
   CUDA_ERR_HANDLER(cudaFree(devResult.data));
   CUDA_ERR_HANDLER(cudaFree(devFilter.data));
}

void CudaFilter::toDevice(void **dev, void *host, int bytes) {
   CUDA_ERR_HANDLER(cudaMalloc(dev, bytes));
   CUDA_ERR_HANDLER(cudaMemcpy(*dev, host, bytes, cudaMemcpyHostToDevice));
}

void CudaFilter::toHost(void *host, void *dev, int bytes) {
   CUDA_ERR_HANDLER(cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost));
}

void CudaFilter::applyFilter() {
   dim3 threadsPerBlock(THREADS_PER_DIM, THREADS_PER_DIM);
   dim3 blocksPerGrid((image.width + THREADS_PER_DIM - 1) / THREADS_PER_DIM, 
                      (image.height + THREADS_PER_DIM - 1) / THREADS_PER_DIM);

   filterKernal<<<blocksPerGrid, threadsPerBlock>>>(devImage, devResult, devFilter);
   cudaThreadSynchronize();
   CUDA_ERR_HANDLER(cudaGetLastError());
}

float CudaFilter::operator() () {
   float ms;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);

   applyFilter();

   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&ms, start, stop);
   return ms;
}