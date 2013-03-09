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

__constant__ float cfilter[FILTER_SIZE * FILTER_SIZE];

static void cudaErrorHandler(cudaError_t err, const char *file, int line) {
   if(err != cudaSuccess) {
      std::cerr<<cudaGetErrorString(err)<<" on line "<<line<<" : "<<file<<std::endl;
      exit(EXIT_FAILURE);
   }
}

__device__ void convolutionFilter(Image image, Image result, int x, int y) {
   
   /*
   
   float3 bgr;

   //multiply every value of the filter with corresponding image pixel 
   for(int filterX = 0; filterX < FILTER_SIZE; filterX++) {
      for(int filterY = 0; filterY < FILTER_SIZE; filterY++) {
         int imageX = x - FILTER_SIZE / 2 + filterX;
         int imageY = y - FILTER_SIZE / 2 + filterY;
         if(imageX < 0 || imageX >= image.width ||
            imageY < 0 || imageY >= image.height)
            continue;
         
         float filterVal = cfilter[filterY * FILTER_SIZE + filterX];
         bgr.x += image.at(imageY, imageX, BLUE) * filterVal;
         bgr.y += image.at(imageY, imageX, GREEN) * filterVal;
         bgr.z += image.at(imageY, imageX, RED) * filterVal;
      } 
   }

   */
   
   
   
   float3 bgr;

   __shared__ unsigned sharedImage[THREADS_PER_DIM][THREADS_PER_DIM * 3];
   int tx = threadIdx.x;
   int ty = threadIdx.y;

   sharedImage[ty][3*tx] = image.at(y, x, BLUE);
   sharedImage[ty][3*tx+1] = image.at(y, x, GREEN);
   sharedImage[ty][3*tx+2] = image.at(y, x, RED);

   __syncthreads();

  // multiply every value of the filter with corresponding image pixel 
   for(int filterX = 0; filterX < FILTER_SIZE; filterX++) {
      for(int filterY = 0; filterY < FILTER_SIZE; filterY++) {

         unsigned shX = tx - FILTER_SIZE / 2 + filterX;
         unsigned shY = ty - FILTER_SIZE / 2 + filterY;
         float filterVal = cfilter[filterY * FILTER_SIZE + filterX];
         
         shX >= THREADS_PER_DIM && (shX = THREADS_PER_DIM - 1);
         shY >= THREADS_PER_DIM && (shY = THREADS_PER_DIM - 1);
         shX < 0 && (shX = 0);
         shY < 0 && (shY = 0);
         

         bgr.x += sharedImage[shY][3 * shX] * filterVal; 
         bgr.y += sharedImage[shY][3 * shX + 1] * filterVal;
         bgr.z += sharedImage[shY][3 * shX + 2] * filterVal; 
      } 
   }
   
   //truncate values smaller than zero and larger than 255 
   result.at(y, x, BLUE) = min(max(int(1 * bgr.x + 0), 0), 255); 
   result.at(y, x, GREEN) = min(max(int(1 * bgr.y + 0), 0), 255); 
   result.at(y, x, RED) = min(max(int(1 * bgr.z + 0), 0), 255); 
}

__global__ void filterKernal(Image image, Image result) {
   int y = blockIdx.y * blockDim.y + threadIdx.y;
   int x = blockIdx.x * blockDim.x + threadIdx.x;

   if(y < image.height && x < image.width) {
      convolutionFilter(image, result, x, y);
   }
}

CudaFilter::CudaFilter(Image image, Filter filter)
   :image(image), devImage(image), devResult(image){
      cudaMemcpyToSymbol(cfilter, filter.data, FILTER_SIZE * FILTER_SIZE * sizeof(float));
      toDevice((void**)&devImage.data, image.data, image.size);
      toDevice((void**)&devResult.data, image.data, image.size);
   //   toDevice((void**)&devFilter.data, filter.data, filter.rows * filter.cols * sizeof(float));
}

CudaFilter::~CudaFilter() {
   assert(image.size == devResult.size);
   toHost(image.data, devResult.data, image.size);
   CUDA_ERR_HANDLER(cudaFree(devImage.data));
   CUDA_ERR_HANDLER(cudaFree(devResult.data));
   //CUDA_ERR_HANDLER(cudaFree(devFilter.data));
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

   filterKernal<<<blocksPerGrid, threadsPerBlock>>>(devImage, devResult);
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