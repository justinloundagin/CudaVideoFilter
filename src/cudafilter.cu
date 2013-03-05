#include "cudafilter.hpp"
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <iostream>

#define THREAD_DIM 16
    
#define CUDA_ERR_HANDLER(err) cudaErrorHandler(err, __FILE__, __LINE__)
static void cudaErrorHandler(cudaError_t err, const char *file, int line) {
   if(err != cudaSuccess) {
      std::cerr<<cudaGetErrorString(err)<<" on line "<<line<<" : "<<file<<std::endl;
      exit(EXIT_FAILURE);
   }
}

__device__ void convolutionFilter(Image image, Image result, Filter filter, int x, int y) {
   float3 bgr;

   //multiply every value of the filter with corresponding image pixel 
   for(int filterX = 0; filterX < filter.cols; filterX++) {
      for(int filterY = 0; filterY < filter.rows; filterY++) {
         int imageX = x - filter.cols / 2 + filterX;
         int imageY = y - filter.rows / 2 + filterY;
         if(imageX < 0 || imageX >= image.height ||
            imageY < 0 || imageY >= image.width)
            continue;
         
         float filterVal = filter[filterX][filterY];
         bgr.x += image.at(imageX, imageY, BLUE) * filterVal;
         bgr.y += image.at(imageX, imageY, GREEN) * filterVal;
         bgr.z += image.at(imageX, imageY, RED) * filterVal;
      } 
   }

   //truncate values smaller than zero and larger than 255 
   result.at(x, y, BLUE) = min(max(int(1 * bgr.x + 0), 0), 255); 
   result.at(x, y, GREEN) = min(max(int(1 * bgr.y + 0), 0), 255); 
   result.at(x, y, RED) = min(max(int(1 * bgr.z + 0), 0), 255); 
}

__global__ void filterKernal(Image image, Image result, Filter filter) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if(row < image.height && col < image.width) {
      convolutionFilter(image, result, filter, row, col);
   }
}

CudaFilter::CudaFilter(cv::Mat img, Filter filter)
   :image(img), filter(filter) {}

Image CudaFilter::imageToDevice(Image img) {
   Image dev;
   memcpy(&dev, &img, sizeof(Image));
   CUDA_ERR_HANDLER(cudaMalloc(&dev.data, img.size));
   CUDA_ERR_HANDLER(cudaMemcpy(dev.data, img.data, img.size, cudaMemcpyHostToDevice));
   return dev;

}

Filter CudaFilter::filterToDevice(Filter filter) {
   Filter dev(filter);
   CUDA_ERR_HANDLER(cudaMalloc(&dev.data, filter.rows * filter.cols * sizeof(float)));
   CUDA_ERR_HANDLER(cudaMemcpy(dev.data, filter.data, filter.rows * filter.cols * sizeof(float), cudaMemcpyHostToDevice));
   return dev;

}

void CudaFilter::applyFilter() {
   Image devImage, devResult;
   Filter devFilter;

   int blkDimX = ceil((float)image.width/ (float) THREAD_DIM);
   int blkDimY = ceil((float)image.height / (float) THREAD_DIM);

   dim3 blkDim(blkDimX, blkDimY);
   dim3 thrDim(THREAD_DIM, THREAD_DIM);

   devImage = imageToDevice(image);
   devResult = imageToDevice(image);
   devFilter = filterToDevice(filter);

   filterKernal<<<blkDim, thrDim>>>(devImage, devResult, devFilter);
   CUDA_ERR_HANDLER(cudaGetLastError());

   CUDA_ERR_HANDLER(cudaMemcpy(image.data, devResult.data, image.size, cudaMemcpyDeviceToHost));
   CUDA_ERR_HANDLER(cudaFree(devImage.data));
   CUDA_ERR_HANDLER(cudaFree(devResult.data));
   CUDA_ERR_HANDLER(cudaFree(devFilter.data));
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