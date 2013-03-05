#include "cudafilter.hpp"
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cmath>
#include <stdio.h>
#include <assert.h>
#include <iostream>

#define THREAD_DIM 16
#define IMAGE_ELEM(image, elemtype, row, col) \
    (((elemtype*)(image.data + image.widthStep*(row)))[(col)])
    
#define CUDA_ERR_HANDLER(err) cudaErrorHandler(err, __FILE__, __LINE__)

static void cudaErrorHandler(cudaError_t err, const char *file, int line) {
   if(err != cudaSuccess) {
      fprintf(stderr, "%s on line %d: %s\n", file, line, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }
}
#define BLUE 0
#define GREEN 1
#define RED 2


__device__ uchar &imageElement(Image image, int row, int col, int ndx) {
   return IMAGE_ELEM(image, uchar, row, col * image.nChannels + ndx);
}

__device__ void convolutionFilter(Image image, Image result, Filter filter, int x, int y) {
   float red = 0.0, green = 0.0, blue = 0.0; 

   //multiply every value of the filter with corresponding image pixel 
   for(int filterX = 0; filterX < filter.cols; filterX++) {
      for(int filterY = 0; filterY < filter.rows; filterY++) {
         int imageX = x - filter.cols / 2 + filterX;
         int imageY = y - filter.rows / 2 + filterY;
         if(imageX < 0 || imageX >= image.height ||
            imageY < 0 || imageY >= image.width)
            continue;
         
         float filterVal = filter[filterX][filterY];
         blue  += imageElement(image, imageX, imageY, BLUE) * filterVal;
         green += imageElement(image, imageX, imageY, GREEN) * filterVal;
         red   += imageElement(image, imageX, imageY, RED) * filterVal;
      } 
   }

   //truncate values smaller than zero and larger than 255 
   imageElement(result, x, y, BLUE) = min(max(int(1 * blue + 0), 0), 255); 
   imageElement(result, x, y, GREEN) = min(max(int(1 * green + 0), 0), 255); 
   imageElement(result, x, y, RED) = min(max(int(1 * red + 0), 0), 255); 
}

__global__ void filterKernal(Image image, Image result, Filter filter) {
   int row = blockIdx.y * blockDim.y + threadIdx.y;
   int col = blockIdx.x * blockDim.x + threadIdx.x;

   if(row < image.height && col < image.width) {
         convolutionFilter(image, result, filter, row, col);
   }
}

CudaFilter::CudaFilter(cv::Mat img, Filter filter) :
   filter(filter) {
      IplImage tmp = img;
      image.width = tmp.width;
      image.height = tmp.height;
      image.nChannels = tmp.nChannels;
      image.widthStep = tmp.widthStep;
      image.size = tmp.imageSize;
      image.data = tmp.imageData;
}


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

void CudaFilter::operator() () {
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