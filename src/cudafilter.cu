#include "cudafilter.hpp"
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cmath>
#include <stdio.h>
#include <assert.h>

#define THREAD_DIM 16
#define CUDA_ERR_HANDLER(err) cudaErrorHandler(err, __FILE__, __LINE__)
#define IMAGE_ELEM(image, elemtype, row, col) \
    (((elemtype*)(image.data + image.widthStep*(row)))[(col)])

#define BLUE 0
#define GREEN 1
#define RED 2

struct Image {
   char *data;
   int width; 
   int height;
   int widthStep;
   int nChannels;
};

static void cudaErrorHandler(cudaError_t err, const char *file, int line) {
   if(err != cudaSuccess) {
      fprintf(stderr, "%s on line %d: %s\n", file, line, cudaGetErrorString(err));
      exit(EXIT_FAILURE);
   }
}

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
         
         float filterVal = filterElement(filter, filterX, filterY);
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

Image cvImageToDevice(IplImage *image) {
   Image dev;
   dev.width = image->width;
   dev.height = image->height;
   dev.nChannels = image->nChannels;
   dev.widthStep = image->widthStep;
   CUDA_ERR_HANDLER(cudaMalloc(&dev.data, image->imageSize));
   CUDA_ERR_HANDLER(cudaMemcpy(dev.data, image->imageData, image->imageSize, cudaMemcpyHostToDevice));
   return dev;
}

Filter filterToDevice(Filter *filter) {
   Filter dev;
   dev.rows = filter->rows;
   dev.cols = filter->cols;
   CUDA_ERR_HANDLER(cudaMalloc(&dev.data, filter->rows * filter->cols * sizeof(float)));
   CUDA_ERR_HANDLER(cudaMemcpy(dev.data, filter->data, filter->rows * filter->cols * sizeof(float), cudaMemcpyHostToDevice));
   return dev;
}

void cudaFilter(IplImage *image, Filter *filter) {
   Image devImage, devResult;
   Filter devFilter;

   int blkDimX = ceil((float)image->width / (float) THREAD_DIM);
   int blkDimY = ceil((float)image->height / (float) THREAD_DIM);

   dim3 blkDim(blkDimX, blkDimY);
   dim3 thrDim(THREAD_DIM, THREAD_DIM);

   devImage = cvImageToDevice(image);
   devResult = cvImageToDevice(image);
   devFilter = filterToDevice(filter);

   filterKernal<<<blkDim, thrDim>>>(devImage, devResult, devFilter);
   CUDA_ERR_HANDLER(cudaGetLastError());

   CUDA_ERR_HANDLER(cudaMemcpy(image->imageData, devResult.data, image->imageSize, cudaMemcpyDeviceToHost));
   CUDA_ERR_HANDLER(cudaFree(devImage.data));
   CUDA_ERR_HANDLER(cudaFree(devResult.data));
   CUDA_ERR_HANDLER(cudaFree(devFilter.data));
}
