#ifndef FILTER_HPP
#define FILTER_HPP

#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cctype>

#ifdef __CUDACC__
#define HOST __host__
#define DEVICE __device__
#define INLINE __forceinline__
#else
#define HOST
#define DEVICE
#define INLINE
#endif

class Filter {

public:
   float factor;
   float bias;
   float *data;
   int rows; 
   int cols;

   HOST DEVICE Filter() {}
   HOST DEVICE Filter(const Filter &filter) {
      rows = filter.rows;
      cols = filter.cols;
      data = filter.data;
      bias = filter.bias;
      factor = filter.factor;
   }

   HOST DEVICE Filter(char *path, float factor, float bias) 
   :rows(0), cols(0), factor(factor), bias(bias) {
      int fd;
      struct stat fileStat;

      if((fd = open(path, O_RDONLY)) == -1);
      else if(fstat(fd, &fileStat) < 0);
      else {
         int bytes;
         char *buff = (char*)mmap(0, bytes = fileStat.st_size, 
                  PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);

         for(int ndx = 0; ndx < bytes; ndx++ ) {
            if(!rows && buff[ndx] == '.') cols++;
            if(buff[ndx] == '\n') rows++;
         }

         data = new float[rows * cols];
         for(int i = 0; bytes && i < rows; i++) {
            for(int j = 0; bytes && j < cols; j++) {
               char *ptr = buff;
               (*this)[i][j] = strtod(buff, &buff);
               bytes -= buff - ptr;
               while(bytes && isspace(*buff)) {
                  buff++;
                  bytes--;
               }
            }
         }
         munmap(buff, bytes);
      }
   }

   HOST DEVICE float &at(int row, int col) { 
      return (*this)[row][col];
   }
   HOST DEVICE float *operator[](int row) { 
      return data + row * cols; 
   }
};

#endif
