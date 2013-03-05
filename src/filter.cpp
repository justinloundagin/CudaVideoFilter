#include "filter.hpp"
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

static char *mapFile(const char *path, int *size, int flags) {
   int fd;
   struct stat fileStat;
   char *ret = NULL;

   if((fd = open(path, flags)) == -1);
   else if(fstat(fd, &fileStat) < 0);
   else ret = (char*)mmap(0, *size = fileStat.st_size, 
               PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
   return ret;
}

HOST DEVICE Filter::Filter(char *path, float factor, float bias) 
   :rows(0), cols(0), factor(factor), bias(bias) {
   char *tmp = NULL;
   int bytes;
   char *buff = mapFile(path, &bytes, O_RDONLY);

   for(int ndx = 0; ndx < bytes; ndx++ ) {
      if(!rows && buff[ndx] == '.')
         cols++;
      if(buff[ndx] == '\n')
         rows++;
   }

   data = new float[rows * cols]; //createFilter(rows, cols, factor, bias, 0.0);
   float *temp = data;
   for(int i = 0; bytes && i < rows; i++) {
      for(int j = 0; bytes && j < cols; j++) {
         char *tmp = buff;
         *temp++ = strtod(buff, &buff);
         bytes -= buff - tmp;
         while(bytes && isspace(*buff)) {
            buff++;
            bytes--;
         }
      }
   }

   munmap(buff, bytes);
}

