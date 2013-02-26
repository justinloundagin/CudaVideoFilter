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

Filter *createFilter(int rows, int cols, double factor, double bias, double val) {
   Filter *filter = (Filter*)malloc(sizeof(Filter));
   filter->data = (double*)malloc(rows * cols * sizeof(double));
   filter->rows = rows;
   filter->cols = cols;
   filter->factor = factor;
   filter->bias = bias;
   for(int i=0; i<rows; i++) {
      for(int j=0; j<cols; j++) {
         filterElement(filter, i, j) = val;
      }
   }
   return filter;
}

Filter **createFiltersFromFiles(char **paths, int size) {
   Filter **filters = (Filter**)calloc(size, sizeof(Filter*));
   for(int i=0; i<size; i++)
      filters[i] = createFilterFromFile(paths[i], 1.0, 0.0);
   return filters;
}


Filter *createFilterFromFile(char *path, double factor, double bias) {
   char *tmp = NULL;
   double *data;
   int bytes, rows = 0, cols = 0;
   char *buff = mapFile(path, &bytes, O_RDONLY);

   for(int ndx = 0; ndx < bytes; ndx++ ) {
      if(!rows && buff[ndx] == '.')
         cols++;
      if(buff[ndx] == '\n')
         rows++;
   }

   Filter *filter = createFilter(rows, cols, factor, bias, 0.0);
   data = filter->data;
   for(rows = 0; bytes && rows < filter->rows; rows++) {
      for(cols = 0; bytes && cols < filter->cols; cols++) {
         *data++ = strtod(tmp = buff, &buff);
         bytes -= buff - tmp;
         while(bytes && isspace(*buff)) {
            buff++;
            bytes--;
         }
      }
   }
   munmap(buff, bytes);
   return filter;
}

void freeFilter(Filter *filter) {
   assert(filter && filter->data);
   free(filter->data);
   free(filter);
}

void freeFilters(Filter **filters, int size) {
   while(size--)
      freeFilter(*filters++);
}
