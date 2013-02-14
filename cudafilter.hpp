#ifndef CUDAFILTER_H
#define CUDAFILTER_H

struct Filter {
   short *data;
   short width;
   short height;
};

template<typename T>
struct Image {
   T *data;
   int width;
   int height;
};

template<typename T>
void cudaFilter(Filter filter, Image<T> image);
void initFilter(Filter *filter, short width, short height);

#endif
