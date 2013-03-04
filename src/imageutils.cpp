#include "imageutils.hpp"

IplImage *stitchImages(IplImage *images[], int numImages) {
    int w = ceil(numImages / 2.0);
    int h = 2.0;
    int size = 300;

    // Create a new 3 channel image
    IplImage *stitched = cvCreateImage( cvSize(100 + size*w, 60 + size*h), 8, 3 );

    // Loop for nArgs number of arguments
    for (int i = 0, m = 20, n = 20; i < numImages; i++, m += (20 + size)) {
        IplImage *img = images[i];
        int max = (img->width > img->height)? img->width: img->height;
        float scale = (float) ( (float) max / size );

        if( i % w == 0 && m!= 20) {
            m = 20;
            n += 20 + size;
        }
        cvSetImageROI(stitched, cvRect(m, n, (int)( img->width/scale ), (int)( img->height/scale )));
        cvResize(img, stitched);
        cvResetImageROI(stitched);
    }
    return stitched;
}
