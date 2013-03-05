CC = nvcc
SRC = src/cudafilter.cu src/main.cpp
TARGET_OS = $(shell uname)

ifeq ($(TARGET_OS), Darwin)
   CFLAGS = -I/usr/local/Cellar/opencv/2.4.4/include/opencv -I/usr/local/Cellar/opencv/2.4.4./includa -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -Iutil/
   NVFLAGS = -m64 -arch=sm_12
else
   CFLAGS = `pkg-config opencv --cflags --libs`
   NVFLAGS = -arch=sm_30
endif

cudafilter: $(SRC)
	$(CC)  $(NVFLAGS) $(CFLAGS) $^ -o $@

clean:
	rm -rf *.o cudafilter
