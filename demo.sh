#!/bin/bash

./cudafilter filters/edge.dat 1.0 0.0 -gray
./cudafilter filters/motionblur.dat .111 128.0
./cudafilter filters/sharpen.dat 1.0 0.0
