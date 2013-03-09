#!/bin/bash

for i in {5..50..5}
do
   make clean
   make FLAGS=-DFILTER_SIZE=\'$i\'
   ./cudafilter ../input/$i.dat
done


