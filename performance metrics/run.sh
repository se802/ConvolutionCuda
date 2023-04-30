#!/bin/bash
rm -r ./output
mkdir output
/snap/clion/222/bin/cmake/linux/x64/bin/cmake --build /home/stelios/CLionProjects/ConvolutionCuda/cmake-build-debug --target hw4 -- -j 6
mv ./cmake-build-debug/hw4 .
exe="./hw4"
blocks=160
threads=128
for image in lady.bmp animal_small.bmp animal.bmp BackgroundRadiationMap.bmp
do
  for filter in 1 2 3 4 5 6 7 8 9 10
  do
                #echo "perf stat -e "$value":u   " $exe $n " >> results.txt 2>&1"
                #perf stat -e $value:u  $exe $n >> results.txt 2>&1
                echo $exe $filter $image $blocks $threads
                $exe $filter $image $blocks $threads
  done
done
