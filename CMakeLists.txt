cmake_minimum_required(VERSION 3.19)
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(ConvolutionCuda CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g    -m64 -gencode arch=compute_50,code=sm_50 -Wno-deprecated-gpu-targets  ")
set(CMAKE_CUDA_STANDARD 98)

add_executable(hw4 main.cu)

set_target_properties(
        hw4
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)