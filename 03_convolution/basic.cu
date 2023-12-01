#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <malloc.h>
#include <cuda.h>
#include <chrono>
#include <device_launch_parameters.h>
#include <iostream>
using namespace std;

#define MASK_WIDTH 3
#define MAX_VALUE 255
#define MIN_VALUE 0

__constant__ float d_Mask[9];

__global__ void convolution_basic(unsigned char* in, unsigned char* out, int w, int h){
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;

  if (Col < w && Row < h){
    float pixR = 0; float pixG = 0; float pixB = 0;

    int in_start_i = Row - (MASK_WIDTH/2);
    int in_start_j = Col - (MASK_WIDTH/2);
    for (int convRow = 0; convRow < MASK_WIDTH; convRow++)
      for (int convCol = 0; convCol < MASK_WIDTH; convCol++){

        if (in_start_i + convRow >= 0 && in_start_i + convRow < h && in_start_j + convCol >= 0 && in_start_j + convCol < w){
          pixR += in[4*((in_start_i + convRow) * w + (in_start_j + convCol))    ] * d_Mask[convRow * MASK_WIDTH + convCol];
          pixG += in[4*((in_start_i + convRow) * w + (in_start_j + convCol)) + 1] * d_Mask[convRow * MASK_WIDTH + convCol];
          pixB += in[4*((in_start_i + convRow) * w + (in_start_j + convCol)) + 2] * d_Mask[convRow * MASK_WIDTH + convCol];
        }
      }

    pixR = pixR > MAX_VALUE ? MAX_VALUE : pixR;
    pixR = pixR < MIN_VALUE ? MIN_VALUE : pixR;  
    pixG = pixG > MAX_VALUE ? MAX_VALUE : pixG;
    pixG = pixG < MIN_VALUE ? MIN_VALUE : pixG;
    pixB = pixB > MAX_VALUE ? MAX_VALUE : pixB;
    pixB = pixB < MIN_VALUE ? MIN_VALUE : pixB;

    out[4*(Row * w + Col) ] = (unsigned char)pixR;
    out[4*(Row * w + Col) + 1] = (unsigned char)pixG;
    out[4*(Row * w + Col) + 2] = (unsigned char)pixB;
    out[4*(Row * w + Col) + 3] = in[4*(Row * w + Col) + 3];
  }
}

float* convertToFloat(unsigned char* image, int size){
  float* newimage =(float*)malloc(size * sizeof(float) * 4);

  for(int i = 0; i < size * 4; i++){
    newimage[i] = (float)image[i];
  }

  return newimage;
}

unsigned char* convertToChar(float* image, int size){
  unsigned char* newimage =(unsigned char*)malloc(size * 4);

  for(int i = 0; i < size * 4; i++){
    newimage[i] = (char)image[i];
  }

  return newimage;
}

void convolution_basic_CPU(unsigned char* in, unsigned char* out, int w, int h){
  float Mask[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
  for(int i =0; i < w; i++)
    for(int j =0; j < h; j++)
    {
      int Col = j;
      int Row = i;

      if (Col < w && Row < h){
        float pixR = 0; float pixG = 0; float pixB = 0;

        int in_start_i = Row - (MASK_WIDTH/2);
        int in_start_j = Col - (MASK_WIDTH/2);
        for (int convRow = 0; convRow < MASK_WIDTH; convRow++)
          for (int convCol = 0; convCol < MASK_WIDTH; convCol++){

            if (in_start_i + convRow >= 0 && in_start_i + convRow < h && in_start_j + convCol >= 0 && in_start_j + convCol < w){
              pixR += in[4*((in_start_i + convRow) * w + (in_start_j + convCol))    ] * Mask[convRow * MASK_WIDTH + convCol];
              pixG += in[4*((in_start_i + convRow) * w + (in_start_j + convCol)) + 1] * Mask[convRow * MASK_WIDTH + convCol];
              pixB += in[4*((in_start_i + convRow) * w + (in_start_j + convCol)) + 2] * Mask[convRow * MASK_WIDTH + convCol];
            }
          }

        pixR = pixR > MAX_VALUE ? MAX_VALUE : pixR;
        pixR = pixR < MIN_VALUE ? MIN_VALUE : pixR;  
        pixG = pixG > MAX_VALUE ? MAX_VALUE : pixG;
        pixG = pixG < MIN_VALUE ? MIN_VALUE : pixG;
        pixB = pixB > MAX_VALUE ? MAX_VALUE : pixB;
        pixB = pixB < MIN_VALUE ? MIN_VALUE : pixB;

        out[4*(Row * w + Col) ] = (unsigned char)pixR;
        out[4*(Row * w + Col) + 1] = (unsigned char)pixG;
        out[4*(Row * w + Col) + 2] = (unsigned char)pixB;
        out[4*(Row * w + Col) + 3] = in[4*(Row * w + Col) + 3];
      }
    }
}


void imageFilter(const char* filename, const char* copyname, int bs_x, int bs_y){
  unsigned width, height;
  unsigned char* original_image;
  unsigned char* modified_image;
  unsigned char* d_original_image;
  unsigned char* d_modified_image;

  unsigned error = lodepng_decode32_file(&original_image, &width, &height, filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

  int size = width * height * 4;

  printf("dimension: %d", width * height);

  modified_image = (unsigned char*)malloc(size);

  cudaMalloc((void**) &d_original_image, size);
  cudaMalloc((void**) &d_modified_image, size);

  cudaMemcpy(d_original_image, original_image, size, cudaMemcpyHostToDevice); 

  //int Mask[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0}; //sharpen
  float Mask[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1}; //edge detection
  //float Mask[9] = {1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9, 1./9}; //blur
  //float Mask[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1}; //gaussian blur

  cudaMemcpyToSymbol(d_Mask, Mask, 9*sizeof(int));

  dim3 block_size(bs_x, bs_y, 1);
	dim3 number_of_blocks(ceil(width / (float)block_size.x), ceil(height / (float)block_size.y), 1);
  
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();

  convolution_basic <<<number_of_blocks, block_size>>> (d_original_image, d_modified_image, width, height);
  cudaError_t err = cudaGetLastError();
  if ( err != cudaSuccess )
  {
     printf("CUDA Error: %s\n", cudaGetErrorString(err));       
     // Possibly: exit(-1) if program cannot continue....
  }
  cudaDeviceSynchronize();
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;  
  cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
  
  cudaMemcpy(modified_image, d_modified_image, size, cudaMemcpyDeviceToHost);

  error = lodepng_encode32_file(copyname, modified_image, width, height);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

  std::chrono::time_point<std::chrono::system_clock> start_cpu, end_cpu;
  start_cpu = std::chrono::system_clock::now();
  convolution_basic_CPU(original_image, modified_image, width, height);
  end_cpu = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds_cpu = end_cpu - start_cpu;  
  cout << "elapsed time: " << elapsed_seconds_cpu.count() << "s\n";

  free(original_image);
  free(modified_image);
  cudaFree(d_original_image);
  cudaFree(d_modified_image);
}




int main(int argc, char *argv[]){
  const char* filename = argv[1];
  const char* copyname = "modified_basic.png";

  int bs_x, bs_y; //block_size.x and block_size.y

  char *e = argv[2];
  bs_x = atoi(e);
	e = argv[3];
	bs_y = atoi(e);

  imageFilter(filename, copyname, bs_x, bs_y);

  return(0);
}

