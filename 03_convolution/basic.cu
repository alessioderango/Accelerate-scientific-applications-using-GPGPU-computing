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


__global__ void convolution_basic_GPU(unsigned char* in, unsigned char* out, int w, int h){
}

void convolution_basic_CPU(unsigned char* in, unsigned char* out, int w, int h){
}


void imageFilter(const char* filename, const char* copyname, int bs_x, int bs_y){
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

