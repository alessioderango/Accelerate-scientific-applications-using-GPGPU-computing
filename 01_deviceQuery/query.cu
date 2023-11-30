#include <stdio.h>
#include <iostream>
using namespace std;

int main() {
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  managedMemory: %d\n", prop.managedMemory);
    printf("  Global Memory [MiB]: %d\n", prop.totalGlobalMem/(1<<20));
    printf("  sharedMemPerMultiprocessor [KiB]: %d\n", prop.sharedMemPerMultiprocessor/(1<<10));
    //printf("  maxBlocksPerMultiProcessor: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("  Shared Memory per Block [KiB]: %d\n", prop.sharedMemPerBlock/(1<<10));
    printf("  regsPerMultiprocessor [32 bits]: %d\n", prop.regsPerMultiprocessor);
    printf("  Registers per Block [32 bits]: %d\n", prop.regsPerBlock);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("\n");
    printf("  maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
    printf("  Per Block maxThreadsDim[0]: %d\n", prop.maxThreadsDim[0]);
    printf("  Per Block maxThreadsDim[1]: %d\n", prop.maxThreadsDim[1]);
    printf("  Per Block maxThreadsDim[2]: %d\n", prop.maxThreadsDim[2]);
    printf("\n");
    printf("  maxGridSize[0]: %d\n", prop.maxGridSize[0]);
    printf("  maxGridSize[1]: %d\n", prop.maxGridSize[1]);
    printf("  maxGridSize[2]: %d\n", prop.maxGridSize[2]);
    printf("\n");
    printf("  unifiedAddressing: %d\n", prop.unifiedAddressing);
    printf("  maxThreadsPerMultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
    // printf("  singleToDoublePrecisionPerfRatio: %f\n", prop.singleToDoublePrecisionPerfRatio);
    // printf("  : %d\n", prop.);
    // printf("  : %d\n", prop.);
    // printf("  : %d\n", prop.);
    // printf("  : %d\n", prop.);
    // printf("  : %d\n", prop.);
    // printf("  : %d\n", prop.);
    // printf("  : %d\n", prop.);
    // printf("  : %d\n", prop.);
    // printf("  : %d\n", prop.);
    // printf("  : %d\n", prop.);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);


    std::cout << "************************************"<< std::endl;

    int numSMs; cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, i);
    std::cout << "Num of SMs     : "<< numSMs << std::endl;

    int maxGridDimX; cudaDeviceGetAttribute(&maxGridDimX, cudaDevAttrMaxGridDimX, i);
    std::cout << "Max Grid dim X : "<< maxGridDimX << std::endl;
    int maxGridDimY; cudaDeviceGetAttribute(&maxGridDimY, cudaDevAttrMaxGridDimY, i);
    std::cout << "Max Grid dim Y : "<< maxGridDimY << std::endl;
    int maxGridDimZ; cudaDeviceGetAttribute(&maxGridDimZ, cudaDevAttrMaxGridDimZ, i);
    std::cout << "Max Grid dim Z : "<< maxGridDimZ << std::endl;

    int maxBlockDimX; cudaDeviceGetAttribute(&maxBlockDimX, cudaDevAttrMaxBlockDimX, i);
    std::cout << "Max Block dim X : "<< maxBlockDimX << std::endl;
    int maxBlockDimY; cudaDeviceGetAttribute(&maxBlockDimY, cudaDevAttrMaxBlockDimY, i);
    std::cout << "Max Block dim Y : "<< maxBlockDimY << std::endl;
    int maxBlockDimZ; cudaDeviceGetAttribute(&maxBlockDimZ, cudaDevAttrMaxBlockDimZ, i);
    std::cout << "Max Block dim Z : "<< maxBlockDimZ << std::endl;

    printf("\n");
    printf("\n");
  }
}
