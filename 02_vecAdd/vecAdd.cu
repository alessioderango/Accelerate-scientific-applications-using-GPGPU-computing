#include <iostream>
#include <cuda.h>
#include <chrono>
using namespace std;

__global__ void vecAdd_GPU(double* A, double* B, double* C, int dim)
{
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if(i < dim)
		C[i] = A[i] + B[i];
}


void vecAdd_CPU(double* A, double* B, double* C, int dim)
{
	for(int i = 0; i < dim; i++)
		C[i] = A[i] + B[i];
}

int main(int argc, char* argv[])
{
	if(argc!=2)
	{
		cout << "ERROR" << endl;
		return 0;
	}

	size_t dim = 1;
	dim = dim << 27;
	cout << "dim :  " << dim << endl;
	int block_size    = atoi(argv[1]);
	int block_numbers = ceil(dim/(float)(block_size));
	
	double *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
	h_A = new double[dim];
	h_B = new double[dim];
	h_C = new double[dim];

	size_t size = dim*sizeof(double);

	cudaMalloc((void**) &d_A, size);
	cudaMalloc((void**) &d_B, size);
	cudaMalloc((void**) &d_C, size);

	for(int i = 0;i < dim; i++)
	{
		h_A[i] = 1.0;
		h_B[i] = 2.0;
		h_C[i] = 0.0;
	}

		//(destination, source, sizeinbyte, type of transfer)
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice); 
	cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice); 

	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	vecAdd_CPU(h_A, h_B, h_C, dim);

	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
	cout << "elapsed time CPU: " << elapsed_seconds.count() << endl;

	std::chrono::time_point<std::chrono::system_clock> start_gpu, end_gpu;
	start_gpu = std::chrono::system_clock::now();

	vecAdd_GPU<<<block_numbers, block_size>>> (d_A, d_B, d_C, dim);
	cudaDeviceSynchronize();

	end_gpu = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds_gpu = end_gpu-start_gpu;
	cout << "elapsed time GPU: " << elapsed_seconds_gpu.count() << endl;

	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        cudaFree(d_A);	
        cudaFree(d_B);	
        cudaFree(d_C);	

	for(int i = 0; i < dim; i++)
	{
		if(h_C[i] != 3)
		{
			cout << "TEST FAILED" << endl;
			cout << h_C[i] << " expected 3 at " << i << endl;
		}
	}
	delete [] h_A;
	delete [] h_B;
	delete [] h_C;

	return 0;

}

