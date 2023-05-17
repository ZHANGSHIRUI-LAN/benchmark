// System includes
#include <stdio.h>
#include"include/test.hpp"
#include <timer.h>
#include "include/simpleMultiGPU.h"
#include "include/common.hpp"
#include"include/gpu_resource.hpp"
#include "include/GPUTestLib.hpp"

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int row = 1024*10*2;
const int col = 2048*10*4;
const int DATA_N = row * col;


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	//Solver config
	int i, j, GPU_N;

	TGPUplan plan[MAX_DEVICE_NUM];

	//获取GPU个数
	CHECK_CUDA_ERROR(cudaGetDeviceCount(&GPU_N), "cudaGetDeviceCount");

	GpuResource gpuRes(GPU_N, plan);

	gpuRes.CudaMemEstabulished(row, col, GPU_N, plan);

	CudaKernelOps cudaOps(&gpuRes);


	if (GPU_N > MAX_DEVICE_NUM)
	{
		GPU_N = MAX_DEVICE_NUM;
	}

	printf("CUDA-capable device count: %i\n\n", GPU_N);

	printf("Generating input data...\n\n");

	//Subdividing input data across GPUs
	//Get data sizes for each GPU
	for (i = 0; i < GPU_N; i++)
	{
		plan[i].dataN = (row / GPU_N)*col;
	}

	//Take into account "odd" data sizes
	for (i = 0; i < DATA_N % GPU_N; i++)
	{
		plan[i].dataN++;
	}


	//Create streams for issuing GPU command asynchronously and allocate memory (GPU and System page-locked)
	for (i = 0; i < GPU_N; i++)
	{
		for (j = 0; j <plan[i].dataN; j++)
		{
			plan[i].h_matrix_a->GetPtr()[j] = (float)rand() / (float)RAND_MAX;
			plan[i].h_matrix_b->GetPtr()[j] = (float)rand() / (float)RAND_MAX;
		}
	}

	//Start timing and compute on GPU(s)
	printf("Computing with %d GPUs by CudaMatrixAdd...\n\n", GPU_N);
	StartTimer();

	//Copy data to GPU, launch the kernel and copy data back. All asynchronously
	for (i = 0; i < GPU_N; i++)
	{
		//Set device
		CHECK_CUDA_ERROR(cudaSetDevice(i),"cudaSetDevice");

		//Copy input data from CPU
		CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].d_matrix_a->GetPtr(), plan[i].h_matrix_a->GetPtr(), plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream),"cudaMemcpyAsync");
		CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].d_matrix_b->GetPtr(), plan[i].h_matrix_b->GetPtr(), plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream), "cudaMemcpyAsync");

		dim3 blockSize(THREAD_NUM_PER_BLOCK_X, THREAD_NUM_PER_BLOCK_Y, 1);
		dim3 gridSize((col + blockSize.x - 1) / blockSize.x, (plan[i].dataN / col + blockSize.y - 1) / blockSize.y, 1);
		//Perform GPU computations
		cudaOps.CudaMatrixAdd(plan[i].d_matrix_a->GetPtr(), plan[i].d_matrix_b->GetPtr(), plan[i].d_matrix_c->GetPtr(), plan[i].dataN / col, col, plan[i].stream);

		//getLastCudaError("MatrixAdd() execution failed.\n");

		//Read back GPU results
		CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].h_matrix_c->GetPtr(), plan[i].d_matrix_c->GetPtr(), plan[i].dataN * sizeof(float), cudaMemcpyDeviceToHost, plan[i].stream),"cudaMemcpyAsync");

	}

	//Process GPU results
	for (i = 0; i < GPU_N; i++)
	{
		//Set device
		CHECK_CUDA_ERROR(cudaSetDevice(i),"cudaSetDevice");

		//Wait for all operations to finish
		CHECK_CUDA_ERROR(cudaStreamSynchronize(plan[i].stream),"cudaStreamSynchronize");
	}

	printf("GPU Processing time: %f (s)\n\n", GetTimer() / 1000);

	//printf("  CudaMatrixAdd result Checking....\n\n");

	temp = new Matrix<float>(row, col, MemType::HostMem);
	temp_local = new Matrix<float>(row, col, MemType::HostMem);
	
	for (i = 0; i < GPU_N; i++)
	{
		for (j = 0; j <plan[i].dataN; j++)
		{
			temp->GetPtr()[i*plan[i].dataN + j] = plan[i].h_matrix_c->GetPtr()[j];
		}
	}

	// Compute on Host CPU
	printf("Computing with Host CPU...\n\n");
	
	StartTimer();
	float test_temp;
	for (i = 0; i < GPU_N; i++)
	{
		for (int index = 0; index <plan[i].dataN; index++)
		{
			for (int j = 0; j < COUNTER; j++)
			{
				test_temp = sqrt(plan[i].h_matrix_b->GetPtr()[index]  / (3.14f*4.28f) + 5.26 + plan[i].h_matrix_a->GetPtr()[index] * powf(107 % 3, 2));

			}
			temp_local->GetPtr()[i*plan[i].dataN + index] = plan[i].h_matrix_a->GetPtr()[index] + plan[i].h_matrix_b->GetPtr()[index] + test_temp*0.002;
		}
	}
	printf("CPU Processing time: %f (s)\n\n", GetTimer() / 1000);


	// Compare GPU and CPU results
	AllClose(temp_local->GetPtr(), temp->GetPtr(), DATA_N, 1e-5);

	/////////////////////////////////////////////////////////////////////TRANSPOSE//////////////////////////////////////////////////////////////////////////////////
	printf("Computing with %d GPUs by CudaMatrixTrans...\n\n", GPU_N);
	StartTimer();
	//Copy data to GPU, launch the kernel and copy data back. All asynchronously
	for (i = 0; i < GPU_N; i++)
	{
		//Set device
		CHECK_CUDA_ERROR(cudaSetDevice(i), "cudaSetDevice");

		//Copy input data from CPU
		CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].d_matrix_a->GetPtr(), plan[i].h_matrix_a->GetPtr(), plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream), "cudaMemcpyAsync");
		//CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].d_matrix_b->GetPtr(), plan[i].h_matrix_b->GetPtr(), plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream), "cudaMemcpyAsync");

		dim3 blockSize(THREAD_NUM_PER_BLOCK_X, THREAD_NUM_PER_BLOCK_Y, 1);
		dim3 gridSize((col + blockSize.x - 1) / blockSize.x, (plan[i].dataN / col + blockSize.y - 1) / blockSize.y, 1);
		//Perform GPU computations
		cudaOps.CudaMatrixTrans(plan[i].dataN / col, col, plan[i].d_matrix_a->GetPtr(), plan[i].d_matrix_c->GetPtr(), plan[i].stream);
		plan[i].d_matrix_c->ResetShape(col, row);

		//getLastCudaError("MatrixAdd() execution failed.\n");

		//Read back GPU results
		CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].h_matrix_c->GetPtr(), plan[i].d_matrix_c->GetPtr(), plan[i].dataN * sizeof(float), cudaMemcpyDeviceToHost, plan[i].stream), "cudaMemcpyAsync");
		plan[i].h_matrix_c->ResetShape(col, row);
	}

	//Process GPU results
	for (i = 0; i < GPU_N; i++)
	{
		//Set device
		CHECK_CUDA_ERROR(cudaSetDevice(i), "cudaSetDevice");

		//Wait for all operations to finish
		CHECK_CUDA_ERROR(cudaStreamSynchronize(plan[i].stream), "cudaStreamSynchronize");
	}

	printf("GPU Processing time: %f (s)\n\n", GetTimer() / 1000);

	for (i = 0; i < GPU_N; i++)
	{
		for (j = 0; j <plan[i].dataN; j++)
		{
			temp->GetPtr()[i*plan[i].dataN + j] = plan[i].h_matrix_c->GetPtr()[j];
		}
	}

	// Compute on Host CPU
	printf("Computing with Host CPU...\n\n");
 
		int M = temp->GetRowNum();
		int N = temp->GetColNum();
		int len = M*N;

		// #pragma omp parallel for  firstprivate(M,N) num_threads(6)
		for (int n = 0; n < len; n++) {
			int i = n / N;
			int j = n % N;
			temp_local->GetPtr()[M*j + i] = plan[0].h_matrix_a->GetPtr()[n];
		}

		temp_local->ResetShape(N, M);
	
	printf("CPU Processing time: %f (s)\n\n", GetTimer() / 1000);

	// Compare GPU and CPU results
	AllClose(temp_local->GetPtr(), temp->GetPtr(), DATA_N, 1e-5);

}