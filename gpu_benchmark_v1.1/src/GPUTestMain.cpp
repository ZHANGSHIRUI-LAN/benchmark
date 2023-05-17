// System includes
#include <stdio.h>
#include"include/test.hpp"
#include "include/timer.h"
#include "include/simpleMultiGPU.h" 
#include "include/common.hpp"
#include"include/gpu_resource.hpp"
#include "include/GPUTestLib.hpp"

////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int row =1024*10*2 ;
const int col = 1024*10*2 ;
const int DATA_N = row * col; 


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
	//Solver config
	int i, j, GPU_N;

	float temp_value;

	int counter = 0;

	TGPUplan plan[MAX_DEVICE_NUM];

	printf("\n===========Data size:%d x %d===========\n", row, col);

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

		//Perform GPU computations
		cudaOps.CudaMatrixAdd(plan[i].d_matrix_a->GetPtr(), plan[i].d_matrix_b->GetPtr(), plan[i].d_matrix_c->GetPtr(), plan[i].dataN / col, col, plan[i].stream);

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
	//printf("Computing with Host CPU...\n\n");
	
	//StartTimer();
	//float test_temp;
	//for (i = 0; i < GPU_N; i++)
	//{
	//	for (int index = 0; index <plan[i].dataN; index++)
	//	{
	//		temp_local->GetPtr()[i*plan[i].dataN + index] = plan[i].h_matrix_a->GetPtr()[index] + plan[i].h_matrix_b->GetPtr()[index];
	//	}
	//}
	//printf("CPU Processing time: %f (s)\n\n", GetTimer() / 1000);


	// Compare GPU and CPU results
	//AllClose(temp_local->GetPtr(), temp->GetPtr(), DATA_N, 1e-5);

	/////////////////////////////////////////////////////////////////////transpose//////////////////////////////////////////////////////////////////////////////////
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

	//// Compute on Host CPU
	//printf("Computing with Host CPU...\n\n");
 //
	//	int M = temp->GetRowNum();
	//	int N = temp->GetColNum();
	//	int len = M*N;

	//	// #pragma omp parallel for  firstprivate(M,N) num_threads(6)
	//	for (int n = 0; n < len; n++) {
	//		int i = n / N;
	//		int j = n % N;
	//		temp_local->GetPtr()[M*j + i] = plan[0].h_matrix_a->GetPtr()[n];
	//	}

	//	temp_local->ResetShape(N, M);
	//
	//printf("CPU Processing time: %f (s)\n\n", GetTimer() / 1000);

	// Compare GPU and CPU results
	//AllClose(temp_local->GetPtr(), temp->GetPtr(), DATA_N, 1e-5);

	/////////////////////////////////////////////////////////////////////subtraction//////////////////////////////////////////////////////////////////////////////////
	printf("Computing with %d GPUs by CudaMatrixSubtraction...\n\n", GPU_N);
	StartTimer();
	//Copy data to GPU, launch the kernel and copy data back. All asynchronously
	for (i = 0; i < GPU_N; i++)
	{
		//Set device
		CHECK_CUDA_ERROR(cudaSetDevice(i), "cudaSetDevice");

		//Copy input data from CPU
		CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].d_matrix_a->GetPtr(), plan[i].h_matrix_a->GetPtr(), plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream), "cudaMemcpyAsync");
		//CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].d_matrix_b->GetPtr(), plan[i].h_matrix_b->GetPtr(), plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream), "cudaMemcpyAsync");

		//Perform GPU computations
		cudaOps.CudaMatrixSubtraction(plan[i].d_matrix_a->GetPtr(), plan[i].d_matrix_b->GetPtr(), plan[i].d_matrix_c->GetPtr(), plan[i].dataN / col, col, plan[i].stream);
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
		for (j = 0; j < plan[i].dataN; j++)
		{
			temp->GetPtr()[i*plan[i].dataN + j] = plan[i].h_matrix_c->GetPtr()[j];
		}
	}

	// Compute on Host CPU
	//printf("Computing with Host CPU...\n\n");

	//for (i = 0; i < GPU_N; i++)
	//{
	//	for (int index = 0; index < plan[i].dataN; index++)
	//	{
	//		temp_local->GetPtr()[i*plan[i].dataN + index] = plan[i].h_matrix_a->GetPtr()[index] - plan[i].h_matrix_b->GetPtr()[index];
	//	}
	//}

	//temp_local->ResetShape(M, N);

	//printf("CPU Processing time: %f (s)\n\n", GetTimer() / 1000);

	//// Compare GPU and CPU results
	//AllClose(temp_local->GetPtr(), temp->GetPtr(), DATA_N, 1e-5);

	/////////////////////////////////////////////////////////////////////division//////////////////////////////////////////////////////////////////////////////////
	printf("Computing with %d GPUs by CudaMatrixDivision...\n\n", GPU_N);
	StartTimer();
	//Copy data to GPU, launch the kernel and copy data back. All asynchronously
	for (i = 0; i < GPU_N; i++)
	{
		//Set device
		CHECK_CUDA_ERROR(cudaSetDevice(i), "cudaSetDevice");

		//Copy input data from CPU
		CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].d_matrix_a->GetPtr(), plan[i].h_matrix_a->GetPtr(), plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream), "cudaMemcpyAsync");
		//CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].d_matrix_b->GetPtr(), plan[i].h_matrix_b->GetPtr(), plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream), "cudaMemcpyAsync");

		//Perform GPU computations
		cudaOps.CudaMatrixDivision(plan[i].d_matrix_a->GetPtr(), plan[i].d_matrix_b->GetPtr(), plan[i].d_matrix_c->GetPtr(), plan[i].dataN / col, col, plan[i].stream);
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
		for (j = 0; j < plan[i].dataN; j++)
		{
			temp->GetPtr()[i*plan[i].dataN + j] = plan[i].h_matrix_c->GetPtr()[j];
		}
	}

	// Compute on Host CPU
	//printf("Computing with Host CPU...\n\n");

	//for (i = 0; i < GPU_N; i++)
	//{
	//	for (int index = 0; index < plan[i].dataN; index++)
	//	{
	//		temp_local->GetPtr()[i*plan[i].dataN + index] = plan[i].h_matrix_a->GetPtr()[index] / plan[i].h_matrix_b->GetPtr()[index];
	//	}
	//}

	//temp_local->ResetShape(M, N);

	//printf("CPU Processing time: %f (s)\n\n", GetTimer() / 1000);

	//// Compare GPU and CPU results
	//AllClose(temp_local->GetPtr(), temp->GetPtr(), DATA_N, 1e-5);

	/////////////////////////////////////////////////////////////////////multiplication//////////////////////////////////////////////////////////////////////////////////

	printf("Computing with %d GPUs by CudaMatrixMul...\n\n", GPU_N);
	StartTimer();
	//Copy data to GPU, launch the kernel and copy data back. All asynchronously
	for (i = 0; i < GPU_N; i++)
	{
		//Set device
		CHECK_CUDA_ERROR(cudaSetDevice(i), "cudaSetDevice");


		//Copy input data from CPU
		CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].d_matrix_a->GetPtr(), plan[i].h_matrix_a->GetPtr(), plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream), "cudaMemcpyAsync");
		CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].d_matrix_b->GetPtr(), plan[i].h_matrix_b->GetPtr(), plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream), "cudaMemcpyAsync");

		//Perform GPU computations
		cudaOps.CudaMatrixMul(plan[i].d_matrix_a->GetPtr(), plan[i].d_matrix_b->GetPtr(), plan[i].d_matrix_c->GetPtr(), row, col, col,row,1,0, plan[i].handle);
		cudaOps.CudaMatrixTrans(plan[i].d_matrix_c->GetRowNum(), plan[i].d_matrix_c->GetColNum(), plan[i].d_matrix_c->GetPtr(), plan[i].d_matrix_c_trans->GetPtr(),plan[i].stream);
		plan[i].d_matrix_c_trans->ResetShape(col, row);

		//getLastCudaError("MatrixAdd() execution failed.\n");

		//Read back GPU results
		CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].h_matrix_c->GetPtr(), plan[i].d_matrix_c_trans->GetPtr(), col*row * sizeof(float), cudaMemcpyDeviceToHost, plan[i].stream), "cudaMemcpyAsync");
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
		for (j = 0; j < plan[i].dataN; j++)
		{
			temp->GetPtr()[i*plan[i].dataN + j] = plan[i].h_matrix_c->GetPtr()[j];
		}
	}

	// Compute on Host CPU
	//printf("Computing with Host CPU...\n\n"); 

	// M = plan[0].h_matrix_b->GetRowNum();
	// N = plan[0].h_matrix_b->GetColNum();
	// len = M * N;


	//// #pragma omp parallel for  firstprivate(M,N) num_threads(6)
	//for (int n = 0; n < len; n++) {
	//	int i = n / N;
	//	int j = n % N;
	//	
	//	plan[0].h_matrix_b_tran->GetPtr()[M*j + i] = plan[0].h_matrix_b->GetPtr()[n];
	//}

	//plan[0].h_matrix_b_tran->ResetShape(N, M);

	//counter = 0;
	//for (int i=0;i< plan[0].h_matrix_a->GetRowNum();i++)
	//{
	//	for (int j = 0; j < plan[0].h_matrix_b_tran->GetRowNum(); j++)
	//	{
	//		temp_value = 0;

	//		for (int k = 0; k < plan[0].h_matrix_a->GetColNum(); k++)
	//		{
	//			temp_value = temp_value +( plan[0].h_matrix_a->GetPtr()[i*plan[0].h_matrix_a->GetColNum() + k] * plan[0].h_matrix_b_tran->GetPtr()[j*plan[0].h_matrix_b_tran->GetColNum() + k]);
	//		}

	//		temp_local->GetPtr()[counter]= temp_value;

	//		counter++;
	//		 
	//	}
	//}
	//

	//printf("CPU Processing time: %f (s)\n\n", GetTimer() / 1000);

	//// Compare GPU and CPU results
	//AllClose(temp_local->GetPtr(), temp->GetPtr(), DATA_N, 1e-3);

	/////////////////////////////////////////////////////////////////////Dotproduct//////////////////////////////////////////////////////////////////////////////////
	printf("Computing with %d GPUs by CudaMatrixDotproduct...\n\n", GPU_N);
	StartTimer();
	//Copy data to GPU, launch the kernel and copy data back. All asynchronously
	for (i = 0; i < GPU_N; i++)
	{
		//Set device
		CHECK_CUDA_ERROR(cudaSetDevice(i), "cudaSetDevice");

		//Copy input data from CPU
		CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].d_matrix_a->GetPtr(), plan[i].h_matrix_a->GetPtr(), plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream), "cudaMemcpyAsync");
		//CHECK_CUDA_ERROR(cudaMemcpyAsync(plan[i].d_matrix_b->GetPtr(), plan[i].h_matrix_b->GetPtr(), plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream), "cudaMemcpyAsync");

		//Perform GPU computations
		cudaOps.CudaMatrixDotproduct(plan[i].d_matrix_a->GetPtr(), plan[i].d_matrix_b->GetPtr(), plan[i].d_matrix_c->GetPtr(), plan[i].dataN / col, col, plan[i].stream);
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
		for (j = 0; j < plan[i].dataN; j++)
		{
			temp->GetPtr()[i*plan[i].dataN + j] = plan[i].h_matrix_c->GetPtr()[j];
		}
	}

	// Compute on Host CPU
	//printf("Computing with Host CPU...\n\n");

	//for (i = 0; i < GPU_N; i++)
	//{
	//	for (int index = 0; index < plan[i].dataN; index++)
	//	{
	//		temp_local->GetPtr()[i*plan[i].dataN + index] = plan[i].h_matrix_a->GetPtr()[index] * plan[i].h_matrix_b->GetPtr()[index];
	//	}
	//}

	//temp_local->ResetShape(M, N);

	//printf("CPU Processing time: %f (s)\n\n", GetTimer() / 1000);

	//// Compare GPU and CPU results
	//AllClose(temp_local->GetPtr(), temp->GetPtr(), DATA_N, 1e-5);

}