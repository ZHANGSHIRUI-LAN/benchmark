
#include "include/GPUTestLib.hpp"
//#include "cub/cub/cub.cuh"
#include <stdio.h>
#include <math_constants.h>
#include "cublas_v2.h"
#include "cuda_runtime.h"

namespace vbfod_gpu
{

	// 构造函数
	CudaKernelOps::CudaKernelOps(GpuResource * gpu_res)
	{
		gpu_res_ = gpu_res;
	}

	// 重载构造函数
	CudaKernelOps::CudaKernelOps(GpuResource * gpu_res, int batch, int nx)
	{
		gpu_res_ = gpu_res;

		tmp_buf_1_ = new Matrix<Complex>(batch, nx, MemType::DeviceMem);
		tmp_buf_2_ = new Matrix<Complex>(batch, nx, MemType::DeviceMem);
	}

	// 重载构造函数
	CudaKernelOps::CudaKernelOps(GpuResource * gpu_res, int batch, int nx, int max_temp_len, int sort_temp_len)
	{
		//gpu_res_ = gpu_res;

		//tmp_buf_1_ = new Matrix<Complex>(batch, nx, MemType::DeviceMem);
		//tmp_buf_2_ = new Matrix<Complex>(batch, nx, MemType::DeviceMem);
		//tmp_offset_sort_end = new Matrix<int>(1, sort_temp_len, MemType::DeviceMem);//TODO:关于这里的参数设置看看 怎么优化下
		//tmp_offset_sort_begin = new Matrix<int>(1, sort_temp_len, MemType::DeviceMem);//TODO:关于这里的参数设置看看 怎么优化下

		//cub::DeviceSegmentedRadixSort::SortKeys((void*)NULL, temp_storage_bytes_sort_, (float *)NULL, (float *)NULL, sort_temp_len*batch, sort_temp_len, tmp_offset_sort_begin->GetPtr(), tmp_offset_sort_end->GetPtr());
		//CHECK_CUDA_ERROR(cudaMalloc(&tmp_buf_sort_, temp_storage_bytes_sort_), "cudaMalloc");
		//cub::DeviceReduce::ArgMax((void *)NULL, temp_storage_bytes_max_, (float *)NULL, (cub::KeyValuePair<int, float>*)NULL, max_temp_len);
		//CHECK_CUDA_ERROR(cudaMalloc(&tmp_buf_max_, temp_storage_bytes_max_), "cudaMalloc");
		//CHECK_CUDA_ERROR(cudaMalloc(&tmp_max_kv_, sizeof(cub::KeyValuePair<int, float>)), "cudaMalloc");

	}

	// 矩阵乘法
	void CudaKernelOps::CudaMatrixMul(float *matrix_A, float *matrix_B, float *matrix_C, int a_row, int a_col,int b_row, int b_col,float alpha,float beta,cublasHandle_t handle)
	{
		cublasSgemm(
			handle,
			CUBLAS_OP_T,    
			CUBLAS_OP_T,    
			a_row,           
			b_col,           
			a_col,           
			&alpha,             
			matrix_A,             
			a_col,           
			matrix_B,             
			b_col,           
			&beta,              
			matrix_C,             
			a_row            
		);
	}

	//矩阵点乘
	__global__  void MatrixDotproduct(float *ipt_a, float *ipt_b, float *opt, int row, int col)
	{
		const int  tid_x = blockIdx.x * blockDim.x + threadIdx.x;
		const int  tid_y = blockIdx.y * blockDim.y + threadIdx.y;

		const int index = tid_y * col + tid_x;

		if (tid_x < col&&tid_y < row)
		{

			opt[index] = ipt_a[index] * ipt_b[index];
		}

	}

	// 矩阵点乘
	void CudaKernelOps::CudaMatrixDotproduct(float *matrix_A, float *matrix_B, float *matrix_C, int row, int col, cudaStream_t st)
	{
		dim3 blockSize(THREAD_NUM_PER_BLOCK_X, THREAD_NUM_PER_BLOCK_Y, 1);
		dim3 gridSize(GetGridSize(col, blockSize.x), GetGridSize(row, blockSize.y), 1);

		MatrixDotproduct << <gridSize, blockSize, 0, st >> > (matrix_A, matrix_B, matrix_C, row, col);
	}


	//矩阵除法
	__global__  void MatrixDivision(float *ipt_a, float *ipt_b, float *opt, int row, int col)
	{
		const int  tid_x = blockIdx.x * blockDim.x + threadIdx.x;
		const int  tid_y = blockIdx.y * blockDim.y + threadIdx.y;

		const int index = tid_y * col + tid_x;

		if (tid_x < col&&tid_y < row)
		{

			opt[index] = ipt_a[index] / ipt_b[index];
		}

	}

	// 矩阵除法
	void CudaKernelOps::CudaMatrixDivision(float *matrix_A, float *matrix_B, float *matrix_C, int row, int col, cudaStream_t st)
	{
		dim3 blockSize(THREAD_NUM_PER_BLOCK_X, THREAD_NUM_PER_BLOCK_Y, 1);
		dim3 gridSize(GetGridSize(col, blockSize.x), GetGridSize(row, blockSize.y), 1);

		MatrixDivision << <gridSize, blockSize, 0, st >> > (matrix_A, matrix_B, matrix_C, row, col);
	}


	//矩阵减法
	__global__  void MatrixSubtraction(float *ipt_a, float *ipt_b, float *opt, int row, int col)
	{
		const int  tid_x = blockIdx.x * blockDim.x + threadIdx.x;
		const int  tid_y = blockIdx.y * blockDim.y + threadIdx.y;

		const int index = tid_y * col + tid_x;

		if (tid_x < col&&tid_y < row)
		{

			opt[index] = ipt_a[index] - ipt_b[index];
		}

	}

	// 矩阵减法
	void CudaKernelOps::CudaMatrixSubtraction(float *matrix_A, float *matrix_B, float *matrix_C, int row, int col, cudaStream_t st)
	{
		dim3 blockSize(THREAD_NUM_PER_BLOCK_X, THREAD_NUM_PER_BLOCK_Y, 1);
		dim3 gridSize(GetGridSize(col, blockSize.x), GetGridSize(row, blockSize.y), 1);

		MatrixSubtraction << <gridSize, blockSize, 0, st >> > (matrix_A, matrix_B, matrix_C, row, col);
	}


	//矩阵加法
	__global__  void MatrixAdd(float *ipt_a, float *ipt_b, float *opt, int row, int col)
	{
		const int  tid_x = blockIdx.x * blockDim.x + threadIdx.x;
		const int  tid_y = blockIdx.y * blockDim.y + threadIdx.y;

		const int index = tid_y*col + tid_x;

		if (tid_x<col&&tid_y<row)
		{

			opt[index] = ipt_a[index] + ipt_b[index] ;
		}

	}

	// 矩阵加法
	void CudaKernelOps::CudaMatrixAdd(float *matrix_A, float *matrix_B, float *matrix_C, int row, int col,cudaStream_t st)
	{
		dim3 blockSize(THREAD_NUM_PER_BLOCK_X, THREAD_NUM_PER_BLOCK_Y, 1);
		dim3 gridSize(GetGridSize(col, blockSize.x), GetGridSize(row, blockSize.y), 1);

		MatrixAdd << <gridSize, blockSize, 0, st >> >(matrix_A,matrix_B, matrix_C,row,col);
	}


	//矩阵转置
	__global__ void KernelMatrixTransposeR(float *odata, const float *idata, const int rows, const int cols)
	{
		__shared__ float block[THREAD_NUM_PER_BLOCK_Y + 1][THREAD_NUM_PER_BLOCK_X];

		unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
		if ((xIndex < cols) && (yIndex < rows))
		{
			unsigned int index_in = yIndex * cols + xIndex;
			block[threadIdx.x][threadIdx.y] = idata[index_in];
		}

		__syncthreads();

		unsigned int newxIndex = blockIdx.y*blockDim.y + threadIdx.x;
		unsigned int newyIndex = blockIdx.x*blockDim.x + threadIdx.y;
		if ((newxIndex < rows) && (newyIndex <cols))
		{
			unsigned int index_out = newyIndex * rows + newxIndex;
			odata[index_out] = block[threadIdx.y][threadIdx.x];
		}
	}

	// GPU上实现转置
	void CudaKernelOps::CudaMatrixTrans(int m, int n, float * input, float *output, cudaStream_t st)
	{
		dim3 blockSize(THREAD_NUM_PER_BLOCK_X, THREAD_NUM_PER_BLOCK_Y, 1);
		dim3 gridSize(GetGridSize(n, blockSize.x), GetGridSize(m, blockSize.y), 1);

		KernelMatrixTransposeR << <gridSize, blockSize, 0, st >> >(output, input, m, n);
	}


} // end of namespace vbfod_gpu 