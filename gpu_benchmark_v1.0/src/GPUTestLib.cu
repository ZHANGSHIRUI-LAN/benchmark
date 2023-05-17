
#include "include/GPUTestLib.hpp"
//#include "cub/cub/cub.cuh"
#include <stdio.h>
#include <math_constants.h>

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

	//矩阵加法测试
	__global__  void MatrixAdd(float *ipt_a, float *ipt_b, float *opt, int row, int col)
	{
		const int  tid_x = blockIdx.x * blockDim.x + threadIdx.x;
		const int  tid_y = blockIdx.y * blockDim.y + threadIdx.y;

		const int index = tid_y*col + tid_x;

		float temp = 0;

		if (tid_x<col&&tid_y<row)
		{
			for (int i = 0; i < COUNTER; i++)
			{
				temp = sqrt(ipt_b[index] / (3.14f*4.28f) + 5.26 + ipt_a[index] * powf(107 % 3, 2));

			}
			opt[index] = ipt_a[index] + ipt_b[index] + temp*0.002;
		}
	}

	// 矩阵加法测试
	void CudaKernelOps::CudaMatrixAdd(float *matrix_A, float *matrix_B, float *matrix_C, int row, int col,cudaStream_t st)
	{
		dim3 blockSize(THREAD_NUM_PER_BLOCK_X, THREAD_NUM_PER_BLOCK_Y, 1);
		dim3 gridSize(GetGridSize(col, blockSize.x), GetGridSize(row, blockSize.y), 1);

		MatrixAdd << <gridSize, blockSize, 0, st >> >(matrix_A,matrix_B, matrix_C,row,col);
	}


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