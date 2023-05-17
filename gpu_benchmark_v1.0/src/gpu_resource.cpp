#include "include/common.hpp"
#include "include/gpu_resource.hpp"
#include "include/matrix.hpp"

namespace vbfod_gpu
{

// 构造函数
GpuResource::GpuResource()
{


}

// 构造函数重载
GpuResource::GpuResource(int gpu_number, TGPUplan *plan)
{
	for (int i = 0; i <gpu_number; i++)
	{
		CHECK_CUDA_ERROR(cudaSetDevice(i), "cudaSetDevice");
		CHECK_CUDA_ERROR(cudaStreamCreate(&plan[i].stream), "cudaStreamCreate");
	}
}

// 析构函数
GpuResource::~GpuResource()
{
	for (int i = 0; i < 32; i++)
	{
		//Shut down this GPU

		if (plan[i].stream)
		{
			CHECK_CUDA_ERROR(cudaSetDevice(i),"cudaSetDevice");
			CHECK_CUDA_ERROR(cudaStreamDestroy(plan[i].stream),"cudaStreamDestroy");
		}
		else
		{
			break;
		}

	}
}

//创建内存
void GpuResource::CudaMemEstabulished(int row,int col,int gpu_number,TGPUplan * plan)
{
	for (int  i = 0; i <gpu_number; i++)
	{
		plan[i].h_matrix_a = new Matrix<float>(row, col, MemType::HostMem);
		plan[i].h_matrix_a->ResetShape(row, col);

		plan[i].h_matrix_b = new Matrix<float>(row, col, MemType::HostMem);
		plan[i].h_matrix_b->ResetShape(row, col);

		plan[i].h_matrix_c = new Matrix<float>(row, col, MemType::HostMem);
		plan[i].h_matrix_c->ResetShape(row, col);

		plan[i].d_matrix_a = new Matrix<float>(row, col, MemType::DeviceMem);
		plan[i].d_matrix_a->ResetShape(row, col);

		plan[i].d_matrix_b = new Matrix<float>(row, col, MemType::DeviceMem);
		plan[i].d_matrix_b->ResetShape(row, col);

		plan[i].d_matrix_c = new Matrix<float>(row, col, MemType::DeviceMem);
		plan[i].d_matrix_c->ResetShape(row, col);
	}

 
}

}