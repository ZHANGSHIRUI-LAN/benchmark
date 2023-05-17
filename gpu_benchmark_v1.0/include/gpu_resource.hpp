#pragma once 

#include <cublas_v2.h>
#include <cufft.h>
#include "include/common.hpp"
#include"simpleMultiGPU.h"

namespace vbfod_gpu
{

class GpuResource 
{
public:

  // 默认构造函数
   GpuResource( );

   //构造函数重载
   GpuResource(int gpu_number, TGPUplan *plan);

  // 析构函数
  ~GpuResource();
   
 //计算要用到的一切数据资源
   TGPUplan  plan[MAX_DEVICE_NUM];

   //开辟内存
   void CudaMemEstabulished(int row, int col, int gpu_number, TGPUplan * plan);

  // 设置新的DevID，并返回当前devID
  int GetSetDevice(int devID) {
    int o_devID;
    CHECK_CUDA_ERROR(cudaSetDevice(devID), "cudaSetDevice");
	CHECK_CUDA_ERROR(cudaGetDevice(&o_devID), "cudaGetDevice");

    return o_devID; 
  }
};

}
