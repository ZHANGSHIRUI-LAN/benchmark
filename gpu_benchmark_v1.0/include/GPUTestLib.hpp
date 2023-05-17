#pragma once 

#include "include/gpu_resource.hpp"
#include "include/matrix.hpp"
//#include "cub/cub/util_type.cuh"

namespace vbfod_gpu 
{

class CudaKernelOps 
{
public:

  // 构造函数
  CudaKernelOps(GpuResource * gpu_res);
  // 构造函数重载
  CudaKernelOps(GpuResource * gpu_res, int nx, int batch);
  // 构造函数重载
  CudaKernelOps(GpuResource * gpu_res, int nx, int batch, int max_temp_len, int sort_temp_len);

  // 析构函数
  ~CudaKernelOps() {}

  void CudaMatrixAdd(float *matrix_A, float *matrix_B, float *matrix_C, int row, int col, cudaStream_t st);

  void CudaMatrixTrans(int m, int n, float * input, float *output, cudaStream_t st);

  
private:
  GpuResource * gpu_res_ = NULL;

  // 为列向FFT申请的临时缓冲区
  Matrix<Complex> * tmp_buf_1_;
  Matrix<Complex> * tmp_buf_2_;

   void * tmp_buf_sort_=NULL; // 为排序申请临时缓冲区
   size_t temp_storage_bytes_sort_ = 0;
   Matrix<int> * tmp_offset_sort_begin;
   Matrix<int> * tmp_offset_sort_end;

  void * tmp_buf_max_ = NULL; // 为求最大值申请临时缓冲区
  //cub::KeyValuePair<int, float> * tmp_max_kv_;
  size_t temp_storage_bytes_max_ = 0;
};

}

