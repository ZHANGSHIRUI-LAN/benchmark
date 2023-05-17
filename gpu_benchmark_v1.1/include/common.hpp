
#pragma once 

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <math.h>


namespace vbfod_gpu 
{


#define CHECK_CUDA_ERROR(condition, function) \
  do { \
    cudaError_t err = condition; \
    if(err != cudaSuccess) { \
      std::cout << "CUDA Error: " << cudaGetErrorString(err) << ", in function " << function << "(file " << __FILE__ << ", line "<< __LINE__ << ")"<< std::endl; \
    } \
  }while(0)

#define GET_LAST_CUDA_ERROR \
  do { \
    cudaDeviceSynchronize(); \
    cudaError_t err = cudaGetLastError(); \
    if(err != cudaSuccess) { \
      std::cout << "CUDA Error: " << cudaGetErrorString(err) << "(file " << __FILE__ << ", line "<< __LINE__ << ")"<< std::endl; \
    } \
  }while(0)

static inline const char* cublasGetErrorString(cublasStatus_t status)
{
  switch(status)
  {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "cublas unknown error";
}

#define CHECK_CUBLAS_ERROR(condition, function) \
  do { \
    cublasStatus_t err = condition; \
    if(err != CUBLAS_STATUS_SUCCESS) { \
      std::cout << "CUBLAS Error: " << cublasGetErrorString(err) << ", in function " << function << std::endl; \
    } \
  }while(0)


static inline const char* cufftGetErrorString(cufftResult_t status)
{
  switch(status)
  {
    case CUFFT_SUCCESS: return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE"; 
    case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE"; 
    case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED"; 
    case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED"; 
    case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";
    case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE"; 
    case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR"; 
    case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED"; 
    case CUFFT_LICENSE_ERROR: return "CUFFT_SETUP_FAILED"; 
    case CUFFT_NOT_SUPPORTED: return "CUFFT_NO_WORKSPACE";
  }
  return "cufft unknown error";
}

#define CHECK_CUFFT_ERROR(condition, function) \
  do { \
    cufftResult_t err = condition; \
    if(err != CUFFT_SUCCESS) { \
      std::cout << "CUFFT Error: " << cufftGetErrorString(err) << ", in function " << function << std::endl; \
    } \
  }while(0)
  


#define PI 3.1415926f

inline float deg2rad(float deg) { return (deg*PI/180.0f); }
inline float rad2deg(float rad) { return (rad*180.0f/PI); }
inline int NextPower2(float x) { return (1 << (int)ceil(log2(x))); }

typedef   cufftComplex  Complex;

#define THREAD_NUM_PER_BLOCK_X  32
#define THREAD_NUM_PER_BLOCK_Y  32

#define WARP_SIZE		32
#define BLOCK_SIZE		256
#define MAX_BLOCK_SIZE	1024

#define COUNTER 1 
#define MAX_DEVICE_NUM   32 

inline int GetBlockSize(int size) 
{
  if(size <= 32) {
    return 32;
  }
  else if(size <= 64) {
    return 64;
  }
  else if(size <= 128) {
    return 128;
  }
  else if(size <= 256) {
    return 256;
  }
  else if(size <= 512) {
    return 512;
  }
  else {
    return MAX_BLOCK_SIZE;
  }
}

inline int GetGridSize(int totalSize, int blockSize)
{
  return ((totalSize + blockSize - 1) / blockSize);
}

#define MIN(a, b) a <= b ? a : b
#define MAX(a, b) a >= b ? a : b 

enum MemType {HostMem, DeviceMem, UnifiedMem};

}
