#pragma once 

#include <memory>
#include "include/common.hpp"

namespace vbfod_gpu
{

template<typename Type>
class Matrix
{
public:
  // 删除默认构造函数
  Matrix() = delete;

  // 构造函数
  Matrix(int row, int col, MemType mem_type): row_(row), col_(col), mem_type_(mem_type)
  {
    size_ = row_ * col_;
    buf_size_ = size_ * sizeof(Type);

    if(mem_type_ == MemType::DeviceMem) { // device memory 
      auto myCudaMalloc = [](size_t mySize) { 
        void* ptr; 
        CHECK_CUDA_ERROR(cudaMalloc((void**)&ptr, mySize), "cudaMalloc"); 
        return ptr; 
      };
      auto deleter = [](Type * ptr) { 
        CHECK_CUDA_ERROR(cudaFree(ptr), "cudaFree"); 
      };
      ptr_.reset((Type*)myCudaMalloc(buf_size_), deleter);
    }
    else if (mem_type_ == MemType::HostMem){ // host memory
      auto  myCudaMalloc = [](size_t mySize) { 
        void* ptr; 
        CHECK_CUDA_ERROR(cudaMallocHost((void**)&ptr, mySize), "cudaMallocHost"); 
        return ptr; 
      };
      auto deleter = [](Type * ptr) { 
        CHECK_CUDA_ERROR(cudaFreeHost(ptr), "cudaFreeHost"); 
      };
      ptr_.reset((Type*)myCudaMalloc(buf_size_), deleter);
    }
    else if (mem_type_ == MemType::UnifiedMem) {
        auto  myCudaMalloc = [](size_t mySize) { 
        void* ptr; 
        CHECK_CUDA_ERROR(cudaMallocManaged((void**)&ptr, mySize), "cudaMallocManaged"); 
        return ptr; 
      };
      auto deleter = [](Type * ptr) { 
        CHECK_CUDA_ERROR(cudaFree(ptr), "cudaFree"); 
      };
      ptr_.reset((Type*)myCudaMalloc(buf_size_), deleter);
    }
  }

  // 获取矩阵行数
  int GetRowNum() { return row_; }

  // 获取矩阵列数
  int GetColNum() { return col_; }

  // 获取矩阵内元素个数
  size_t GetSize() { return size_; }

  // 获取缓冲区大小，单位Byte
  size_t GetBufSize() { return buf_size_; }

  // 获取矩阵对应的内存指针
  Type * GetPtr() { return ptr_.get(); }

  // 重载运算符[]
  Type& operator[](const int index) {
      return ptr_.get()[index];
  }

  // 重置矩阵的行和列
  void ResetShape(int row, int col) { 
    try {
      if((row * col * sizeof(Type)) > buf_size_) {
        throw "ResetShape is overflow";
      }
      else {
        row_ = row;
        col_ = col;  
        size_ = row_ * col_;
      }
    } catch (const char* msg) {
     std::cout << msg << ": original buf size=" << buf_size_
      << ", new buf size=" << row * col * sizeof(Type) << std::endl;
    }
  }
  
private:
  int row_; // row number 
  int col_; // col number 
  MemType mem_type_; // true: device memory; false: host memory
  size_t size_; // element number in the matrix 
  size_t buf_size_; // pysical buffer size, in Byte
  std::shared_ptr<Type> ptr_; // allcated buffer pointer
};

}