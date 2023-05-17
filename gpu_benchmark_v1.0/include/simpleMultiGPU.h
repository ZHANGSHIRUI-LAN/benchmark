#include"include/matrix.hpp"

#ifndef SIMPLEMULTIGPU_H
#define SIMPLEMULTIGPU_H

using namespace vbfod_gpu;

typedef struct
{
    int dataN;

	//用到的数据
	Matrix<float> *h_matrix_a;
	Matrix<float> *h_matrix_b;
	Matrix<float> *h_matrix_c;

	Matrix<float> *d_matrix_a;
	Matrix<float> *d_matrix_b;
	Matrix<float> *d_matrix_c;

	cudaStream_t stream=NULL;
    //Stream for asynchronous command execution
 

} TGPUplan;

//extern "C"
//void launch_reduceKernel(float *d_Result, float *d_Input, int N, int BLOCK_N, int THREAD_N, cudaStream_t &s);

#endif
