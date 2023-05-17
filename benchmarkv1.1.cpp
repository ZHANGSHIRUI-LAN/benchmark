// ginpie_benchmark.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include"mkl.h"
#include"omp.h"
#include "mkl_dfti.h"
#include <complex.h>
#include"stdio.h" 
#include"math.h" 

bool AllClose(float * x, float * y, int size, float eps)
{

	printf("Comparing GPU and Host CPU results...\n\n");

	for (int i = 0; i < size; i++) {
		float err = fabs(x[i] - y[i]);
		if (err > eps) {
			std::cout << "x[" << i << "]=" << x[i] << ", y[" << i << "]=" << y[i] << std::endl;

			std::cout << "error" << std::endl;

			return false;
		}
	}

	printf("success...\n\n");
	return true;
}

void testMultiplication(float *data, float *data_A, float *data_B, int row_a, int col_a, int row_b, int col_b)
{
	float *test = NULL;
	float *test_trans = NULL;

	float temp_value = 0;

	int  counter = 0;

	int len;

	int M, N;

	M = row_b;
	N = col_b;

	len = M * N;


	test = (float*)mkl_malloc(row_a*col_b * sizeof(float), 64);
	test_trans = (float*)mkl_malloc(row_b*col_b * sizeof(float), 64);

	if (test == NULL || test_trans == NULL)
	{
		exit(-1);
	}


	for (int n = 0; n < len; n++) {
		int i = n / N;
		int j = n % N;

		test_trans[M*j + i] = data_B[n];
	}

	for (int i = 0; i < row_a; i++)
	{
		for (int j = 0; j < col_b; j++)
		{
			temp_value = 0;

			for (int k = 0; k < col_a; k++)
			{
				temp_value = temp_value + (data_A[i*col_a + k] * test_trans[j*row_b + k]);
			}

			test[counter] = temp_value;

			counter++;

		}
	}

	AllClose(test, data, row_a*col_b, 1e-2);

	mkl_free(test);
	mkl_free(test_trans);


}

void MatrtixMultiplication(float *data_A, float *data_B, float *data_C, int row_a, int col_a, int row_b, int col_b, int num)
{
	std::cout << "Running Ginpie benchMark for Matrix multiplication!\n";

	const int allignment = 64;

	int  counter;

	double alpha = 1.0;
	double beta = 0.0;

	double s_initial;
	double s_elapsed;


	data_A = (float*)mkl_malloc(row_a*col_a * sizeof(float), allignment);
	data_B = (float*)mkl_malloc(row_b*col_b * sizeof(float), allignment);
	data_C = (float*)mkl_malloc(row_a*col_b * sizeof(float), allignment);

	if (data_A == NULL || data_B == NULL || data_C == NULL)
	{
		exit(-1);
	}

	std::cout << "Data space established !\n";

	for (int i = 0; i < row_a; i++)
	{
		for (int j = 0; j < col_a; j++)
		{
			data_A[i* col_a + j] = (rand() % 100) / 10.0f;
		}

	}

	for (int i = 0; i < row_b; i++)
	{
		for (int j = 0; j < col_b; j++)
		{
			data_B[i*col_b + j] = (rand() % 100) / 10.0f;
		}

	}

	std::cout << "Data  initialized !\n";

	std::cout << " Computing !\n";

	s_initial = dsecnd();
	for (int i = 0; i < num; i++)
	{
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row_a, col_b, col_a, alpha, data_A, col_a, data_B, col_b, beta, data_C, col_b);
	}

	s_elapsed = (dsecnd() - s_initial) / num;

	std::cout << "Computations completed !\n";

	printf("MatrtixMultiplication elapsed CPU time  %.5f seconds  \n", (s_elapsed));

	//testMultiplication(data_C, data_A, data_B, ROW_A, COL_A, ROW_B, COL_B);

	mkl_free(data_A);
	mkl_free(data_B);
	mkl_free(data_C);
}
void MatrtixDotProduct(float *data_A, float *data_B, float *data_C, int row_a, int col_a, int num)
{
	std::cout << "Running Ginpie benchMark for Dot product!\n";

	const int allignment = 64;

	int  counter;

	double s_initial;
	double s_elapsed;


	data_A = (float*)mkl_malloc(row_a*col_a * sizeof(float), allignment);
	data_B = (float*)mkl_malloc(row_a*col_a * sizeof(float), allignment);
	data_C = (float*)mkl_malloc(row_a*col_a * sizeof(float), allignment);

	if (data_A == NULL || data_B == NULL || data_C == NULL)
	{
		exit(-1);
	}

	std::cout << "Data space established !\n";

#pragma omp parallel for
	for (int i = 0; i < row_a; i++)
	{
		for (int j = 0; j < col_a; j++)
		{
			data_A[i* col_a + j] = (rand() % 100) / 10.0f;
			data_B[i*col_a + j] = (rand() % 100) / 10.0f;
		}

	}

	std::cout << "Data  initialized !\n";

	std::cout << " Computing !\n";


	s_initial = omp_get_wtime();
#pragma omp parallel for
	for (int i = 0; i < row_a; i++)
	{
		for (int j = 0; j < col_a; j++)
		{
			for (int k = 0; k < num; k++)
			{
				data_C[i* col_a + j] = data_A[i* col_a + j] * data_B[i* col_a + j] / (k % 12);
			}

		}

	}

	s_elapsed = (omp_get_wtime() - s_initial);

	std::cout << "Computations completed !\n";

	printf("MatrtixDotProduct elapsed CPU time  %.5f seconds  \n", (s_elapsed));

	mkl_free(data_A);
	mkl_free(data_B);
	mkl_free(data_C);
}
void MatrtixAdd(float *data_A, float *data_B, float *data_C, int row_a, int col_a, int num)
{
	std::cout << "Running Ginpie benchMark for Matrtix Add!\n";

	const int allignment = 64;

	int  counter;

	double s_initial;
	double s_elapsed;


	data_A = (float*)mkl_malloc(row_a*col_a * sizeof(float), allignment);
	data_B = (float*)mkl_malloc(row_a*col_a * sizeof(float), allignment);
	data_C = (float*)mkl_malloc(row_a*col_a * sizeof(float), allignment);

	if (data_A == NULL || data_B == NULL || data_C == NULL)
	{
		exit(-1);
	}

	std::cout << "Data space established !\n";

#pragma omp parallel for
	for (int i = 0; i < row_a; i++)
	{
		for (int j = 0; j < col_a; j++)
		{
			data_A[i* col_a + j] = (rand() % 100) / 10.0f;
			data_B[i*col_a + j] = (rand() % 100) / 10.0f;
		}

	}

	std::cout << "Data  initialized !\n";

	std::cout << " Computing !\n";


	s_initial = omp_get_wtime();
#pragma omp parallel for
	for (int i = 0; i < row_a; i++)
	{
		for (int j = 0; j < col_a; j++)
		{
			for (int k = 0; k < num; k++)
			{
				data_C[i* col_a + j] = data_A[i* col_a + j] + data_B[i* col_a + j];
			}

		}

	}

	s_elapsed = (omp_get_wtime() - s_initial);

	std::cout << "Computations completed !\n";

	printf("MatrtixAdd elapsed CPU time  %.5f seconds  \n", (s_elapsed));

	mkl_free(data_A);
	mkl_free(data_B);
	mkl_free(data_C);
}

void TwoDimFFTC2C(MKL_Complex8 *data_A, int row_a, int col_a, int num)
{
	std::cout << "Running Ginpie benchMark for 2D FFT C2C!\n";

	const int allignment = 64;

	int  counter;

	double alpha = 1.0;
	double beta = 0.0;

	double s_initial;
	double s_elapsed;

	DFTI_DESCRIPTOR_HANDLE handle = NULL;

	MKL_LONG status;

	MKL_LONG dim_sizes[2] = { row_a, col_a };


	data_A = (MKL_Complex8*)mkl_malloc(row_a*col_a * sizeof(MKL_Complex8), allignment);

	if (data_A == NULL)
	{
		exit(-1);
	}

	std::cout << "Data space established !\n";

	for (int i = 0; i < row_a; i++)
	{
		for (int j = 0; j < col_a; j++)
		{
			data_A[i* col_a + j].imag = (rand() % 100) / 10.0f;
			data_A[i* col_a + j].real = (rand() % 100) / 10.0f;
		}

	}

	std::cout << "Data  initialized !\n";

	std::cout << " Computing !\n";

	status = DftiCreateDescriptor(&handle, DFTI_SINGLE, DFTI_COMPLEX, 2, dim_sizes);
	status = DftiCommitDescriptor(handle);

	s_initial = dsecnd();
	for (int i = 0; i < num; i++)
	{
		status = DftiComputeForward(handle, data_A);
	}

	s_elapsed = (dsecnd() - s_initial) / num;

	status = DftiFreeDescriptor(&handle);

	std::cout << "Computations completed !\n";

	printf("TwoDimFFTC2C elapsed CPU time  %.5f seconds  \n", (s_elapsed));

	mkl_free(data_A);

}
int main()
{
	std::cout << "welcome to Ginpie benchMark !\n";

	float *ipt_A = NULL;
	float *ipt_B = NULL;
	float *opt = NULL;

	MKL_Complex8*ipt_A_fft = NULL;

	int row_A = 0;
	int col_A = 0;

	int row_B = 0;
	int col_B = 0;

	int num;

	//测试矩阵乘法
	row_A = 1000;//矩阵A的行
	col_A = 1000;//矩阵A的列

	row_B = 1000;//矩阵B的行
	col_B = 1000;//矩阵B的列

	num = 1000;//迭代次数

	//MatrtixMultiplication(ipt_A, ipt_B, opt, row_A, col_A, row_B, col_B,num);

	//测试矩阵点乘
	row_A = 10000;
	col_A = 10000;

	row_B = 10000;
	col_B = 10000;

	num = 1000;

	//MatrtixDotProduct(ipt_A, ipt_B, opt, row_A, col_A,num);

	//矩阵相加
	row_A = 10000;
	col_A = 10000;

	row_B = 10000;
	col_B = 10000;

	num = 1000;

	//MatrtixAdd(ipt_A, ipt_B, opt, row_A, col_A, num);


	//2D FFT C2C 
	row_A = 200;
	col_A = 500;

	num = 1000;
	TwoDimFFTC2C(ipt_A_fft, row_A, col_A, num);


}
