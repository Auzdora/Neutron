/*
	Copyright © 2022 Melrose-Lbt
	All rights reserved.
	Filename: operator.cu
	Description: CUDA accelerated matrix operation.
	Created by Melrose-Lbt 2022-8-17
*/
#include "operator.h"
#include <iostream>
#include <cudnn.h>
#include <assert.h>
#define CUDA_CHECK(func)                                                       \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    assert((e == cudaSuccess) || (e == cudaErrorCudartUnloading));             \
  }


/* Function name: matrix_multiply_2d
   Params       : float* arr1, float* arr2, float* target, size_t M, size_t N, size_t K
   Description  : Kernel function of CUDA, multiply 2d matrix.
   return       : None
*/

__global__ void matrix_multiply_2d(float* arr1, float* arr2, float* target, size_t M, size_t N, size_t K){
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if(row < M && col < K){
        int temp = 0;
        for(int i = 0; i < N; i++){
            temp += arr1[N * row + i] * arr2[i * K + col];
        }

        target[col + K * row] = temp;
    }
}

extern "C" void matMul2D(Quark* arr1, Quark* arr2, Quark* arr_target){
    // M * N and N * K
    int M = arr1->shape[0];
    int N = arr1->shape[1];
    int K = arr2->shape[1];

    constexpr const int TP = 16;
	dim3 threads(TP, TP, 1);
	dim3 blocks((M + TP - 1) / TP, (K + TP - 1) / TP, 1);

    matrix_multiply_2d <<<blocks, threads>>> (arr1->data, arr2->data, arr_target->data, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}


/* Function name: matrix_multiply_3d
   Params       : float* arr1, float* arr2, float* target, size_t B, size_t M, size_t N, size_t K
   Description  : Kernel function of CUDA, multiply 2d matrix.
   return       : None
*/

__global__ void matrix_multiply_3d(float* arr1, float* arr2, float* target, size_t B, size_t M, size_t N, size_t K){
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int height = blockIdx.z;

    if(row < M && col < K && height < B){
        int temp = 0;
        for(int i = 0; i < N; i++){
            temp += arr1[height * M * N + N * row + i] * arr2[height * K * N + i * K + col];
        }

        target[col + K * row + height * M * K] = temp;
    }
}

extern "C" void matMul3D(Quark* arr1, Quark* arr2, Quark* arr_target){
    // M * N and N * K
    int B = arr1->shape[0];
    int M = arr1->shape[1];
    int N = arr1->shape[2];
    int K = arr2->shape[2];

    constexpr const int TP = 16;
	dim3 threads(TP, TP, 1);
	dim3 blocks((M + TP - 1) / TP, (K + TP - 1) / TP, K);

    matrix_multiply_3d <<<blocks, threads>>> (arr1->data, arr2->data, arr_target->data, B, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}
 
/* Function name: element_wise_multiply
   Params       : float* arr1, float* arr2, float* target, size_t M, size_t N
   Description  : Kernel function of CUDA, element wise/pixel wise multiply 2d.
   return       : None
*/
__global__ void element_wise_multiply(float* arr1, float* arr2, float* target, size_t n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < n){
        target[idx] = arr1[idx] * arr2[idx];
    }
}

extern "C" void GpuElemMul(Quark* arr1, Quark* arr2, Quark* arr_target){
    int n = 1;
    for (int i = 0; i < arr1->dim; i++) {
        n *= arr1->shape[i];
    }

    constexpr const int TP = 1024;
	dim3 threads(TP, 1, 1);
	dim3 blocks((n + TP - 1) / TP, 1, 1);

    element_wise_multiply <<<blocks, threads>>> (arr1->data, arr2->data, arr_target->data, n);
    CUDA_CHECK(cudaGetLastError());
}

/* Function name: element_wise_div
   Params       : float* arr1, float* arr2, float* target, size_t M, size_t N
   Description  : Kernel function of CUDA, element wise/pixel wise div 2d.
   return       : None
*/
__global__ void element_wise_div(float* arr1, float* arr2, float* target, size_t n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < n){
        target[idx] = arr1[idx] / arr2[idx];
    }
}

extern "C" void GpuElemDiv(Quark* arr1, Quark* arr2, Quark* arr_target){
    int n = 1;
    for (int i = 0; i < arr1->dim; i++) {
        n *= arr1->shape[i];
    }

    constexpr const int TP = 1024;
	dim3 threads(TP, 1, 1);
	dim3 blocks((n + TP - 1) / TP, 1, 1);

    element_wise_div <<<blocks, threads>>> (arr1->data, arr2->data, arr_target->data, n);
    CUDA_CHECK(cudaGetLastError());
}

/* Function name: element_wise_sqrt
   Params       : float* arr1, float* arr2, float* target, size_t M, size_t N
   Description  : Kernel function of CUDA, element wise/pixel wise sqrt 2d.
   return       : None
*/
__global__ void element_wise_sqrt(float* arr1, float* target, size_t n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < n){
        target[idx] = sqrt(arr1[idx]);
    }
}

extern "C" void GpuElemSqrt(Quark* arr1, Quark* arr_target){
    int n = 1;
    for (int i = 0; i < arr1->dim; i++) {
        n *= arr1->shape[i];
    }

    constexpr const int TP = 1024;
	dim3 threads(TP, 1, 1);
	dim3 blocks((n + TP - 1) / TP, 1, 1);

    element_wise_sqrt <<<blocks, threads>>> (arr1->data, arr_target->data, n);
    CUDA_CHECK(cudaGetLastError());
}

/* Function name: element_wise_exp
   Params       : float* arr1, float* arr2, float* target, size_t M, size_t N
   Description  : Kernel function of CUDA, element wise/pixel wise exp.
   return       : None
*/
__global__ void element_wise_exp(float* arr1, float* target, size_t n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < n){
        target[idx] = exp(arr1[idx]);
    }
}

extern "C" void GpuElemExp(Quark* arr1, Quark* arr_target){
    int n = 1;
    for (int i = 0; i < arr1->dim; i++) {
        n *= arr1->shape[i];
    }

    constexpr const int TP = 1024;
	dim3 threads(TP, 1, 1);
	dim3 blocks((n + TP - 1) / TP, 1, 1);

    element_wise_exp <<<blocks, threads>>> (arr1->data, arr_target->data, n);
    CUDA_CHECK(cudaGetLastError());
}

/* Function name: relu_kernel
   Params       : float* arr1, float* arr2, float* target, size_t M, size_t N
   Description  : Kernel function of CUDA, element wise/pixel wise relu 2d.
   return       : None
*/
__global__ void relu_kernel(float* arr1, float* target, size_t n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < n){
        float data = arr1[idx];
        if(data <= 0) target[idx] = 0;
        else target[idx] = data;
    }
}

extern "C" void GpuReLU(Quark* arr1, Quark* arr_target){
    int n = 1;
    for (int i = 0; i < arr1->dim; i++) {
        n *= arr1->shape[i];
    }

    constexpr const int TP = 1024;
	dim3 threads(TP, 1, 1);
	dim3 blocks((n + TP - 1) / TP, 1, 1);

    relu_kernel <<<blocks, threads>>> (arr1->data, arr_target->data, n);
    CUDA_CHECK(cudaGetLastError());
}

/* Function name: add_kernel
   Params       : float* arr1, float* arr2, float* target, size_t M, size_t N
   Description  : Kernel function of CUDA, element wise/pixel wise add.
   return       : None
*/
__global__ void add_kernel(float* arr1, float* arr2, float* target, size_t n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < n){
        target[idx] = arr1[idx] + arr2[idx];
    }
}

extern "C" void GpuAdd(Quark* arr1, Quark* arr2, Quark* arr_target){
    // check matrix shape before call this function, has to be same
    int n = 1;
    for (int i = 0; i < arr1->dim; i++) {
        n *= arr1->shape[i];
    }

    constexpr const int TP = 1024;
	dim3 threads(TP, 1, 1);
	dim3 blocks((n + TP - 1) / TP, 1, 1);

    add_kernel <<<blocks, threads>>> (arr1->data, arr2->data, arr_target->data, n);
    CUDA_CHECK(cudaGetLastError());
}

/* Function name: sub_kernel
   Params       : float* arr1, float* arr2, float* target, size_t M, size_t N
   Description  : Kernel function of CUDA, element wise/pixel wise sub.
   return       : None
*/
__global__ void sub_kernel(float* arr1, float* arr2, float* target, size_t n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < n){
        target[idx] = arr1[idx] - arr2[idx];
    }
}

extern "C" void GpuSub(Quark* arr1, Quark* arr2, Quark* arr_target){
    // check matrix shape before call this function, has to be same
    int n = 1;
    for (int i = 0; i < arr1->dim; i++) {
        n *= arr1->shape[i];
    }

    constexpr const int TP = 1024;
	dim3 threads(TP, 1, 1);
	dim3 blocks((n + TP - 1) / TP, 1, 1);

    sub_kernel <<<blocks, threads>>> (arr1->data, arr2->data, arr_target->data, n);
    CUDA_CHECK(cudaGetLastError());
}

/* Function name: add_const_kernel
   Params       : float* arr1, float* arr2, float* target, size_t n.
   Description  : Kernel function of CUDA, element wise/pixel wise add constant.
   return       : None
*/
__global__ void add_const_kernel(float* arr1, float* arr2, float* target, size_t n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < n){
        target[idx] = arr1[idx] + arr2[0];
    }
}

extern "C" void GpuAddConst(Quark* arr1, Quark* arr2, Quark* arr_target){
    // check matrix shape before call this function, has to be same
    int n = 1;
    for (int i = 0; i < arr1->dim; i++) {
        n *= arr1->shape[i];
    }

    constexpr const int TP = 1024;
	dim3 threads(TP, 1, 1);
	dim3 blocks((n + TP - 1) / TP, 1, 1);

    add_const_kernel <<<blocks, threads>>> (arr1->data, arr2->data, arr_target->data, n);
    CUDA_CHECK(cudaGetLastError());
}

/* Function name: sub_const_kernel
   Params       : float* arr1, float* arr2, float* target, size_t n.
   Description  : Kernel function of CUDA, element wise/pixel wise sub constant.
   return       : None
*/
__global__ void sub_const_kernel(float* arr1, float* arr2, float* target, size_t n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < n){
        target[idx] = arr1[idx] - arr2[0];
    }
}

extern "C" void GpuSubConst(Quark* arr1, Quark* arr2, Quark* arr_target){
    // check matrix shape before call this function, has to be same
    int n = 1;
    for (int i = 0; i < arr1->dim; i++) {
        n *= arr1->shape[i];
    }

    constexpr const int TP = 1024;
	dim3 threads(TP, 1, 1);
	dim3 blocks((n + TP - 1) / TP, 1, 1);

    sub_const_kernel <<<blocks, threads>>> (arr1->data, arr2->data, arr_target->data, n);
    CUDA_CHECK(cudaGetLastError());
}

/* Function name: div_const_kernel
   Params       : float* arr1, float* arr2, float* target, size_t n.
   Description  : Kernel function of CUDA, element wise/pixel wise div constant.
   return       : None
*/
__global__ void div_const_kernel(float* arr1, float* arr2, float* target, size_t n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < n){
        target[idx] = arr1[idx] / arr2[0];
    }
}

extern "C" void GpuDivConst(Quark* arr1, Quark* arr2, Quark* arr_target){
    // check matrix shape before call this function, has to be same
    int n = 1;
    for (int i = 0; i < arr1->dim; i++) {
        n *= arr1->shape[i];
    }

    constexpr const int TP = 1024;
	dim3 threads(TP, 1, 1);
	dim3 blocks((n + TP - 1) / TP, 1, 1);

    div_const_kernel <<<blocks, threads>>> (arr1->data, arr2->data, arr_target->data, n);
    CUDA_CHECK(cudaGetLastError());
}

/* Function name: mul_const_kernel
   Params       : float* arr1, float* arr2, float* target, size_t n.
   Description  : Kernel function of CUDA, element wise/pixel wise div constant.
   return       : None
*/
__global__ void mul_const_kernel(float* arr1, float* arr2, float* target, size_t n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < n){
        target[idx] = arr1[idx] * arr2[0];
    }
}

extern "C" void GpuMulConst(Quark* arr1, Quark* arr2, Quark* arr_target){
    // check matrix shape before call this function, has to be same
    int n = 1;
    for (int i = 0; i < arr1->dim; i++) {
        n *= arr1->shape[i];
    }

    constexpr const int TP = 1024;
	dim3 threads(TP, 1, 1);
	dim3 blocks((n + TP - 1) / TP, 1, 1);

    mul_const_kernel <<<blocks, threads>>> (arr1->data, arr2->data, arr_target->data, n);
    CUDA_CHECK(cudaGetLastError());
}

/* Function name: 1d_conv_kernel
   Params       : float* arr1, float* arr2, float* target, size_t n.
   Description  : Kernel function of CUDA, 1 dimensional convolution.
   return       : None
*/
__global__ void conv_kernel_1d(float* arr, float* filter, float* result, size_t n, int len_filter){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < n){
        int temp = 0;
        int r = len_filter/2;
        for(int i=0; i<len_filter; i++){
            if((idx - r + i) >= 0 && (idx - r + i) < n) 
                temp += arr[idx - r + i] * filter[i];
        }

        result[idx] = temp;
    }
}

extern "C" void GpuConv1D(Quark* arr, Quark* filter, Quark* arr_target){
    // check matrix shape before call this function, has to be same
    int n = 1;
    for (int i = 0; i < arr->dim; i++) {
        n *= arr->shape[i];
    }

    constexpr const int TP = 1024;
	dim3 threads(TP, 1, 1);
	dim3 blocks((n + TP - 1) / TP, 1, 1);

    conv_kernel_1d <<<blocks, threads>>> (arr->data, filter->data, arr_target->data, n, filter->shape[0]);
    CUDA_CHECK(cudaGetLastError());
}

/* Function name: 2d_conv_kernel
   Params       : float* arr1, float* filter, float* target, size_t n.
   Description  : Kernel function of CUDA, 1 dimensional convolution.
   return       : None
*/
__global__ void conv_kernel_2d(float* arr, float* filter, float* result, int high, int width, int filter_size)
{
	int Col = blockIdx.x*blockDim.x + threadIdx.x;
	int Row = blockIdx.y*blockDim.y + threadIdx.y;
	if (Row < high && Col < width)
	{
		float pixVal = 0;
		//start
		int startCol = Col - filter_size / 2;
		int startRow = Row - filter_size / 2;	
		//caculate the res
		for (int i = 0; i < filter_size; i++)
		{
			for (int j = 0; j < filter_size; j++)
			{
				int curRow = startRow + i;
				int curCol = startCol + j;
				if (curRow > -1 && curRow < high && curCol>-1 && curCol < width)
				{
					pixVal += filter[i*filter_size + j] * arr[curRow*width + curCol];
				}
			}
		}
		result[Row*width + Col] = pixVal;
	}
}

extern "C" void GpuConv2D(Quark* arr, Quark* filter, Quark* arr_target){

    int kernel_size = filter->shape[0];
    int high = arr->shape[0];
    int width = arr->shape[1];
    constexpr const int TP = 16;
	dim3 threads(TP, TP, 1);
	dim3 blocks((filter->shape[0] + TP - 1) / TP, (filter->shape[0] + TP - 1) / TP, 1);

    conv_kernel_2d <<<blocks, threads>>> (arr->data, filter->data, arr_target->data, high, width, kernel_size);
    CUDA_CHECK(cudaGetLastError());
}

