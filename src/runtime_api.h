/*
	Copyright © 2022 Melrose-Lbt
	All rights reserved.
	Filename: runtime_api.h
	Description: provide C++ interface for python.
	Created by Melrose-Lbt 2022-8-18
*/
#ifndef __MEMSCHEDULOR_H
#define __MEMSCHEDULOR_H
#include "array.h"
#include "operator.h"
#define EXPORT __declspec(dllexport)


extern "C"{

	EXPORT float *AllocateDeviceData(int size);
	EXPORT float *AllocateHostData(int size);
	EXPORT void CopyDataFromTo(float *from_data, float *to_data, Device from, Device to, int size);
	EXPORT void FreeDeviceData(float *data);
	EXPORT void FreeHostData(float *data);
	EXPORT inline int getSize(int dim, int *shape);

	EXPORT void matMul2D(Quark* arr1, Quark* arr2, Quark* arr_target);
	EXPORT void matMul3D(Quark* arr1, Quark* arr2, Quark* arr_target);
	EXPORT void GpuElemMul(Quark* arr1, Quark* arr2, Quark* arr_target);
	EXPORT void GpuElemDiv(Quark* arr1, Quark* arr2, Quark* arr_target);
	EXPORT void GpuElemSqrt(Quark* arr1, Quark* arr_target);
	EXPORT void GpuElemExp(Quark* arr1, Quark* arr_target);
	EXPORT void GpuReLU(Quark* arr1, Quark* arr_target);
	EXPORT void GpuAdd(Quark* arr1, Quark* arr2, Quark* arr_target);
	EXPORT void GpuSub(Quark* arr1, Quark* arr2, Quark* arr_target);
	EXPORT void GpuAddConst(Quark* arr1, Quark* arr2, Quark* arr_target);
	EXPORT void GpuSubConst(Quark* arr1, Quark* arr2, Quark* arr_target);
	EXPORT void GpuDivConst(Quark* arr1, Quark* arr2, Quark* arr_target);
	EXPORT void GpuMulConst(Quark* arr1, Quark* arr2, Quark* arr_target);

	EXPORT void GpuConv1D(Quark* arr, Quark* filter, Quark* arr_target);

	// Abandoned version
	EXPORT void GpuConv2D(Quark* arr, Quark* filter, Quark* arr_target);

	// Cudnn version convolution operation
	EXPORT void cudnnConv2D(Quark* devPtrInput, Quark* devPtrFilter, Quark* devPtrOutput, int pad_height, int pad_width, int stride_height, int stride_width, int dilation_height, int dilation_width);
	EXPORT void cudnnConv2dGetKernelGradient(
												Quark* devPtrInput, 
												Quark* devPtrFilter, 
												Quark* devPtrOutput,
												int pad_height, 
												int pad_width, 
												int stride_height, 
												int stride_width,
												int dilation_height, 
												int dilation_width);
	EXPORT void cudnnConv2dGetDataGradient(
											Quark* devPtrInput, 
											Quark* devPtrFilter, 
											Quark* devPtrOutput,
											int pad_height, 
											int pad_width, 
											int stride_height, 
											int stride_width,
											int dilation_height, 
											int dilation_width);
}

#endif
