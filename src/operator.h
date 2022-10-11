/*
	Copyright © 2022 Melrose-Lbt
	All rights reserved.
	Filename: operator.h
	Description: CUDA accelerated matrix operation.
	Created by Melrose-Lbt 2022-8-17
*/
#ifndef __OPERATOR_H
#define __OPERATOR_H

#define EXPORT __declspec(dllexport)
#include "array.h"

extern "C"{

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
	EXPORT void GpuConv2D(Quark* arr, Quark* filter, Quark* arr_target);
	
}


#endif
