/*
	Copyright © 2022 Melrose-Lbt
	All rights reserved.
	Filename: MemSchedulor.cu
	Description: Memory schedulor. C++ back-end memory operation, define the 
        data create, data releash, data transfer operation. Provide API for
        Python.
	Created by Melrose-Lbt 2022-8-18
*/
#include <iostream>
#include "runtime_api.h"
#include <assert.h>

#define CUDA_CHECK(func)                                                        \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    assert((e == cudaSuccess) || (e == cudaErrorCudartUnloading));             \
  }

extern "C" float *AllocateDeviceData(int size){
	float *dev_data;
	CUDA_CHECK(cudaMalloc((void **)&dev_data, size));
	return dev_data;
}

extern "C" float *AllocateHostData(int size){
	float *host_data = (float *)malloc(size);
	return host_data;
}

extern "C" void FreeDeviceData(float *data){
	CUDA_CHECK(cudaFree(data));
}

extern "C" void FreeHostData(float *data){
	free(data);
}

extern "C" void CopyDataFromTo(float *from_data, float *to_data, Device from, Device to, int size){
	if(from == CPU && to == GPU){
		CUDA_CHECK(cudaMemcpy(to_data, from_data, size, cudaMemcpyHostToDevice));
	}
	else if(from == GPU && to == CPU){
		float *dev_data = (float *)from_data;
		float *host_data = (float *)to_data;

		CUDA_CHECK(cudaMemcpy(host_data, dev_data, size, cudaMemcpyDeviceToHost));
	}
}


extern "C" inline int getSize(int dim, int *shape){
	// float32 by default, 4 bytes
	int size = 1;
	for(int i=0; i < dim; i++){
		size = size * shape[i];
		}
	return size * 4;
}


