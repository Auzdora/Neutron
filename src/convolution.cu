/*
	Copyright © 2022 Melrose-Lbt
	All rights reserved.
	Filename: convolution.cu
	Description: CUDA accelerated convolution computation API.
	Created by Melrose-Lbt 2022-8-28
*/
#include "cudnn.h"
#include "runtime_api.h"
#include <assert.h>
#include "array.h"
#include <iostream>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }


extern "C" void cudnnConv2D(
    Quark* devPtrInput, 
    Quark* devPtrFilter, 
    Quark* devPtrOutput,
    int pad_height, 
    int pad_width, 
    int stride_height, 
    int stride_width,
    int dilation_height, 
    int dilation_width){

    // define and create context handle
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // define descriptor handle
    cudnnTensorDescriptor_t input_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;

    //create descriptor handle
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));

    //config descriptor
    int batch_size = devPtrInput->shape[0];
    int in_channel = devPtrInput->shape[1];
    int in_height = devPtrInput->shape[2];
    int in_width = devPtrInput->shape[3];
    // printf("batch:%d, in_c:%d, in_h:%d, in_w:%d\n", batch_size, in_channel, in_height, in_width);

    int filter_out = devPtrFilter->shape[0];
    int filter_in = devPtrFilter->shape[1];
    int kernel_size = devPtrFilter->shape[2];
    // printf("filter_out:%d, filter_in:%d, kernel_size:%d\n", filter_out, filter_in, kernel_size);

    int out_channel = devPtrOutput->shape[1];
    int out_height = devPtrOutput->shape[2];
    int out_width = devPtrOutput->shape[3];
    // printf("out_channel:%d, out_height:%d, out_width:%d\n", out_channel, out_height, out_width);


    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/batch_size,
                                        /*channels=*/in_channel,
                                        /*image_height=*/in_height,
                                        /*image_width=*/in_width));

    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/batch_size,
                                        /*channels=*/out_channel,
                                        /*image_height=*/out_height,
                                        /*image_width=*/out_width));

    checkCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/filter_out,
                                        /*in_channels=*/filter_in,
                                        /*kernel_height=*/kernel_size,
                                        /*kernel_width=*/kernel_size));

    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                        /*pad_height=*/pad_height,
                                        /*pad_width=*/pad_width,
                                        /*vertical_stride=*/stride_height,
                                        /*horizontal_stride=*/stride_width,
                                        /*dilation_height=*/dilation_height,
                                        /*dilation_width=*/dilation_width,
                                        /*mode=*/CUDNN_CONVOLUTION,
                                        /*computeType=*/CUDNN_DATA_FLOAT));
    
    // explanation for CUDNN_CROSS_CORRELATION and CUDNN_CONVOLUTION 
    // refer to https://zhuanlan.zhihu.com/p/33194385

    // choose convolution algorithm
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    // set workspace
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                    input_descriptor,
                                                    filter_descriptor,
                                                    convolution_descriptor,
                                                    output_descriptor,
                                                    algo,
                                                    &workspace_bytes));
    // std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
    //        << std::endl;

    // allocate workspace memory
    void *workSpace = 0;
    if (workspace_bytes > 0) {
        cudaMalloc(&workSpace, workspace_bytes);
    }

    // compute
    float alpha = 1, beta = 0;
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                        &alpha,
                                        input_descriptor,
                                        devPtrInput->data,
                                        filter_descriptor,
                                        devPtrFilter->data,
                                        convolution_descriptor,
                                        algo,
                                        workSpace,
                                        workspace_bytes,
                                        &beta,
                                        output_descriptor,
                                        devPtrOutput->data));
    
    // release the memory
    cudaFree(workSpace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);

}

/*
  @param: devPtrInput should be NCHW format, it is the input data while compute forward.
  @param: devPtrFilter should be NCHW format, 
*/
extern "C" void cudnnConv2dGetKernelGradient(
    Quark* devPtrInput, 
    Quark* devPtrFilter, 
    Quark* devPtrOutput,
    int pad_height, 
    int pad_width, 
    int stride_height, 
    int stride_width,
    int dilation_height, 
    int dilation_width){
    
    // define and create context handle
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // define descriptor handle
    cudnnTensorDescriptor_t input_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;

    //create descriptor handle
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));

    //config descriptor
    int batch_size = devPtrInput->shape[0];
    int in_channel = devPtrInput->shape[1];
    int in_height = devPtrInput->shape[2];
    int in_width = devPtrInput->shape[3];
    // printf("batch:%d, in_c:%d, in_h:%d, in_w:%d\n", batch_size, in_channel, in_height, in_width);

    int filter_out = devPtrFilter->shape[0];
    int filter_in = devPtrFilter->shape[1];
    int kernel_size = devPtrFilter->shape[2];
    // printf("filter_out:%d, filter_in:%d, kernel_size:%d\n", filter_out, filter_in, kernel_size);

    int out_N = devPtrOutput->shape[0];
    int out_channel = devPtrOutput->shape[1];
    int out_height = devPtrOutput->shape[2];
    int out_width = devPtrOutput->shape[3];
    // printf("out_channel:%d, out_height:%d, out_width:%d\n", out_channel, out_height, out_width);


    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/batch_size,
                                        /*channels=*/in_channel,
                                        /*image_height=*/in_height,
                                        /*image_width=*/in_width));

    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/out_N,
                                        /*channels=*/out_channel,
                                        /*image_height=*/out_height,
                                        /*image_width=*/out_width));

    checkCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/filter_out,
                                        /*in_channels=*/filter_in,
                                        /*kernel_height=*/kernel_size,
                                        /*kernel_width=*/kernel_size));

    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                        /*pad_height=*/pad_height,
                                        /*pad_width=*/pad_width,
                                        /*vertical_stride=*/stride_height,
                                        /*horizontal_stride=*/stride_width,
                                        /*dilation_height=*/dilation_height,
                                        /*dilation_width=*/dilation_width,
                                        /*mode=*/CUDNN_CONVOLUTION,
                                        /*computeType=*/CUDNN_DATA_FLOAT));
    
    // explanation for CUDNN_CROSS_CORRELATION and CUDNN_CONVOLUTION 
    // refer to https://zhuanlan.zhihu.com/p/33194385
    
    // choose convolution algorithm
    cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

  // set workspace
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn,
                                                    input_descriptor,
                                                    output_descriptor,
                                                    convolution_descriptor,
                                                    filter_descriptor,
                                                    algo,
                                                    &workspace_bytes));

    // allocate workspace memory
    void *workSpace = 0;
    if (workspace_bytes > 0) {
        cudaMalloc(&workSpace, workspace_bytes);
    }

    // compute
    float alpha = 1, beta = 0;
    checkCUDNN(cudnnConvolutionBackwardFilter(cudnn,
                                              &alpha,
                                              input_descriptor, devPtrInput->data,
                                              output_descriptor, devPtrOutput->data,
                                              convolution_descriptor,
                                              algo,
                                              workSpace, workspace_bytes,
                                              &beta,
                                              filter_descriptor, devPtrFilter->data));
    
    // release the memory
    cudaFree(workSpace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);
}


extern "C" void cudnnConv2dGetDataGradient(
    Quark* devPtrInput, 
    Quark* devPtrFilter, 
    Quark* devPtrOutput,
    int pad_height, 
    int pad_width, 
    int stride_height, 
    int stride_width,
    int dilation_height, 
    int dilation_width){
    
    // define and create context handle
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // define descriptor handle
    cudnnTensorDescriptor_t input_descriptor;
    cudnnFilterDescriptor_t filter_descriptor;
    cudnnTensorDescriptor_t output_descriptor;
    cudnnConvolutionDescriptor_t convolution_descriptor;

    //create descriptor handle
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));

    //config descriptor
    int batch_size = devPtrInput->shape[0];
    int in_channel = devPtrInput->shape[1];
    int in_height = devPtrInput->shape[2];
    int in_width = devPtrInput->shape[3];
    // printf("batch:%d, in_c:%d, in_h:%d, in_w:%d\n", batch_size, in_channel, in_height, in_width);

    int filter_out = devPtrFilter->shape[0];
    int filter_in = devPtrFilter->shape[1];
    int kernel_size = devPtrFilter->shape[2];
    // printf("filter_out:%d, filter_in:%d, kernel_size:%d\n", filter_out, filter_in, kernel_size);

    int out_N = devPtrOutput->shape[0];
    int out_channel = devPtrOutput->shape[1];
    int out_height = devPtrOutput->shape[2];
    int out_width = devPtrOutput->shape[3];
    // printf("out_channel:%d, out_height:%d, out_width:%d\n", out_channel, out_height, out_width);


    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/batch_size,
                                        /*channels=*/in_channel,
                                        /*image_height=*/in_height,
                                        /*image_width=*/in_width));

    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/out_N,
                                        /*channels=*/out_channel,
                                        /*image_height=*/out_height,
                                        /*image_width=*/out_width));

    checkCUDNN(cudnnSetFilter4dDescriptor(filter_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/filter_out,
                                        /*in_channels=*/filter_in,
                                        /*kernel_height=*/kernel_size,
                                        /*kernel_width=*/kernel_size));

    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                        /*pad_height=*/pad_height,
                                        /*pad_width=*/pad_width,
                                        /*vertical_stride=*/stride_height,
                                        /*horizontal_stride=*/stride_width,
                                        /*dilation_height=*/dilation_height,
                                        /*dilation_width=*/dilation_width,
                                        /*mode=*/CUDNN_CONVOLUTION,
                                        /*computeType=*/CUDNN_DATA_FLOAT));
    
    // explanation for CUDNN_CROSS_CORRELATION and CUDNN_CONVOLUTION 
    // refer to https://zhuanlan.zhihu.com/p/33194385
    
    // choose convolution algorithm
    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

    // set workspace
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn,
                                                    filter_descriptor,
                                                    output_descriptor,
                                                    convolution_descriptor,
                                                    input_descriptor,
                                                    algo,
                                                    &workspace_bytes));

    // allocate workspace memory
    void *workSpace = 0;
    if (workspace_bytes > 0) {
        cudaMalloc(&workSpace, workspace_bytes);
    }

    // compute
    float alpha = 1, beta = 0;
    checkCUDNN(cudnnConvolutionBackwardData(cudnn,
                                              &alpha,
                                              filter_descriptor, devPtrFilter->data,                                             
                                              output_descriptor, devPtrOutput->data,
                                              convolution_descriptor,
                                              algo,
                                              workSpace, workspace_bytes,
                                              &beta,
                                              input_descriptor, devPtrInput->data));
    
    // release the memory
    cudaFree(workSpace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);
}