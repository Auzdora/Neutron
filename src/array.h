/*
	Copyright © 2022 Melrose-Lbt
	All rights reserved.
	Filename: array.h
	Description: C++ back-end data structure.
	Created by Melrose-Lbt 2022-8-17
*/
#ifndef __ARRAY_H
#define __ARRAY_H


extern "C"{
    typedef enum{
        CPU = 0,
        GPU = 1
    }Device;

    typedef struct{
        float* data;
        Device device;
        int* shape;
        int dim;
    }Quark;
}


#endif
