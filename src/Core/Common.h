#ifndef __COMMON_H__
#define __COMMON_H__

#define ALIGN(x) __attribute__((aligned(x)))

#define MIN(a,b) (((a)<(b)?(a):(b)))
#define MAX(a,b) (((a)>(b)?(a):(b)))

#include <cuda_runtime.h>
#define CUDA_CALLABLE __host__ __device__

#endif //__COMMON_H__