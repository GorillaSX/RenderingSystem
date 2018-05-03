#ifndef __CUDAUTILS_H__
#define __CUDAUTILS_H__

#include <cuda_runtime.h>

namespace Gorilla
{
    class CudaUtils 
    {
    public:
        static void checkError(cudaError_t code, const std::string& message);
    };
}

#endif //__CUDAUTILS_H__