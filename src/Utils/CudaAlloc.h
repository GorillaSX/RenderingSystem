#ifndef __CUDAALLOC_H__
#define __CUDAALLOC_H__

#include "Core/Common.h"

namespace Gorilla
{
    template <typename T>
    class CudaAlloc 
    {
    public:
        explicit CudaAlloc(bool pinned);
        ~CudaAlloc();

        void resize(size_t count);
        void write(T* source, size_t count);
        void read(size_t count);

        CUDA_CALLABLE T* getPtr() const;

        T* getHostPtr() const;
        T* getDevicePtr() const;

    private:

        void release();
        bool pinned = false;

        T* hostPtr = nullptr;
        T* devicePtr = nullptr;

        size_t maxCount = 0;
    };
}


#endif //__CUDAALLOC_H__