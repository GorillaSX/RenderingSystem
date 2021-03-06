#include "commonHeaders.h"
#include <cuda_runtime.h>
#include "Utils/CudaUtils.h"

#include "Core/Common.h"
#include "Utils/CudaAlloc.h"
#include "Core/Scene.h"
#include "Core/Film.h"


using namespace Gorilla;

template <typename T>
CudaAlloc<T>::CudaAlloc(bool pinned_):pinned(pinned_)
{
}

template <typename T>
CudaAlloc<T>::~CudaAlloc()
{
    release();
}

template <typename T>
void CudaAlloc<T>::resize(size_t count)
{
    assert(count > 0);
    release();
    maxCount = count;

    if(pinned)
    {
        CudaUtils::checkError(cudaMallocHost(&hostPtr, sizeof(T) * count), "Could not allocate pinned host memory");
        
        if(hostPtr == nullptr)
            throw std::runtime_error("Could not allocate pinned host memory");
    }
    else
    {
        hostPtr = static_cast<T*>(malloc(sizeof(T) * count));

        if(hostPtr == nullptr)
            throw std::runtime_error("Could not allocate host memory");
    }

    CudaUtils::checkError(cudaMalloc(&devicePtr, sizeof(T) * count), "Could not allocate device memory");

    if(devicePtr == nullptr)
        throw std::runtime_error("Could not allocate device memory");
}


template <typename T>
void CudaAlloc<T>::write(T* source, size_t count)
{
    assert(count <= maxCount);
    memcpy(hostPtr, source, sizeof(T) * count);

    CudaUtils::checkError(cudaMemcpy(devicePtr, hostPtr, sizeof(T)*count, cudaMemcpyHostToDevice), "Could not write data to device");

}

template <typename T>
void CudaAlloc<T>::read(size_t count)
{
    assert(count < maxCount);
    CudaUtils::checkError(cudaMemcpy(hostPtr, devicePtr, sizeof(T) * count, cudaMemcpyDeviceToHost), "Could not read data from device");

}

template <typename T>
CUDA_CALLABLE T* CudaAlloc<T>::getPtr() const
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
    return devicePtr;
#else 
    return hostPtr;
#endif
}

template <typename T>
T* CudaAlloc<T>::getHostPtr() const
{
    return hostPtr;
}

template <typename T>
T* CudaAlloc<T>::getDevicePtr() const
{
    return devicePtr;
}

template <typename T>
void CudaAlloc<T>::release()
{
    maxCount = 0;

    if(hostPtr != nullptr)
    {
        if(pinned)
            CudaUtils::checkError(cudaFreeHost(hostPtr), "Could not free pinned host memory");
        else
            free(hostPtr);
        hostPtr = nullptr;
    }

    if(devicePtr != nullptr)
    {
        CudaUtils::checkError(cudaFree(devicePtr), "Could not free device memory");
        devicePtr = nullptr;
    }
}

namespace Gorilla
{
	template class CudaAlloc<uint32_t>;
	template class CudaAlloc<Scene>;
	template class CudaAlloc<Film>;
	template class CudaAlloc<Image>;
	template class CudaAlloc<Texture>;
	template class CudaAlloc<Material>;
	template class CudaAlloc<Triangle>;
	template class CudaAlloc<BVHNode>;
	template class CudaAlloc<RandomGeneratorState>;
}