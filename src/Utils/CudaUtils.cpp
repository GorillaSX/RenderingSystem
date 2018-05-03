#include "commonHeaders.h"
#include "tinyformat/tinyformat.h"
#include "Utils/CudaUtils.h"

using namespace Gorilla;

void CudaUtils::checkError(cudaError_t code, const std::string& message)
{
    if(code != cudaSuccess)
        throw std::runtime_error(tfm::format("Cuda error: %s : %s", message, cudaGetErrorString(code)));
}

