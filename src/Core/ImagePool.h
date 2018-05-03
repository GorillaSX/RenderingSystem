#ifndef __IMAGEPOOL_H__
#define __IMAGEPOOL_H__

#include <cstdint>
#include <map>
#include <vector>
#include "Core/Image.h"
#include "Utils/CudaAlloc.h"
#include "Core/Common.h"

namespace Gorilla
{

    class Image; 

    class ImagePool
    {
    public:
        ImagePool();

        uint32_t load(const std::string& fileName, bool applyGamma);
        void commit();

        CUDA_CALLABLE Image* getImages() const;
        CUDA_CALLABLE Image& getImage(uint32_t index) const;
    private:

        std::vector<Image> images;
        std::map<std::string, uint32_t> imagesMap;
        CudaAlloc<Image> imagesAlloc;
    };
}

#endif //__IMAGE_H__