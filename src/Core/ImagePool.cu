#include "commonHeaders.h"
#include "Core/ImagePool.h"
#include "Core/Image.h"
#include <iostream>

using namespace Gorilla;

ImagePool::ImagePool(): imagesAlloc(true)
{
    images.reserve(1000);
}

uint32_t ImagePool::load(const std::string& fileName, bool applyGamma)
{
    if(!imagesMap.count(fileName))
    {
        images.emplace_back(fileName);
        imagesMap[fileName] = uint32_t(images.size() - 1);

        if(applyGamma)
            images.back().applyFastGamma(2.2f);
        images.back().upload();
    }

    return imagesMap[fileName];
}

void ImagePool::commit()
{
    if(images.size() > 0)
    {
        imagesAlloc.resize(images.size());
        imagesAlloc.write(images.data(),images.size());
    }
}

CUDA_CALLABLE Image* ImagePool::getImages() const
{
    return imagesAlloc.getPtr();
}

CUDA_CALLABLE Image& ImagePool::getImage(uint32_t index) const
{
    return imagesAlloc.getPtr()[index];
}
