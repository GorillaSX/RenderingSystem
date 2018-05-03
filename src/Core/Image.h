#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <cstdint>
#include <string>
#include <cuda_runtime.h>

#include "Core/Common.h"
#include "Math/Color.h"


/*
Origin (0, 0) is at the bottom left corner 
*/
namespace Gorilla
{
    class Filter;

    class Image 
    {
    public:
        Image();
        ~Image();

        Image(uint32_t width, uint32_t height);
        Image(uint32_t width, uint32_t height, float* rgbaData);
        Image(const std::string& fileName);
        Image(const Image& other);

        Image& operator=(const Image& other);

        void load(uint32_t width, uint32_t height, float * rgbaData);
        void load(const std::string& fileName);
        void save(const std::string& fileName) const;
        void resize(uint32_t length);
        void resize(uint32_t width, uint32_t height);
        void clear();
        void clear(const Color& color);

        void applyGamma(float gamma);
        void applyFastGamma(float gamma);
        void swapComponents();
        void fillWithTestPattern();

        CUDA_CALLABLE void setPixel(uint32_t x, uint32_t y, const Color& color);
        CUDA_CALLABLE void setPixel(uint32_t index, const Color& color);

        CUDA_CALLABLE Color getPixel(uint32_t x, uint32_t y) const;
        CUDA_CALLABLE Color getPixel(uint32_t index) const;
        CUDA_CALLABLE Color getPixelNearest(float u, float v) const;
        CUDA_CALLABLE Color getPixelBilinear(float u, float v) const;

        CUDA_CALLABLE uint32_t getWidth() const;
        CUDA_CALLABLE uint32_t getHeight() const;
        CUDA_CALLABLE uint32_t getLength() const;

        void upload();
        void download();
        Color* getData();
        const Color* getData() const;
        
        CUDA_CALLABLE cudaSurfaceObject_t getSurfaceObject() const;

        private:

            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t length = 0;

            Color* data = nullptr;
            cudaArray* cudaData = nullptr;
            cudaTextureObject_t textureObject = 0;
            cudaSurfaceObject_t surfaceObject = 0;
    };
}


#endif //__IMAGE_H__
