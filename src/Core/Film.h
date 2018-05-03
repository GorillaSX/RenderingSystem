#ifndef __FILM_H__
#define __FILM_H__

#include <atomic>
#include <cstdint>

#include <GL/glcorearb.h>

#include <cuda_runtime.h>
#include "Core/Common.h"
#include "Core/Image.h"

namespace Gorilla
{
    class Color;
    class Tonemapper;

    class Film 
    {
    public:
        explicit Film(bool windowed);

        void initialize();
        void shutdown();
        void resize(uint32_t width, uint32_t height);
        void clear();
        bool hasBeenCleared() const;
        void resetCleared();
        void load(uint32_t width, uint32_t height, const std::string& fileName);
        void loadMultiple(uint32_t width, uint32_t height, const std::string& dirName);
        void save(const std::string& fileName) const;

        CUDA_CALLABLE void addSample(uint32_t x, uint32_t y, const Color& color, float filterWeight);
        CUDA_CALLABLE void addSample(uint32_t index, const Color& color, float filterWeight);

        void normalize();
        void tonemap();
        void updateTexture();

        Color getCumulativeColor(uint32_t x, uint32_t y) const;
        Color getNormalizedColor(uint32_t x, uint32_t y) const;
        Color getTonemappedColor(uint32_t x, uint32_t y) const;

        CUDA_CALLABLE Image& getCumulativeImage();
        CUDA_CALLABLE Image& getNormalizedImage();
        CUDA_CALLABLE Image& getTonemappedImage();

        CUDA_CALLABLE uint32_t getWidth() const;
        CUDA_CALLABLE uint32_t getHeight() const;
        CUDA_CALLABLE uint32_t getLength() const;

        GLuint getTextureId() const;
        std::atomic<uint32_t> pixelSamples;

    private:

        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t length = 0;

        bool windowed = false;
        bool cleared = false;

        Image cumulativeImage;
        Image normalizedImage;
        Image tonemappedImage;

        GLuint textureId = 0;

        cudaGraphicsResource* textureResource = nullptr;
    };
}

#endif //__FILM_H__