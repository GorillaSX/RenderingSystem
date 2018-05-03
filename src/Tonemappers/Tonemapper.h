#ifndef __TONEMAPPER_H__
#define __TONEMAPPER_H__

#include "Tonemappers/SimpleTonemapper.h"

namespace Gorilla
{
    class Image;

    enum class TonemapperType {SIMPLE};

    class Tonemapper 
    {
    public:
        void apply(const Image& inputImage, Image& outputImage);

        TonemapperType type = TonemapperType::SIMPLE;

        SimpleTonemapper simpleTonemapper;
    };
}

#endif //__TONEMAPPER_H__