#include "commonHeaders.h"

#include "Tonemappers/Tonemapper.h"
#include "Core/Image.h"

using namespace Gorilla;

void Tonemapper::apply(const Image& inputImage, Image& outputImage)
{
    assert(inputImage.getLength() == outputImage.getLength());

    switch(type)
    {
        case TonemapperType::SIMPLE:
        simpleTonemapper.apply(inputImage, outputImage); break;
        default: break;
    }
}

