#ifndef __SIMPLETONEMAPPER_H__
#define __SIMPLETONEMAPPER_H__

namespace Gorilla
{
    class Image;

    class SimpleTonemapper 
    {
    public:
        void apply(const Image& inputImage, Image& outputImage);

        bool applyGamma = true;
        bool shouldClamp = true;
        float gamma = 2.2f;
        float exposure = 0.0f;

    };
}

#endif //__SIMPLETONEMAPPER_H__