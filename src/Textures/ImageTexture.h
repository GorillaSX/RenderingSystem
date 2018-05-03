#ifndef __IMAGETEXTURE_H__
#define __IMAGETEXTURE_H__

#include "Core/Common.h"
#include "Core/Image.h"

namespace Gorilla
{
    class Scene;
    class Vector2;
    class Vector3;

    class ImageTexture 
    {
    public:
        void initialize(Scene& scene);

        CUDA_CALLABLE Color getColor(const Scene& scene, const Vector2& texcoord, const Vector3& position) const;

        std::string imageFileName;
        bool applyGamma = false;
    
    private:
        uint32_t imageIndex = 0;
    };
}

#endif //__IMAGETEXTURE_H__