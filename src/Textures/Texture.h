#ifndef __TEXTURE_H__
#define __TEXTURE_H__

#include "Core/Common.h"
#include "Math/Color.h"
#include "Textures/ImageTexture.h"


namespace Gorilla
{
    class Scene;
    class Vector2;
    class Vector3;

    enum class TextureType{IMAGE};

    class Texture
    {
    public:
        explicit Texture(TextureType type = TextureType::IMAGE);

        void initialize(Scene& scene);

        CUDA_CALLABLE Color getColor(const Scene& scene, const Vector2& texcoord, const Vector3& position) const;

        int32_t id = -1;

        TextureType type = TextureType::IMAGE;

        ImageTexture imageTexture;
    };
}


#endif //__TEXTURE_H__