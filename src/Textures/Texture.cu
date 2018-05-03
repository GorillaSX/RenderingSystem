#include "commonHeaders.h"
#include "Math/Color.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Textures/Texture.h"
#include <iostream>

using namespace Gorilla;

Texture::Texture(TextureType type_) : type(type_)
{}

void Texture::initialize(Scene& scene)
{
    switch(type)
    {
        case TextureType::IMAGE: imageTexture.initialize(scene);
        break;
        default: break;
    }
}

CUDA_CALLABLE Color Texture::getColor(const Scene& scene, const Vector2& texcoord, const Vector3& position) const
{
    switch(type)
    {
        case TextureType::IMAGE: return imageTexture.getColor(scene, texcoord, position);
        default: return Color::black();
    }
}