#include "commonHeaders.h"
#include "Core/Scene.h"
#include "Core/ImagePool.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Textures/ImageTexture.h"
#include <iostream>


using namespace Gorilla;

void ImageTexture::initialize(Scene& scene)
{
    imageIndex = scene.imagePool.load(imageFileName, applyGamma);
}

CUDA_CALLABLE Color ImageTexture::getColor(const Scene& scene, const Vector2& texcoord, const Vector3& position) const
{
    Image& image = scene.imagePool.getImage(imageIndex);
    return image.getPixelBilinear(texcoord.x, texcoord.y);
}