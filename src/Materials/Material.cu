#include "commonHeaders.h"

#include "Core/Scene.h"
#include "Materials/Material.h"
#include "Math/Vector3.h"
#include "Textures/Texture.h"

using namespace Gorilla;

CUDA_CALLABLE bool Material::isEmissive() const
{
    return emittanceTextureIndex != -1 || !emittance.isZero();
}

CUDA_CALLABLE Color Material::getEmittance(const Scene& scene, const Vector2& texcoord, const Vector3& position) const
{
    if(emittanceTextureIndex != -1)
        return scene.getTexture(emittanceTextureIndex).getColor(scene, texcoord, position);
    else
        return emittance;
}

CUDA_CALLABLE Color Material::getAmbient(const Scene& scene, const Vector2& texcoord, const Vector3& position) const
{
    if(ambientTextureIndex != -1)
        return scene.getTexture(ambientTextureIndex).getColor(scene, texcoord, position);
    else
        return ambient;
}
CUDA_CALLABLE Color Material::getDiffuse(const Scene& scene, const Vector2& texcoord, const Vector3& position) const
{
    if(diffuseTextureIndex != -1)
        return scene.getTexture(diffuseTextureIndex).getColor(scene, texcoord, position);
    else
        return diffuse;
}
CUDA_CALLABLE Color Material::getSpecular(const Scene& scene, const Vector2& texcoord, const Vector3& position) const
{
    if(specularTextureIndex != -1)
        return scene.getTexture(specularTextureIndex).getColor(scene, texcoord, position);
    else
        return specular;
}
CUDA_CALLABLE float Material::getShininess(const Scene& scene, const Vector2& texcoord, const Vector3& position) const
{
    if(shininessTextureIndex != -1)
        return scene.getTexture(shininessTextureIndex).getColor(scene, texcoord, position).r;
    else
        return shininess;
}

CUDA_CALLABLE float Material::getIor()
{
    return ior;
}

CUDA_CALLABLE float Material::getIllum()
{
    return illum;
}