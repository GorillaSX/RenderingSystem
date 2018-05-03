#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include "Core/Common.h"
#include "Math/Color.h"
#include "Math/Vector2.h"

namespace Gorilla
{
    class Scene;
    class Material
    {
    public:
        CUDA_CALLABLE bool isEmissive() const;
        CUDA_CALLABLE Color getEmittance(const Scene& scene, const Vector2& texcoord, const Vector3& position) const;
        CUDA_CALLABLE Color getAmbient(const Scene& scene, const Vector2& texcoord, const Vector3& position) const;
        CUDA_CALLABLE Color getDiffuse(const Scene& scene, const Vector2& texcoord, const Vector3& position) const;
        CUDA_CALLABLE Color getSpecular(const Scene& scene, const Vector2& texcoord, const Vector3& position) const;
        CUDA_CALLABLE float getShininess(const Scene& scene, const Vector2& texcoord, const Vector3& position) const;
        CUDA_CALLABLE float getIor();
        CUDA_CALLABLE float getIllum();


        int32_t id = -1;

        bool normalInterpolation = true;
        bool autoInvertNormal = true;
        bool invertNormal = false;
        bool invisible = false;
        bool primaryRayInvisible = false;
        bool showEmittance = true;

        Color ambient = Color(0.0f, 0.0f, 0.0f);
        int32_t ambientTextureId = -1;
        int32_t ambientTextureIndex = -1;
        Color diffuse = Color(0.0f, 0.0f, 0.0f);
        int32_t diffuseTextureId = -1;
        int32_t diffuseTextureIndex = -1;
        Color specular = Color(0.0f, 0.0f, 0.0f);
        int32_t specularTextureId = -1;
        int32_t specularTextureIndex = -1;
        Color transmittance = Color(0.0f, 0.0f, 0.0f);
        float ior = 1.0f;
        float shininess = 1.0f;
        int32_t shininessTextureId = -1;
        int32_t shininessTextureIndex = -1;
        int illum = 1;

        int32_t bumpTextureId = -1;
        int32_t bumpTextureIndex = -1;

        



        

        Vector2 texcoordScale = Vector2(1.0f, 1.0f);

        Color emittance = Color(0.0f, 0.0f, 0.0f);
        int32_t emittanceTextureId = -1;
        int32_t emittanceTextureIndex = -1;

        int32_t normalTextureId = -1;
        int32_t normalTextureIndex = -1;

        int32_t maskTextureId = -1;
        int32_t maskTextureIndex = -1;
    };
}
#endif //__MATERIAL_H__
