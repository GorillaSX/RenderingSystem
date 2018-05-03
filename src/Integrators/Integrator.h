#ifndef __INTEGRATOR_H__
#define __INTEGRATOR_H__

#include "Core/Common.h"
#include "Core/Intersection.h"
#include "Integrators/SimpleIntegrator.h"
#include "Materials/Material.h"

namespace Gorilla
{
    class Color;
    class Scene;
    class Ray;
    class Random;

    enum class IntegratorType  { SIMPLEINTEGRATOR };

    struct DirectLightSample
    {
        Color emittance;
        Vector3 direction;
        float distance2 = 0.0f;
        float originCosine = 0.0f;
        float lightCosine = 0.0f;
        float lightPdf = 0.0f;
        bool visible = false;
    };

    class Integrator 
    {
    public:
        CUDA_CALLABLE Color calculateLight(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const;

        CUDA_CALLABLE static Intersection getRandomEmissiveIntersection(const Scene& scene, Random& random);

        CUDA_CALLABLE static bool isIntersectionVisible(const Scene& scene, const Intersection& origin, const Intersection& emissiveIntersection);
        CUDA_CALLABLE static DirectLightSample calculateDirectLightSample(const Scene& scene, const Intersection& origin, const Intersection& emissiveIntersection);

        IntegratorType type = IntegratorType::SIMPLEINTEGRATOR;

        SimpleIntegrator simpleIntegrator;
    };
}

#endif //__INTEGRATOR_H__