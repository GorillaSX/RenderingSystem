#ifndef __SIMPLEINTEGRATOR_H__
#define __SIMPLEINTEGRATOR_H__

#include "Core/Common.h"

namespace Gorilla
{
    class Color;
    class Scene;
    class Intersection;
    class Ray;
    class Random;

    class SimpleIntegrator 
    {
    public:
        CUDA_CALLABLE Color getColorWithoutReflection(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const;
        CUDA_CALLABLE Color calculateLight(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const;

        uint32_t minPathLength = 2;
        uint32_t maxPathLength = 5;
    };
}

#endif //SIMPLEINTEGRATOR_H__