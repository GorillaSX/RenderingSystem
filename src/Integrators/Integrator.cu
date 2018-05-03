#include "commonHeaders.h"
#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Integrators/Integrator.h"
#include "Materials/Material.h"
#include "Math/Random.h"

using namespace Gorilla;

CUDA_CALLABLE Color Integrator::calculateLight(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const
{
    switch(type)
    {
        case IntegratorType::SIMPLEINTEGRATOR: return simpleIntegrator.calculateLight(scene, intersection, ray, random);
        default: return Color::black();
    }
}


CUDA_CALLABLE Intersection Integrator::getRandomEmissiveIntersection( const Scene& scene, Random& random)
{
    const Triangle& triangle = scene.getEmissiveTriangles()[random.getUint32(0, scene.getEmissiveTrianglesCount() - 1)];
    return triangle.getRandomIntersection(scene, random);
}

CUDA_CALLABLE bool Integrator::isIntersectionVisible(const Scene& scene, const Intersection& origin, const Intersection& emissiveIntersection)
{
    Vector3 originToEmissive = emissiveIntersection.position - origin.position;
    float distance = originToEmissive.length();
    Vector3 direction = originToEmissive / distance;

    Ray visibilityRay;
    visibilityRay.origin = origin.position;
    visibilityRay.direction = direction;
    visibilityRay.minDistance = scene.general.rayMinDistance;
    visibilityRay.maxDistance = distance - scene.general.rayMinDistance;
    visibilityRay.isVisibilityRay = true;
    visibilityRay.precalculate();

    Intersection visibilityIntersection;
    return !scene.intersect(visibilityRay, visibilityIntersection);
}

CUDA_CALLABLE DirectLightSample Integrator::calculateDirectLightSample(const Scene& scene, const Intersection& origin, const Intersection& emissiveIntersection)
{
    Vector3 originToEmissive = emissiveIntersection.position - origin.position;
    float distance2 = originToEmissive.lengthSquared();
    float distance = std::sqrt(distance2);

    DirectLightSample result;
    result.direction = originToEmissive / distance;
    result.distance2 = distance2;
    result.originCosine = result.direction.dot(origin.normal);
    result.lightCosine = result.direction.dot(-emissiveIntersection.normal);

    if(result.originCosine <= 0.0f || result.lightCosine <= 0.0f)
    {
        result.visible = false;
        return result;
    }

    const Material& emissiveMaterial = scene.getMaterial(emissiveIntersection.materialIndex);

    result.emittance = emissiveMaterial.getEmittance(scene, emissiveIntersection.texcoord, emissiveIntersection.position);

    result.lightPdf = (1.0f / scene.getEmissiveTrianglesCount()) * (1.0f / emissiveIntersection.area) * (distance2 / result.lightCosine);
    result.visible = true;

    return result;
}
