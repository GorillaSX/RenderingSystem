#include "commonHeaders.h"
#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Materials/Material.h"
#include "Integrators/SimpleIntegrator.h"
#include "Math/Random.h"
#include "Integrators/Integrator.h"
#include "Math/MathUtils.h"

using namespace Gorilla;

CUDA_CALLABLE Color SimpleIntegrator::getColorWithoutReflection(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const 
{
    Color result(0.0f, 0.0f, 0.0f);
    const Material& material = scene.getMaterial(intersection.materialIndex);
    Intersection light = Integrator::getRandomEmissiveIntersection(scene, random);
    const Material& lightMaterial = scene.getMaterial(light.materialIndex);
    DirectLightSample lightSample = Integrator::calculateDirectLightSample(scene, intersection, light);
    result += material.getAmbient(scene, intersection.texcoord, intersection.position) * lightMaterial.getAmbient(scene, light.texcoord, light.position);
    if(!lightSample.visible)
        return result;

    if(!Integrator::isIntersectionVisible(scene, intersection, light))
        return result;
    

    result += material.getDiffuse(scene, intersection.texcoord, intersection.position) * lightMaterial.getDiffuse(scene, light.texcoord, light.position) * lightSample.originCosine;
    
    Vector3 reflect = ray.direction.reflect(intersection.normal);
    Vector3 originToEmissive = light.position - intersection.position;
    float distance = originToEmissive.length();
    Vector3 direction = originToEmissive / distance;
    float specular = MAX(0, direction.dot(reflect));
    result += material.getSpecular(scene, intersection.texcoord, intersection.position) * lightMaterial.getDiffuse(scene, light.texcoord, light.position) * MathUtils::fastPow(specular, material.getShininess(scene, intersection.texcoord, intersection.position));

    return result;
}

CUDA_CALLABLE Color SimpleIntegrator::calculateLight(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const
{
    if(scene.getMaterial(intersection.materialIndex).isEmissive())
        return scene.getMaterial(intersection.materialIndex).getDiffuse(scene, intersection.texcoord, intersection.position);
    Color result = getColorWithoutReflection(scene, intersection, ray, random);
    Ray input;
    Ray output = ray;
    Intersection pathIntersection = intersection;
    for(int i = 0;i < maxPathLength;++i)
    {
        Material material = scene.getMaterial(pathIntersection.materialIndex);
        if(material.getSpecular(scene, pathIntersection.texcoord, pathIntersection.position).isZero())
            return result;
        input = output;
        Vector3 reflectDirection = input.direction.reflect(pathIntersection.normal);
        output.origin = pathIntersection.position;
        output.direction = reflectDirection;
        output.maxDistance = FLT_MAX;
        output.isVisibilityRay = false;
        output.isPrimaryRay = false;
        output.precalculate();
        Intersection prevIntersection = pathIntersection;
        
        if(!scene.intersect(output, pathIntersection))
            return result;
        

        if(scene.getMaterial(pathIntersection.materialIndex).isEmissive())
            return result + material.getSpecular(scene, prevIntersection.texcoord, prevIntersection.position) * scene.getMaterial(pathIntersection.materialIndex).getDiffuse(scene, pathIntersection.texcoord, pathIntersection.position);
        
        scene.calculateNormalMapping(pathIntersection);
        result += material.getSpecular(scene, prevIntersection.texcoord, prevIntersection.position) * getColorWithoutReflection(scene, pathIntersection, output, random);
    }
    return result;
}