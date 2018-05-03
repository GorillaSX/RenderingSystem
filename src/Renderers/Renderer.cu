#include "commonHeaders.h"
#include <device_launch_parameters.h>
#include "Core/Ray.h"
#include "Core/Intersection.h"
#include "Utils/CudaUtils.h"
#include "Core/Film.h"
#include "Core/Scene.h"
#include "Renderers/Renderer.h"
#include "Utils/Settings.h"
#include "App.h"

using namespace Gorilla;

Renderer::Renderer():sceneAlloc(true), filmAlloc(true), randomStatesAlloc(false)
{
}

void Renderer::initialize(const Settings& settings)
{
    sceneAlloc.resize(1);
    filmAlloc.resize(1);
}

void Renderer::resize(uint32_t width, uint32_t height) 
{
    std::vector<RandomGeneratorState> randomStates(width * height);

    std::random_device rd;
    std::mt19937_64 generator(rd());

    for(RandomGeneratorState& randomState : randomStates)
    {
        randomState.state = generator();
        randomState.inc = generator();
    }

    randomStatesAlloc.resize(width * height);
    randomStatesAlloc.write(randomStates.data(), width * height);
}

__global__ void renderKernel(const Scene& scene, Film& film, RandomGeneratorState* randomStates, uint32_t pixelSamples)
{
    uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t index = y * film.getWidth() + x;

    if(x >= film.getWidth() || y > film.getHeight())
        return ;
    
    Random random(randomStates[index]);

    for(uint32_t i = 0;i < pixelSamples; ++i)
    {
        Vector2 pixel = Vector2(x, y);
        float filterWeight = 1.0f;

        if(scene.renderer.filtering)
        {
            Vector2 offset = (random.getVector2() - Vector2(0.5f, 0.5f)) * 2.0f * scene.renderer.filter.getRadius();
            filterWeight = scene.renderer.filter.getWeight(offset);
            pixel += offset;
        }

        CameraRay cameraRay = scene.camera.getRay(pixel, random);
        cameraRay.ray.isPrimaryRay = true;

        Intersection intersection;


        if(!scene.intersect(cameraRay.ray, intersection))
        {
            film.addSample(x, y, scene.general.backgroundColor, filterWeight);
            randomStates[index] = random.getState();
            return ;
        }

        if(intersection.hasColor)
        {
            film.addSample(x, y, intersection.color, filterWeight);
            randomStates[index] = random.getState();
            return ;
        }

        scene.calculateNormalMapping(intersection);

        if(scene.general.normalVisualization)
        {
            film.addSample(x, y, Color::fromNormal(intersection.normal), filterWeight);
            randomStates[index] = random.getState();
            return;
        }

        Color color = scene.integrator.calculateLight(scene, intersection, cameraRay.ray, random);

        if(!color.isNegative() && !color.isNan())
        film.addSample(x, y, color* cameraRay.brightness, filterWeight);
    }

    randomStates[index] = random.getState();
}

void Renderer::render(RenderJob& job)
{
    Scene& scene = *job.scene;
    Film& film = *job.film;
    Settings& settings = App::getSettings();
    

    sceneAlloc.write(&scene, 1);
    filmAlloc.write(&film, 1);

    dim3 dimBlock(16, 16);
    dim3 dimGrid;

    dimGrid.x = (film.getWidth() + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (film.getHeight() + dimBlock.y - 1) / dimBlock.y;

    renderKernel<<<dimGrid, dimBlock>>>(*sceneAlloc.getDevicePtr(), *filmAlloc.getDevicePtr(), randomStatesAlloc.getDevicePtr(), settings.renderer.pixelSamples);
    CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch render kernel");
    CudaUtils::checkError(cudaDeviceSynchronize(),"Could not execute render kernel");
}