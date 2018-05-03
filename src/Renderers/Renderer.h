#ifndef __RENDERER_H__
#define __RENDERER_H__


#include "Utils/CudaAlloc.h"
#include "Utils/Timer.h"
#include "Math/Random.h"


namespace Gorilla
{
    class Scene;
    class Film;
    class Settings;
    struct RandomGeneratorState;

    struct RenderJob
    {
        Scene* scene = nullptr;
        Film* film = nullptr;
    };

    class Renderer 
    {
    public:
        Renderer();

        void initialize(const Settings& settings);
        void resize(uint32_t width, uint32_t height);
        void render(RenderJob& job);

    private:
        CudaAlloc<Scene> sceneAlloc;
        CudaAlloc<Film> filmAlloc;
        CudaAlloc<RandomGeneratorState> randomStatesAlloc;
    };
}

#endif //__RENDERER_H__