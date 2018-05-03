#ifndef __CONSOLERUNNER_H__
#define __CONSOLERUNNER_H__

#include "Math/MovingAverage.h"
#include "Renderers/Renderer.h"

namespace Gorilla 
{
    class ConsoleRunner 
    {
    public:
        int run();
        void interrupt();

    private:
        //void printProgress(float percentage, const TimerData& elapsed, const TimerData& remaining, uint32_t pixelSamples);
        RenderJob renderJob;
    };
}

#endif //__CONSOLERUNNNER_H__