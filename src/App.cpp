//This file is a good example to demonstrate my programming skills in C++ and oo design.
#include "commonHeaders.h"
#include <signal.h>
#include <cuda_runtime.h>
#include "App.h"
#include "Core/Common.h"
#include "Utils/Settings.h"
#include "Utils/CudaUtils.h"
#include "Runners/ConsoleRunner.h"
#include "Runners/WindowRunner.h"
#include "EyeTracker/EyeTracker.h"
#include <iostream>

using namespace Gorilla;

int main(int argc, char** argv)
{
    int result;
    
    result = App::run(argc, argv);
    cudaDeviceReset();
    return result;
}

void signalHandler(int signal)
{
    App::getConsoleRunner().interrupt();
}

int App::run(int argc, char** argv)
{
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    try
    {
        Settings& settings = getSettings();
        ConsoleRunner& consoleRunner = getConsoleRunner();
        WindowRunner& windowRunner = getWindowRunner();

        if(!settings.load(argc, argv))
            return 0;
        
        int deviceCount;
        CudaUtils::checkError(cudaGetDeviceCount(&deviceCount), "Could not get device count");
        CudaUtils::checkError(cudaSetDevice(settings.general.cudaDeviceNumber), "Could not set device");

        printf("CUDA select device: %d (device count: %d)\n", settings.general.cudaDeviceNumber, deviceCount);

        cudaDeviceProp deviceProp;
        CudaUtils::checkError(cudaGetDeviceProperties(&deviceProp, settings.general.cudaDeviceNumber), "Could not get device properties");

        int driverVersion;
        CudaUtils::checkError(cudaDriverGetVersion(&driverVersion), "Could not get driver version");

        int runtimeVersion;
        CudaUtils::checkError(cudaRuntimeGetVersion(&runtimeVersion), "Could not get runtime version");

        printf("CUDA device: %s | Compute capability: %d.%d | Driver version: %d | Runtime version: %d\n", deviceProp.name, deviceProp.major, deviceProp.minor, driverVersion, runtimeVersion);

        if(settings.general.windowed)
        {
            int result = windowRunner.run();
            return result;
        }
        else
            return consoleRunner.run();
    }
    catch(...)
    {
        throw std::current_exception();
        return -1;
    }
}

Settings& App::getSettings()
{
    static Settings settings;
    return settings;
}


ConsoleRunner& App::getConsoleRunner()
{
    static ConsoleRunner consoleRunner;
    return consoleRunner;
}

WindowRunner& App::getWindowRunner() 
{
    static WindowRunner windowRunner;
    return windowRunner;
}

EyeTracker& App::getEyeTracker()
{
    static EyeTracker eyeTracker;
    return eyeTracker;
}
