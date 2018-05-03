#include "commonHeaders.h"
#include "App.h"
#include "Core/Camera.h"
#include "Core/Scene.h"
#include "Core/Film.h"
#include "Renderers/Renderer.h"
#include "Runners/ConsoleRunner.h"
#include "Utils/Settings.h"
#include "Utils/StringUtils.h"
#include "Utils/SysUtils.h"
#include <iostream>

using namespace Gorilla;
using namespace std::chrono;

int ConsoleRunner::run()
{
    Settings& settings = App::getSettings();
    Timer totalElapsedTimer;
    Renderer renderer;

    Scene scene;
    Film film(false);

    std::cout << settings.scene.fileName << std::endl; 
    scene.load(settings.scene.fileName);
    scene.initialize();
    film.initialize();
    renderer.initialize(settings);

    film.resize(settings.image.width, settings.image.height);
    renderer.resize(settings.image.width, settings.image.height);

    scene.camera.setImagePlaneSize(settings.image.width, settings.image.height);
    scene.camera.update(0.0f);

    renderJob.scene = &scene;
    renderJob.film = &film;

    std::exception_ptr renderException = nullptr;

    try
    {
        std::cout << "render start" << std::endl;
        renderer.render(renderJob);
        std::cout << "render end" << std::endl;
    }
    catch(...)
    {
        renderException = std::current_exception();
    }

    if(renderException != nullptr)
        std::rethrow_exception(renderException);

    film.normalize();
    film.tonemap();
    film.getCumulativeImage().download();
    film.getTonemappedImage().download();

    film.getTonemappedImage().save(settings.image.fileName);
    if(settings.image.autoView)
        SysUtils::openFileExternally(settings.image.fileName);
    
    return 0;
}

void ConsoleRunner::interrupt()
{
    return;
}