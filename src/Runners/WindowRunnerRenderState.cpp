#include "commonHeaders.h"

#include <GLFW/glfw3.h>

#include "App.h"
#include "Core/Camera.h"
#include "Core/Film.h"
#include "Core/Scene.h"
#include "Runners/WindowRunner.h"
#include "Runners/WindowRunnerRenderState.h"
#include "Utils/Settings.h"

using namespace Gorilla;

WindowRunnerRenderState::WindowRunnerRenderState() : film(true)
{
}

void WindowRunnerRenderState::initialize()
{
    Settings& settings = App::getSettings();
    scene.load(settings.scene.fileName);
	scene.initialize();
	film.initialize();
	renderer.initialize(settings);
	filmQuad.initialize();

	printf("%d\n",__LINE__);
    
	resizeFilm(); 

}

void WindowRunnerRenderState::shutdown()
{
	film.shutdown();
}

void WindowRunnerRenderState::update(float timeStep)
{
	Settings& settings = App::getSettings();
	WindowRunner& windowRunner = App::getWindowRunner();

	bool ctrlIsPressed = windowRunner.keyIsDown(GLFW_KEY_LEFT_CONTROL) || windowRunner.keyIsDown(GLFW_KEY_RIGHT_CONTROL);
	
	// INFO PANEL //


	// RENDERER / CAMERA / INTEGRATOR / FILTER / TONEMAPPER //

	if (!ctrlIsPressed)
	{
		if (windowRunner.keyWasPressed(GLFW_KEY_F3))
		{
			if (scene.camera.type == CameraType::PERSPECTIVE)
				scene.camera.type = CameraType::ORTHOGRAPHIC;
			else if (scene.camera.type == CameraType::ORTHOGRAPHIC)
				scene.camera.type = CameraType::FISHEYE;
			else if (scene.camera.type == CameraType::FISHEYE)
				scene.camera.type = CameraType::PERSPECTIVE;

			film.clear();
		}

		if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
		{
			if (scene.camera.type == CameraType::PERSPECTIVE)
				scene.camera.fov -= 50.0f * timeStep;
			else if (scene.camera.type == CameraType::ORTHOGRAPHIC)
				scene.camera.orthoSize -= 10.0f * timeStep;
			else if (scene.camera.type == CameraType::FISHEYE)
				scene.camera.fishEyeAngle -= 50.0f * timeStep;

			film.clear();
		}

		if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
		{
			if (scene.camera.type == CameraType::PERSPECTIVE)
				scene.camera.fov += 50.0f * timeStep;
			else if (scene.camera.type == CameraType::ORTHOGRAPHIC)
				scene.camera.orthoSize += 10.0f * timeStep;
			else if (scene.camera.type == CameraType::FISHEYE)
				scene.camera.fishEyeAngle += 50.0f * timeStep;

			film.clear();
		}
	}

	// RENDER SCALE //

	if (!ctrlIsPressed)
	{
		if (windowRunner.keyWasPressed(GLFW_KEY_F7))
		{
			float newScale = settings.window.renderScale * 0.5f;
			uint32_t newWidth = uint32_t(float(windowRunner.getWindowWidth()) * newScale + 0.5f);
			uint32_t newHeight = uint32_t(float(windowRunner.getWindowHeight()) * newScale + 0.5f);

			if (newWidth >= 2 && newHeight >= 2)
			{
				settings.window.renderScale = newScale;
				resizeFilm();
			}
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_F8))
		{
			if (settings.window.renderScale < 1.0f)
			{
				settings.window.renderScale *= 2.0f;

				if (settings.window.renderScale > 1.0f)
					settings.window.renderScale = 1.0f;

				resizeFilm();
			}
		}
	}

	// MISC //
	
	if (windowRunner.keyWasPressed(GLFW_KEY_R))
	{
		scene.camera.reset();
		film.clear();
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_F))
	{
		scene.renderer.filtering = !scene.renderer.filtering;
		film.clear();
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_P))
		scene.camera.enableMovement = !scene.camera.enableMovement;

	if (windowRunner.keyWasPressed(GLFW_KEY_M))
	{
		scene.general.normalMapping = !scene.general.normalMapping;
		film.clear();
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_N))
	{
		scene.general.normalInterpolation = !scene.general.normalInterpolation;
		film.clear();
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_B))
	{
		scene.general.normalVisualization = !scene.general.normalVisualization;
		scene.general.interpolationVisualization = false;
		film.clear();
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_V))
	{
		scene.general.interpolationVisualization = !scene.general.interpolationVisualization;
		scene.general.normalVisualization = false;
		film.clear();
	}


	// SCENE SAVING //

	if (ctrlIsPressed)
	{
		//if (windowRunner.keyWasPressed(GLFW_KEY_F1))
		//	scene.save("scene.xml");

		if (windowRunner.keyWasPressed(GLFW_KEY_F2))
			scene.camera.saveState("camera.txt");

		if (windowRunner.keyWasPressed(GLFW_KEY_F3))
		{
			film.normalize();
			film.tonemap();
			film.getTonemappedImage().download();
			film.getTonemappedImage().save("image.png");
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_F4))
		{
			film.getCumulativeImage().download();
			film.save("film.bin");
		}
	}

	// TEST SCENE LOADING //

    scene.camera.update(timeStep);
    
}

void WindowRunnerRenderState::render(float timeStep, float interpolation)
{
	(void)timeStep;
	(void)interpolation;


	if (scene.camera.isMoving())
		film.clear();

	film.resetCleared();
	RenderJob job;
	job.scene = &scene;
	job.film = &film;
    
	renderer.render(job);
	film.normalize();
	film.tonemap();
	film.updateTexture();
	
	filmQuad.render(film);
}

void WindowRunnerRenderState::windowResized(uint32_t width, uint32_t height)
{
	(void)width;
	(void)height;

	resizeFilm();
}

void WindowRunnerRenderState::resizeFilm()
{
	Settings& settings = App::getSettings();
	WindowRunner& windowRunner = App::getWindowRunner();

	uint32_t filmWidth = uint32_t(float(windowRunner.getWindowWidth()) * settings.window.renderScale + 0.5);
	uint32_t filmHeight = uint32_t(float(windowRunner.getWindowHeight()) * settings.window.renderScale + 0.5);

    filmWidth = MAX(uint32_t(1), filmWidth);
    filmHeight = MAX(uint32_t(1), filmHeight);

	film.resize(filmWidth, filmHeight);
	renderer.resize(filmWidth, filmHeight);
	scene.camera.setImagePlaneSize(filmWidth, filmHeight);
}
