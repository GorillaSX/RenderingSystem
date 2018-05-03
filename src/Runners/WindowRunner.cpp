#include "commonHeaders.h"

#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

#include "App.h"
#include "Core/Image.h"
#include "Runners/WindowRunner.h"
#include "Utils/GLUtils.h"
#include <utility>
#include "Utils/Settings.h"
#include "EyeTracker/EyeTracker.h"

using namespace Gorilla;

namespace
{
	void glfwErrorCallback(int32_t error, const char* description)
	{
		printf("GLFW error (%d): %s", error, description);
	}

	MouseInfo* mouseInfoPtr = nullptr;

	void glfwMouseWheelScroll(GLFWwindow* window, double xoffset, double yoffset)
	{
		(void)window;
		(void)xoffset;

		mouseInfoPtr->scrollY = float(yoffset);
		mouseInfoPtr->hasScrolled = true;
	}
}

WindowRunner::~WindowRunner()
{
	if (glfwInitialized)
		glfwTerminate();
}

int WindowRunner::run()
{
	initialize();
	mainloop();
	shutdown();

	return 0;
}

void WindowRunner::stop()
{
	shouldRun = false;
}

GLFWwindow* WindowRunner::getGlfwWindow() const
{
	return glfwWindow;
}

uint32_t WindowRunner::getWindowWidth() const
{
	return windowWidth;
}

uint32_t WindowRunner::getWindowHeight() const
{
	return windowHeight;
}

const MouseInfo& WindowRunner::getMouseInfo() const
{
	return mouseInfo;
}

float WindowRunner::getElapsedTime() const
{
	return float(glfwGetTime() - startTime);
}

const FpsCounter& WindowRunner::getFpsCounter() const
{
	return fpsCounter;
}

bool WindowRunner::keyIsDown(int32_t key)
{
	return (glfwGetKey(glfwWindow, key) == GLFW_PRESS);
}

bool WindowRunner::mouseIsDown(int32_t button)
{
	return (glfwGetMouseButton(glfwWindow, button) == GLFW_PRESS);
}

bool WindowRunner::keyWasPressed(int32_t key)
{
	if (keyIsDown(key))
	{
		if (!keyStates[key])
		{
			keyStates[key] = true;
			return true;
		}
	}
	else
		keyStates[key] = false;

	return false;
}

bool WindowRunner::mouseWasPressed(int32_t button)
{
	if (mouseIsDown(button))
	{
		if (!mouseStates[button])
		{
			mouseStates[button] = true;
			return true;
		}
	}
	else
		mouseStates[button] = false;

	return false;
}

float WindowRunner::getMouseWheelScroll()
{
	if (mouseInfo.hasScrolled)
	{
		mouseInfo.hasScrolled = false;
		return mouseInfo.scrollY;
	}

	return 0.0f;
}

void WindowRunner::initialize()
{
	Settings& settings = App::getSettings();
	EyeTracker& eyeTracker = App::getEyeTracker();
	eyeTracker.Initialize();

	mouseInfoPtr = &mouseInfo;

	printf("Initializing GLFW library\n");

	glfwSetErrorCallback(::glfwErrorCallback);

	if (!glfwInit())
		throw std::runtime_error("Could not initialize GLFW library");

	glfwInitialized = true;
	startTime = glfwGetTime();

	printf("Creating window and OpenGL context (%dx%d, fullscreen: %d)", settings.window.width, settings.window.height, settings.window.fullscreen);

	glfwWindow = glfwCreateWindow(int32_t(settings.window.width), int32_t(settings.window.height), "Gorilla", settings.window.fullscreen ? glfwGetPrimaryMonitor() : nullptr, nullptr);

	if (!glfwWindow)
		throw std::runtime_error("Could not create the window");

	printWindowSize();
	glfwSetScrollCallback(glfwWindow, ::glfwMouseWheelScroll);

	const GLFWvidmode* videoMode = glfwGetVideoMode(glfwGetPrimaryMonitor());
	glfwSetWindowPos(glfwWindow, (videoMode->width / 2 - int32_t(settings.window.width) / 2), (videoMode->height / 2 - int32_t(settings.window.height) / 2));
	glfwMakeContextCurrent(glfwWindow);

	printf("Initializing GL3W library\n");

	int32_t result = gl3wInit();

	if (result == -1)
		throw std::runtime_error("Could not initialize GL3W library");

	printf("OpenGL Vendor: %s | Renderer: %s | Version: %s | GLSL: %s\n", glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));

	glfwSwapInterval(settings.window.vsync ? 1 : 0);

	if (settings.window.hideCursor)
		glfwSetInputMode(glfwWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	windowWidth = settings.window.width;
	windowHeight = settings.window.height;

	glViewport(0, 0, GLsizei(windowWidth), GLsizei(windowHeight));

    renderState = new WindowRunnerRenderState();
	renderState->initialize();
	m_thread = std::thread(std::bind(&WindowRunner::threadFunction, this));
	m_thread.detach();
}

void WindowRunner::shutdown()
{
	renderState->shutdown();
	renderState = nullptr;
}

void WindowRunner::checkWindowSize()
{
	int tempWindowWidth, tempWindowHeight;

	glfwGetFramebufferSize(glfwWindow, &tempWindowWidth, &tempWindowHeight);

	if (tempWindowWidth == 0 || tempWindowHeight == 0)
		return;

	if (uint32_t(tempWindowWidth) != windowWidth || uint32_t(tempWindowHeight) != windowHeight)
	{
		printWindowSize();
		windowResized(uint32_t(tempWindowWidth), uint32_t(tempWindowHeight));
	}
}

void WindowRunner::printWindowSize()
{
	int tempWindowWidth, tempWindowHeight, tempFramebufferWidth, tempFramebufferHeight;

	glfwGetWindowSize(glfwWindow, &tempWindowWidth, &tempWindowHeight);
	glfwGetFramebufferSize(glfwWindow, &tempFramebufferWidth, &tempFramebufferHeight);

	printf("GLFW window size: %dx%d | GLFW framebuffer size: %dx%d", tempWindowWidth, tempWindowHeight, tempFramebufferWidth, tempFramebufferHeight);
}

void WindowRunner::windowResized(uint32_t width, uint32_t height)
{
	windowWidth = width;
	windowHeight = height;

	glViewport(0, 0, GLsizei(windowWidth), GLsizei(windowHeight));

	renderState->windowResized(windowWidth, windowHeight);
}

// http://gafferongames.com/game-physics/fix-your-timestep/
// http://gamesfromwithin.com/casey-and-the-clearly-deterministic-contraptions
// https://randomascii.wordpress.com/2012/02/13/dont-store-that-in-a-float/
void WindowRunner::mainloop()
{
	printf("Entering the main loop\n");

	double timeStep = 1.0 / 60.0;
	double previousTime = glfwGetTime();
	double timeAccumulator = 0.0;

    update(0.0f);

	while (shouldRun)
	{
		double currentTime = glfwGetTime();
		double frameTime = currentTime - previousTime;
		previousTime = currentTime;

		// prevent too large frametimes (e.g. program was paused)
		if (frameTime > 0.25)
			frameTime = 0.25;

		timeAccumulator += frameTime;

		while (timeAccumulator >= timeStep)
		{
			update(float(timeStep));
			timeAccumulator -= timeStep;
		}

		double interpolation = timeAccumulator / timeStep;
		render(float(frameTime), float(interpolation));
	}
}

void WindowRunner::update(float timeStep)
{
	Settings& settings = App::getSettings();
	
	fpsCounter.update();
	glfwPollEvents();

	checkWindowSize();

	double newMouseX, newMouseY;
	glfwGetCursorPos(glfwWindow, &newMouseX, &newMouseY);

	mouseInfo.windowX = int32_t(newMouseX + 0.5);
	mouseInfo.windowY = int32_t(double(windowHeight) - newMouseY - 1.0 + 0.5);
	mouseInfo.filmX = int32_t((mouseInfo.windowX / double(windowWidth)) * (double(windowWidth) * settings.window.renderScale) + 0.5);
	mouseInfo.filmY = int32_t((mouseInfo.windowY / double(windowHeight)) * (double(windowHeight) * settings.window.renderScale) + 0.5);
	mouseInfo.deltaX = mouseInfo.windowX - previousMouseX;
	mouseInfo.deltaY = mouseInfo.windowY - previousMouseY;
	previousMouseX = mouseInfo.windowX;
	previousMouseY = mouseInfo.windowY;
	
//	
	std::lock_guard<std::mutex> lg(m_mutex);
	mouseInfo.deltaY = std::abs(m_delta.first) < 3 ? 0 : abs((float(m_delta.first % 30) / 300.0f) * 300.0);
	mouseInfo.deltaX = std::abs(m_delta.second) < 2 ? 0 : abs((float(m_delta.second % 15) / 200.0f) * 200.0);
	m_delta.first = 0;
	m_delta.second = 0;
//
	

	if (glfwWindowShouldClose(glfwWindow))
		shouldRun = false;

	if (keyWasPressed(GLFW_KEY_ESCAPE))
		shouldRun = false;

	renderState->update(timeStep);
}

void WindowRunner::render(float timeStep, float interpolation)
{
	fpsCounter.tick();

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	renderState->render(timeStep, interpolation);

	glfwSwapBuffers(glfwWindow);

	if (keyWasPressed(GLFW_KEY_F12))
		takeScreenshot();
}

void WindowRunner::takeScreenshot() const
{
	std::vector<float> data(windowWidth * windowHeight * 4);

	glReadPixels(0, 0, GLsizei(windowWidth), GLsizei(windowHeight), GL_RGBA, GL_FLOAT, &data[0]);
	GLUtils::checkError("Could not read pixels from renderbuffer");

	Image image(windowWidth, windowHeight, &data[0]);
	image.save("screenshot.png");
}

void WindowRunner::threadFunction(void)
{
	EyeTracker& eyeTracker = App::getEyeTracker();
	eyeTracker.GetDelta(m_mutex, m_delta);
}