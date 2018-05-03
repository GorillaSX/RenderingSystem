#ifndef __WINDOWRUNNER_H__
#define __WINDOWRUNNER_H__

#include <map>
#include <memory>
#include "Runners/WindowRunnerRenderState.h"
#include "Utils/FpsCounter.h"
#include <mutex>
#include <thread>

struct GLFWwindow;

namespace Gorilla
{
    struct MouseInfo
    {
        int32_t windowX = 0;
        int32_t windowY = 0;
        int32_t filmX = 0;
        int32_t filmY = 0;
        int32_t deltaX = 0;
        int32_t deltaY = 0;
        float scrollY = 0.0f;
        bool hasScrolled = false;
    };

    class WindowRunner
	{
	public:

		~WindowRunner();

		int run();
		void stop();

		GLFWwindow* getGlfwWindow() const;
		uint32_t getWindowWidth() const;
		uint32_t getWindowHeight() const;
		const MouseInfo& getMouseInfo() const;
		float getElapsedTime() const;
		const FpsCounter& getFpsCounter() const;

		bool keyIsDown(int32_t key);
		bool mouseIsDown(int32_t button);
		bool keyWasPressed(int32_t key);
		bool mouseWasPressed(int32_t button);
		float getMouseWheelScroll();

	private:

		void initialize();
		void shutdown();

		void checkWindowSize();
		void printWindowSize();
		void windowResized(uint32_t width, uint32_t height);
		
		void mainloop();
		void update(float timeStep);
		void render(float timeStep, float interpolation);

		void takeScreenshot() const;

		bool shouldRun = true;
		bool glfwInitialized = false;

		double startTime = 0.0;

		GLFWwindow* glfwWindow = nullptr;
		uint32_t windowWidth = 0;
		uint32_t windowHeight = 0;

		MouseInfo mouseInfo;
		int32_t previousMouseX = 0;
		int32_t previousMouseY = 0;

		std::map<int32_t, bool> keyStates;
		std::map<int32_t, bool> mouseStates;

		WindowRunnerRenderState* renderState = nullptr;
		FpsCounter fpsCounter;

		void threadFunction(void);

		std::mutex m_mutex;
		std::pair<int,int> m_delta;
		std::thread m_thread;
	};
};

#endif //__WINDOWRUNNER_H__