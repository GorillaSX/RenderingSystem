#ifndef __WINDOWRUNNERRENDERSTATE_H__
#define __WINDOWRUNNERRENDERSTATE_H__

#include "Core/Scene.h"
#include "Core/Film.h"
#include "Renderers/Renderer.h"
#include "Utils/FilmQuad.h"

namespace Gorilla
{
	class WindowRunnerRenderState
	{
	public:

		WindowRunnerRenderState();

		void initialize();
		void shutdown();

		void update(float timeStep);
		void render(float timeStep, float interpolation);

		void windowResized(uint32_t width, uint32_t height);

	private:

		void resizeFilm();

		Scene scene;
		Film film;
		Renderer renderer;
		FilmQuad filmQuad;
	};
}


#endif //__WINDOWRUNNERRENDERSTATE_H__