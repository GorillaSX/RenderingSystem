#ifndef __FPSCOUNTER_H__
#define __FPSCOUNTER_H__

#include "Math/MovingAverage.h"

namespace Gorilla
{
	class FpsCounter
	{
	public:

		FpsCounter();

		void tick();
		void update();
		float getFrameTime() const;
		float getFps() const;
		std::string getFpsString() const;

	private:

		double lastTime = 0.0;
		double frameTime = 0.0;
		MovingAverage averageFrameTime;
	};
}

#endif //__FPSCOUNTER_H__