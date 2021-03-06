#include "commonHeaders.h"

#include "Math/MathUtils.h"

using namespace Gorilla;

CUDA_CALLABLE bool MathUtils::almostZero(float value, float threshold)
{
	return std::abs(value) < threshold;
}

// https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
CUDA_CALLABLE bool MathUtils::almostSame(float first, float second, float threshold)
{
	float difference = std::abs(first - second);

	if (difference < threshold)
		return true;

	float larger = MAX(std::abs(first), std::abs(second));

	if (difference < (larger * threshold))
		return true;

	return false;
}

bool MathUtils::almostSame(const std::complex<float>& first, const std::complex<float>& second, float threshold)
{
	return almostSame(first.real(), second.real(), threshold) && almostSame(first.imag(), second.imag(), threshold);
}

CUDA_CALLABLE float MathUtils::degToRad(float degrees)
{
	return (degrees * (float(M_PI) / 180.0f));
}

CUDA_CALLABLE float MathUtils::radToDeg(float radians)
{
	return (radians * (180.0f / float(M_PI)));
}

CUDA_CALLABLE float MathUtils::smoothstep(float t)
{
	return t * t * (3.0f - 2.0f * t);
}

CUDA_CALLABLE float MathUtils::smootherstep(float t)
{
	return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

CUDA_CALLABLE float MathUtils::fastPow(float a, float b)
{
	union
	{
		double d;
		int32_t x[2];
	} u = {double(a)};

	u.x[1] = static_cast<int32_t>(b * (u.x[1] - 1072632447) + 1072632447);
	u.x[0] = 0;

	return float(u.d);
}
