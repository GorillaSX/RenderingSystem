#include "commonHeaders.h"
#include "Core/Ray.h"

using namespace Gorilla;

CUDA_CALLABLE void Ray::precalculate()
{
    inverseDirection = direction.inversed();
    directionIsNegative[0] = direction.x < 0.0f;
    directionIsNegative[1] = direction.y < 0.0f;
    directionIsNegative[2] = direction.z < 0.0f;
}