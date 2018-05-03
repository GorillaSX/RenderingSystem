#ifndef __RAY_H__
#define __RAY_H__

#include <cfloat>
#include "Core/Common.h"
#include "Math/Vector3.h"

namespace Gorilla
{
    class Ray 
    {
    public:
        CUDA_CALLABLE void precalculate();

        Vector3 origin;
        Vector3 direction;
        Vector3 inverseDirection;

        float minDistance = 0.0f;
        float maxDistance = FLT_MAX;

        bool isVisibilityRay = false;
        bool isPrimaryRay =false;
        bool directionIsNegative[3];
    };
}

#endif //__RAY_H__