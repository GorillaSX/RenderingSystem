#ifndef __INTERSECTION_H__
#define __INTERSECTION_H__

#include <cfloat>
#include "Math/Vector3.h"
#include "Math/Vector2.h"
#include "Math/Color.h"
#include "Math/ONB.h"

namespace Gorilla
{
    class Intersection 
    {
    public:
        bool wasFound = false;
        bool isBehind = false;
        bool hasColor = false;

        float distance = FLT_MAX;
        float area = 0.0f;

        Vector3 position;
        Vector3 normal;
        Vector2 texcoord;
        Color color;

        ONB onb;

        uint32_t materialIndex = 0;
    };
}

#endif //__INTERSECTION_H__