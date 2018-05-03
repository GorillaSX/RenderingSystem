#ifndef __MITCHELLFILTER_H__
#define __MITCHELLFILTER_H__

#include "Core/Common.h"
#include "Math/Vector2.h"

namespace Gorilla
{
    class Vector2;

    class MitchellFilter 
    {
    public:
        CUDA_CALLABLE float getWeight(float s) const;
        CUDA_CALLABLE float getWeight(const Vector2& point) const;

        CUDA_CALLABLE Vector2 getRadius() const;

        float B = (1.0f / 3.0f);
        float C = (1.0f / 3.0f);
    };
}

#endif //__MITCHELLFILTER_H__