#ifndef __FILTER_H__
#define __FILTER_H__

#include <string>
#include "Core/Common.h"
#include "Filters/MitchellFilter.h"

namespace Gorilla
{
    class Vector2;
    enum class FilterType { MITCHELL = 1};

    class Filter 
    {
    public:
        explicit Filter(FilterType type = FilterType::MITCHELL);
        CUDA_CALLABLE float getWeight(float s) const;
        CUDA_CALLABLE float getWeight(const Vector2& point) const;

        CUDA_CALLABLE Vector2 getRadius() const;

        FilterType type = FilterType::MITCHELL;

        MitchellFilter mitchellFilter;
    };
}

#endif //__FILTER_H__