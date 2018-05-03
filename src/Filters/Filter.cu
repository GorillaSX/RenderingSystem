#include "commonHeaders.h"
#include "Filters/Filter.h"
#include "Math/Vector2.h"

using namespace Gorilla;

Filter::Filter(FilterType type_)
{
    type = type_;
}

CUDA_CALLABLE float Filter::getWeight(float s) const
{
    switch(type)
    {
        case FilterType::MITCHELL: return mitchellFilter.getWeight(s);
        default: return 0.0f;
    }
}

CUDA_CALLABLE float Filter::getWeight(const Vector2& point) const 
{
    switch(type)
    {
        case FilterType::MITCHELL: return mitchellFilter.getWeight(point);
        default: return 0.0f;
    }
}

CUDA_CALLABLE Vector2 Filter::getRadius() const 
{
    switch(type)
    {
        case FilterType::MITCHELL: return mitchellFilter.getRadius();
        default: return Vector2(0.0, 0.0f);
    }
}

