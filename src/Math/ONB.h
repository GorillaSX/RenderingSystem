#ifndef __ONB_H__
#define __ONB_H__


#include "Core/Common.h"
#include "Math/Vector3.h"

/* orthoNormal Basis

left-hand coordinate system

-z is towards the monitor

u = right
v = up
w = forward(normal)
*/

namespace Gorilla
{
    class Matrix4x4;

    class ONB 
    {
    public:
        CUDA_CALLABLE ONB();
        CUDA_CALLABLE ONB(const Vector3& u, const Vector3& v, const Vector3& w);

        CUDA_CALLABLE ONB transformed(const Matrix4x4& tranformation) const;

        CUDA_CALLABLE static ONB fromNormal(const Vector3& normal, const Vector3& up = Vector3::almostUp());

        CUDA_CALLABLE static ONB up();
        
        Vector3 u;
        Vector3 v;
        Vector3 w;
    };
}

#endif //__ONB_H__