#include "commonHeaders.h"
#include "Math/ONB.h"
#include "Math/Matrix4x4.h"

using namespace Gorilla;

CUDA_CALLABLE ONB::ONB()
{
}

CUDA_CALLABLE ONB::ONB(const Vector3& u_, const Vector3& v_, const Vector3& w_) : u(u_), v(v_), w(w_)
{
}

CUDA_CALLABLE ONB ONB::transformed(const Matrix4x4& transformation) const 
{
    ONB result;

    result.u = transformation.transformDirection(u).normalized();
    result.v = transformation.transformDirection(v).normalized();
    result.w = transformation.transformDirection(w).normalized();

    return result;
}

CUDA_CALLABLE ONB ONB::fromNormal(const Vector3& normal, const Vector3& up)
{
    Vector3 u_ = normal.cross(up).normalized();
    Vector3 v_ = u_.cross(normal).normalized();
    Vector3 w_ = normal;
    return ONB(u_, v_, w_);
}

CUDA_CALLABLE ONB ONB::up()
{
    return ONB(Vector3(-1.0f, 0.0f, 0.0f), Vector3(0.0f, 0.0f, -1.0f), Vector3(0.0f, 1.0f, 0.0f));
}