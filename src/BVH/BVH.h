#ifndef __BVH_H__
#define __BVH_H__

#include <cstdint>
#include <vector>

#include "Core/AABB.h"
#include "Core/Common.h"
#include "Utils/CudaAlloc.h"

namespace Gorilla
{
    class Triangle;
    class Scene;
    class Ray;
    class Intersection;


    struct BVHBuildEntry 
    {
        uint32_t start;
        uint32_t end;
        int32_t parent;
    };

    struct BVHBuildTriangle
    {
        Triangle* triangle;
        AABB aabb;
        Vector3 center;
    };

    struct BVHSplitCache
    {
        AABB aabb;
        float cost;
    };

    struct BVHSplitOutput
    {
        uint32_t index;
        uint32_t axis;
        AABB fullAABB;
        AABB leftAABB;
        AABB rightAABB;
    };

    struct BVHNode
    {
        AABB aabb;
        int32_t rightOffset;
        uint32_t triangleOffset;
        uint32_t triangleCount;
        uint32_t splitAxis;
    };

    template<uint32_t N>
    struct BVHNodeSOA
    {
        float aabbMinX[N];
        float aabbMinY[N];
        float aabbMinZ[N];
        float aabbMaxX[N];
        float aabbMaxY[N];
        float aabbMaxZ[N];
        uint32_t rightOffSet[N-1];
        uint32_t splitAxis[N-1];
        uint32_t triangleOffset;
        uint32_t triangleCount;
        uint32_t isLeaf;
    };

    class BVH 
    {
    public:

        BVH();
        void build(std::vector<Triangle>& triangles);
        CUDA_CALLABLE bool intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const;
        
        static BVHSplitOutput calculateSplit(std::vector<BVHBuildTriangle>& buildTriangles, std::vector<BVHSplitCache>& cache, uint32_t start, uint32_t end);

        uint32_t maxLeafSize = 4;
    private:
        CudaAlloc<BVHNode> nodesAlloc;
    };
}



#endif //__BVH_H__