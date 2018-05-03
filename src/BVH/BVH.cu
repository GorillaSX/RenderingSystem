#include "commonHeaders.h"

#include "BVH/BVH.h"
#include <cfloat>
#include "App.h"
#include "Core/Common.h"
#include "Utils/Timer.h"
#include "Core/Scene.h"
#include "Core/Triangle.h"
#include "Core/Ray.h"
#include "Core/Intersection.h"

#include "tinyformat/tinyformat.h"
#include <iostream>

//#include <parallel/algorithm>
//#define PARALLEL_SORT __gnu_parallel::sort 
#define PARALLEL_SORT std::sort


using namespace Gorilla;

BVH::BVH(): nodesAlloc(false)
{
}

void BVH::build(std::vector<Triangle>& triangles)
{
    Timer timer;
    uint32_t triangleCount = uint32_t(triangles.size());

    if(triangleCount == 0)
    {
        std::cout << "this" << std::endl;
        throw std::runtime_error(tfm::format("Could not build BVH from empty triangle list"));
    }

    std::vector<BVHBuildTriangle> buildTriangles(triangleCount);
    std::vector<BVHSplitCache> cache(triangleCount);
    BVHSplitOutput splitOutput;

    for(uint32_t i = 0;i < triangleCount; ++i)
    {
        AABB aabb = triangles[i].getAABB();

        buildTriangles[i].triangle = &triangles[i];
        buildTriangles[i].aabb = aabb;
        buildTriangles[i].center = aabb.getCenter();
    }

    std::vector<BVHNode> nodes;
    nodes.reserve(triangleCount);

    BVHBuildEntry stack[128];
    uint32_t stackIndex = 0;
    uint32_t nodeCount = 0;
    uint32_t leafCount = 0;

    stack[stackIndex].start = 0;
    stack[stackIndex].end = triangleCount;
    stack[stackIndex].parent = -1;
    stackIndex++;

    while(stackIndex > 0)
    {
        nodeCount++;
        BVHBuildEntry buildEntry = stack[--stackIndex];

        BVHNode node;
        node.rightOffset = -3;
        node.triangleOffset = uint32_t(buildEntry.start);
        node.triangleCount = uint32_t(buildEntry.end - buildEntry.start);
        node.splitAxis = 0;

        if(node.triangleCount <= maxLeafSize)
            node.rightOffset = 0;
        
        if(buildEntry.parent != -1)
        {
            uint32_t parent = uint32_t(buildEntry.parent);

            if(++nodes[parent].rightOffset == -1)
                nodes[parent].rightOffset = int32_t(nodeCount - 1 - parent);
        }

        if(node.rightOffset != 0)
        {
            splitOutput = BVH::calculateSplit(buildTriangles, cache, buildEntry.start, buildEntry.end);
            node.splitAxis = uint32_t(splitOutput.axis);
            node.aabb = splitOutput.fullAABB;
        }

        nodes.push_back(node);

        if(node.rightOffset == 0)
        {
            leafCount++;
            continue;
        }

        stack[stackIndex].start = splitOutput.index;
        stack[stackIndex].end = buildEntry.end;
        stack[stackIndex].parent = int32_t(nodeCount) - 1;
        stackIndex++;

        stack[stackIndex].start = buildEntry.start;
        stack[stackIndex].end = splitOutput.index;
        stack[stackIndex].parent = int32_t(nodeCount) - 1;
        stackIndex++;
    }

    if(nodes.size() > 0)
    {
        nodesAlloc.resize(nodes.size());
        nodesAlloc.write(nodes.data(), nodes.size());
    }

    std::vector<Triangle> sortedTriangles(triangleCount);
    for(uint32_t i = 0;i < triangleCount; ++i)
        sortedTriangles[i] = *buildTriangles[i].triangle;

    triangles = sortedTriangles;
}

CUDA_CALLABLE bool BVH::intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const
{
    if(nodesAlloc.getPtr() == nullptr || scene.getTriangles() == nullptr)
        return false;

    if(ray.isVisibilityRay && intersection.wasFound)
        return true;

    uint32_t stack[64];
    uint32_t stackIndex = 0;
    bool wasFound = false;

    stack[stackIndex++] = 0;
    while(stackIndex > 0)
    {
        uint32_t nodeIndex = stack[--stackIndex];
        const BVHNode& node = nodesAlloc.getPtr()[nodeIndex];

        if(node.rightOffset == 0)
        {
            //there are may exists some problem
            for(uint32_t i = 0;i < node.triangleCount; ++i)
            {
                if(scene.getTriangles()[node.triangleOffset + i].intersect(scene, ray, intersection))
                {
                    if(ray.isVisibilityRay)
                        return true;
                    wasFound = true;
                }
            }

            continue;
        }

        if(node.aabb.intersects(ray))
        {
            if(ray.directionIsNegative[node.splitAxis])
            {
                stack[stackIndex++] = nodeIndex + 1;
                stack[stackIndex++] = nodeIndex + uint32_t(node.rightOffset);
            }
            else
            {
                stack[stackIndex++] = nodeIndex + uint32_t(node.rightOffset);
                stack[stackIndex++] = nodeIndex + 1;
            }
        }
    }

    return wasFound;
}

BVHSplitOutput BVH::calculateSplit(std::vector<BVHBuildTriangle>& buildTriangles, std::vector<BVHSplitCache>& cache, uint32_t start, uint32_t end)
{
    assert(end > start);
    BVHSplitOutput output;
    float lowestCost = FLT_MAX;
    AABB fullAABB[3];

    for(uint32_t axis = 0; axis <= 2; ++axis)
    {
        PARALLEL_SORT(buildTriangles.begin() + start, buildTriangles.begin() + end, [axis](const BVHBuildTriangle& t1, const BVHBuildTriangle& t2){
            return (&t1.center.x)[axis] < (&t2.center.x)[axis];
        });

        AABB rightAABB;
        uint32_t rightCount = 0;

        for(int32_t i = end - 1;i >= int32_t(start); --i)
        {
            rightAABB.expand(buildTriangles[i].aabb);
            rightCount++;

            cache[i].aabb = rightAABB;
            cache[i].cost = rightAABB.getSurfaceArea() * float(rightCount);
        }

        AABB leftAABB;
        uint32_t leftCount = 0;
        for(uint32_t i = start; i < end; ++i)
        {
            leftAABB.expand(buildTriangles[i].aabb);
            leftCount++;

            float cost = leftAABB.getSurfaceArea() * float(leftCount);

            if(i + 1 < end)
                cost += cache[i + 1].cost;
            
            if(cost < lowestCost)
            {
                output.index = i + 1;
                output.axis = axis;
                output.leftAABB = leftAABB;

                if(output.index < end)
                    output.rightAABB = cache[output.index].aabb;

                lowestCost = cost;
            }
        }

        fullAABB[axis] = leftAABB;
    }

    assert(output.index >= start && output.index <= end);

    if(output.axis != 2)
    {
        PARALLEL_SORT(buildTriangles.begin() + start, buildTriangles.begin() + end, [output](const BVHBuildTriangle& t1, const BVHBuildTriangle& t2){
            return (&t1.center.x)[output.axis] < (&t2.center.x)[output.axis];
        });
    }

    output.fullAABB = fullAABB[output.axis];
    return output;
}