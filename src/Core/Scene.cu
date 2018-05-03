#include "commonHeaders.h"

#include "tinyformat/tinyformat.h"

#include "App.h"
#include "Core/Common.h"
#include "Core/Intersection.h"
#include "Core/Scene.h"
#include "Textures/Texture.h"
#include "Utils/Timer.h"
#include "Materials/Material.h"
#include <iostream>
#include "rapidjson/filereadstream.h"
#include "rapidjson/document.h"
#include <cstdio>
#include <iostream>
#include <fstream>

using namespace rapidjson;
using namespace Gorilla;

Scene::Scene() : texturesAlloc(false), materialAlloc(false), trianglesAlloc(false), emissiveTrianglesAlloc(false)
{
}
void Scene::load(std::string& fileName)
{
	FILE* fp = fopen(fileName.c_str(), "rb");
	char readBuffer[65536];
	FileReadStream is(fp, readBuffer, sizeof(readBuffer));

	Document d;
	d.ParseStream(is);
	Value::ConstMemberIterator it = d.FindMember("General");
	if(it != d.MemberEnd())
	{
		general.rayMinDistance = it->value["rayMinDistance"].GetFloat();
		general.backgroundColor = Color(it->value["backgroundColor"][0].GetFloat(), it->value["backgroundColor"][1].GetFloat(), it->value["backgroundColor"][2].GetFloat());
		general.offLensColor = Color(it->value["offLensColor"][0].GetFloat(), it->value["offLensColor"][1].GetFloat(), it->value["offLensColor"][2].GetFloat());
		general.normalMapping = it->value["normalMapping"].GetBool();
		general.normalInterpolation = it->value["normalInterpolation"].GetBool();
		general.normalVisualization = it->value["normalVisualization"].GetBool();
		general.interpolationVisualization = it->value["interpolationVisualization"].GetBool();
	}
	it = d.FindMember("Renderer");
	if(it != d.MemberEnd())
	{
		renderer.filtering = it->value["filtering"].GetBool();
		renderer.filter.type = FilterType(it->value["type"].GetInt());
	}
	it = d.FindMember("Camera");
	if(it != d.MemberEnd())
	{
		camera.position = Vector3(it->value["position"][0].GetFloat(), it->value["position"][1].GetFloat(), it->value["position"][2].GetFloat());
		camera.orientation = EulerAngle(it->value["orientation"][0].GetFloat(), it->value["orientation"][1].GetFloat(), it->value["orientation"][2].GetFloat());
	}
	it = d.FindMember("Models");
	ModelLoaderInfo model;
	model.modelFileName = it->value[0].GetString();
	models.push_back(model);
	fclose(fp);	
}
void Scene::initialize()
{
	std::cout << "start initializing Scene" << std::endl;

	Timer timer;

	allTextures.insert(allTextures.end(), textures.begin(), textures.end());
	allMaterials.insert(allMaterials.end(), materials.begin(), materials.end());
	allTriangles.insert(allTriangles.end(), triangles.begin(), triangles.end());

	// MODEL LOADING
	if (!models.empty())
	{
		ModelLoader modelLoader;

		for (ModelLoaderInfo& modelInfo : models)
		{
			ModelLoaderResult result = modelLoader.load(modelInfo);
	        std::map<int32_t, int32_t> texturesMap;
            std::map<int32_t, int32_t> materialMap;
			

			std::cout << "result.textures.size()" << result.textures.size() << std::endl;

            for(Texture& texture: result.textures)
            {
                texturesMap[texture.id] = allTextures.size();
                allTextures.push_back(texture);
            }
            for(Material& material: result.materials)
            {
                materialMap[material.id] = allMaterials.size();
				if(texturesMap.count(material.ambientTextureId) != 0)
                	material.ambientTextureIndex = texturesMap[material.ambientTextureId];
				if(texturesMap.count(material.diffuseTextureId) != 0)
    	            material.diffuseTextureIndex = texturesMap[material.diffuseTextureId];
				if(texturesMap.count(material.specularTextureId) != 0)
    	            material.specularTextureIndex = texturesMap[material.specularTextureId];
				if(texturesMap.count(material.bumpTextureId) != 0)
    	            material.bumpTextureIndex = texturesMap[material.bumpTextureId];
				if(texturesMap.count(material.shininessTextureId) != 0)
    	            material.shininessTextureIndex = texturesMap[material.shininessTextureId]; 
				if(texturesMap.count(material.emittanceTextureId) != 0)
    	            material.emittanceTextureIndex = texturesMap[material.emittanceTextureId]; 
				if(texturesMap.count(material.normalTextureId) != 0)
    	            material.normalTextureIndex = texturesMap[material.normalTextureId]; 
				if(texturesMap.count(material.maskTextureId) != 0)
    	            material.maskTextureIndex = texturesMap[material.maskTextureId]; 
                allMaterials.push_back(material);
            }
            for(Triangle& triangle: result.triangles)
            {
				if(materialMap.count(triangle.materialId) != 0)
                	triangle.materialIndex = materialMap[triangle.materialId];
                allTriangles.push_back(triangle);
            }
		}
	}



	for (uint32_t i = 0; i < allTextures.size(); ++i)
	{
		allTextures[i].initialize(*this);
	}


	for (Triangle& triangle : allTriangles)
	{
		triangle.initialize();
		if (allMaterials[triangle.materialIndex].isEmissive())
			emissiveTriangles.push_back(triangle);
	}

	// BVH BUILD

	bvh.build(allTriangles);

	// MEMORY ALLOC & WRITE

	if (allTextures.size() > 0)
	{
		texturesAlloc.resize(allTextures.size());
		texturesAlloc.write(allTextures.data(), allTextures.size());
	}
	
	if (allMaterials.size() > 0)
	{
		materialAlloc.resize(allMaterials.size());
		materialAlloc.write(allMaterials.data(), allMaterials.size());
	}
	
	if (allTriangles.size() > 0)
	{
		trianglesAlloc.resize(allTriangles.size());
		trianglesAlloc.write(allTriangles.data(), allTriangles.size());
	}
	
	if (emissiveTriangles.size() > 0)
	{
		emissiveTrianglesAlloc.resize(emissiveTriangles.size());
		emissiveTrianglesAlloc.write(emissiveTriangles.data(), emissiveTriangles.size());
	}

	emissiveTrianglesCount = uint32_t(emissiveTriangles.size());
	
	// MISC

	camera.initialize();
	imagePool.commit();
	std::cout << "end initializing Scene" << std::endl;
}

CUDA_CALLABLE bool Scene::intersect(const Ray& ray, Intersection& intersection) const
{
	return bvh.intersect(*this, ray, intersection);
}

CUDA_CALLABLE void Scene::calculateNormalMapping(Intersection& intersection) const
{
	const Material& material = getMaterial(intersection.materialIndex);

	if (!general.normalMapping || material.normalTextureIndex == -1)
		return;

	const Texture& normalTexture = getTexture(material.normalTextureIndex);
	Color normalColor = normalTexture.getColor(*this, intersection.texcoord, intersection.position);
	Vector3 normal(normalColor.r * 2.0f - 1.0f, normalColor.g * 2.0f - 1.0f, normalColor.b * 2.0f - 1.0f);
	Vector3 mappedNormal = intersection.onb.u * normal.x + intersection.onb.v * normal.y + intersection.onb.w * normal.z;
	intersection.normal = mappedNormal.normalized();
}

CUDA_CALLABLE const Texture* Scene::getTextures() const
{
	return texturesAlloc.getPtr();
}

CUDA_CALLABLE const Material* Scene::getMaterials() const
{
	return materialAlloc.getPtr();
}

CUDA_CALLABLE const Triangle* Scene::getTriangles() const
{
	return trianglesAlloc.getPtr();
}

CUDA_CALLABLE const Triangle* Scene::getEmissiveTriangles() const
{
	return emissiveTrianglesAlloc.getPtr();
}

CUDA_CALLABLE uint32_t Scene::getEmissiveTrianglesCount() const
{
	return emissiveTrianglesCount;
}

CUDA_CALLABLE const Texture& Scene::getTexture(uint32_t index) const
{
	return texturesAlloc.getPtr()[index];
}

CUDA_CALLABLE const Material& Scene::getMaterial(uint32_t index) const
{
	return materialAlloc.getPtr()[index];
}

CUDA_CALLABLE const Triangle& Scene::getTriangle(uint32_t index) const
{
	return trianglesAlloc.getPtr()[index];
}
