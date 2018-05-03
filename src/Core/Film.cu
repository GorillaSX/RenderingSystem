#include "commonHeaders.h"
#include <GL/gl3w.h>

#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include "Utils/CudaUtils.h"

#include "Core/Common.h"
#include "App.h"
#include "Core/Film.h"
#include "Tonemappers/Tonemapper.h"
#include "Utils/GLUtils.h"
#include "Utils/SysUtils.h"

using namespace Gorilla;

Film::Film(bool windowed_): windowed(windowed_)
{
}

void Film::initialize()
{
    if(windowed)
    {
        glGenTextures(1, &textureId);

        GLUtils::checkError("Could not create OpenGL texture");
        
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        GLUtils::checkError("Could not set OpenGL texture parameters");
    }
}

void Film::shutdown()
{
    if(windowed)
    {
        if(textureResource != nullptr)
        {
            CudaUtils::checkError(cudaGraphicsUnregisterResource(textureResource),"Could not unregister OpenGL texture");
            textureResource = nullptr;
        }

        glDeleteTextures(1, &textureId);
        GLUtils::checkError("Could not delete OpenGL texture");
    }
}

void Film::resize(uint32_t width_, uint32_t height_)
{
    width = width_;
    height = height_;
    length = width * height;

    cumulativeImage.resize(width, height);
    normalizedImage.resize(width, height);
    tonemappedImage.resize(width, height);

    if(windowed)
    {
        if(textureResource != nullptr)
        {
            CudaUtils::checkError(cudaGraphicsUnregisterResource(textureResource), "Could not unregister OpenGL texture");
            textureResource = nullptr;
        }

        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, GLsizei(width), GLsizei(height), 0, GL_RGBA, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_2D, 0);

        GLUtils::checkError("Could not reserve OpenGL texture memory");

        CudaUtils::checkError(cudaGraphicsGLRegisterImage(&textureResource, textureId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore), "Could not register OpenGL texture");
    }
    clear();
}

void Film::clear()
{
    cumulativeImage.clear();
    pixelSamples = 0;
    cleared = true;
}

bool Film::hasBeenCleared() const
{
    return cleared;
}

void Film::resetCleared() 
{
    cleared = false;
}

void Film::load(uint32_t width_, uint32_t height_, const std::string& fileName)
{

	uint64_t fileSize = SysUtils::getFileSize(fileName);

	if (fileSize != (width_ * height_ * sizeof(Color)))
		throw std::runtime_error("Film file has wrong size");

	std::ifstream file(fileName, std::ios::in | std::ios::binary);

	if (!file.is_open())
		throw std::runtime_error("Could not open the film file for reading");

	resize(width_, height_);

	file.read(reinterpret_cast<char*>(cumulativeImage.getData()), fileSize);
	file.close();

	cumulativeImage.upload();
}

void Film::loadMultiple(uint32_t width_, uint32_t height_, const std::string& dirName)
{

	resize(width_, height_);

	std::vector<std::string> fileNames = SysUtils::getAllFiles(dirName);
	std::vector<Color> inputData(length);

	Color* inputPtr = inputData.data();
	Color* cumulativePtr = cumulativeImage.getData();

	for (const std::string& fileName : fileNames)
	{
		uint64_t fileSize = SysUtils::getFileSize(fileName);

		if (fileSize != length * sizeof(Color))
			throw std::runtime_error("Film file has wrong size");

		std::ifstream file(fileName, std::ios::in | std::ios::binary);

		if (!file.is_open())
			throw std::runtime_error("Could not open the film file for reading");

		file.read(reinterpret_cast<char*>(inputPtr), fileSize);
		file.close();

		for (uint32_t i = 0; i < length; ++i)
			cumulativePtr[i] += inputPtr[i];
	}

	cumulativeImage.upload();
}

void Film::save(const std::string& fileName)const
{

	std::ofstream file(fileName, std::ios::out | std::ios::binary);

	if (!file.is_open())
		throw std::runtime_error("Could not open the film file for writing");

	file.write(reinterpret_cast<const char*>(cumulativeImage.getData()), sizeof(Color) * length);
	file.close();
}

CUDA_CALLABLE void Film::addSample(uint32_t x, uint32_t y, const Color& color, float filterWeight)
{
	Color temp = cumulativeImage.getPixel(x, y);

	temp.r += color.r * filterWeight;
	temp.g += color.g * filterWeight;
	temp.b += color.b * filterWeight;
	temp.a += filterWeight;

	cumulativeImage.setPixel(x, y, temp);
}

CUDA_CALLABLE void Film::addSample(uint32_t index, const Color& color, float filterWeight)
{
	Color temp = cumulativeImage.getPixel(index);

	temp.r += color.r * filterWeight;
	temp.g += color.g * filterWeight;
	temp.b += color.b * filterWeight;
	temp.a += filterWeight;

	cumulativeImage.setPixel(index, temp);
}

__global__ void normalizeKernel(cudaSurfaceObject_t cumulative, cudaSurfaceObject_t normalized, uint32_t width, uint32_t height)
{
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
		return;

	float4 color;
	surf2Dread(&color, cumulative, x * sizeof(float4), y);

	color.x /= color.w;
	color.y /= color.w;
	color.z /= color.w;
	color.w = 1.0f;

	surf2Dwrite(color, normalized, x * sizeof(float4), y);
}


void Film::normalize()
{
		dim3 dimBlock(16, 16);
		dim3 dimGrid;

		dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

		normalizeKernel<<<dimGrid, dimBlock>>>(cumulativeImage.getSurfaceObject(), normalizedImage.getSurfaceObject(), width, height);
		CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch normalize kernel");
		CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute normalize kernel");
}

__global__ void tonemapKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output, uint32_t width, uint32_t height)
{
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
		return;

	float4 temp;
	surf2Dread(&temp, input, x * sizeof(float4), y);

	Color color(temp.x, temp.y, temp.z, temp.w);
	color.clamp();
	color = Color::pow(color, 1.0f / 2.2f);
	color.a = 1.0f;

	surf2Dwrite(make_float4(color.r, color.g, color.b, color.a), output, x * sizeof(float4), y);
}


void Film::tonemap()
{
		dim3 dimBlock(16, 16);
		dim3 dimGrid;

		dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

		tonemapKernel<<<dimGrid, dimBlock>>>(normalizedImage.getSurfaceObject(), tonemappedImage.getSurfaceObject(), width, height);
		CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch tonemap kernel");
		CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute tonemap kernel");
}

__global__ void updateTextureKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output, uint32_t width, uint32_t height)
{
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
		return;

	float4 color;
	surf2Dread(&color, input, x * sizeof(float4), y);
	surf2Dwrite(color, output, x * sizeof(float4), y);
}

void Film::updateTexture()
{
	if (windowed)
	{
			CudaUtils::checkError(cudaGraphicsMapResources(1, &textureResource, 0), "Could not map texture resource");

			cudaArray_t textureData;
			CudaUtils::checkError(cudaGraphicsSubResourceGetMappedArray(&textureData, textureResource, 0, 0), "Could not get mapped array");

			cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = textureData;

			cudaSurfaceObject_t surfaceObject;
			CudaUtils::checkError(cudaCreateSurfaceObject(&surfaceObject, &resDesc), "Could not create surface object");
			
			dim3 dimBlock(16, 16);
			dim3 dimGrid;

			dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

			updateTextureKernel<<<dimGrid, dimBlock>>>(tonemappedImage.getSurfaceObject(), surfaceObject, width, height);
			CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch update texture kernel");
			CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute update texture kernel");

			CudaUtils::checkError(cudaDestroySurfaceObject(surfaceObject), "Could not destroy surface object");
			CudaUtils::checkError(cudaGraphicsUnmapResources(1, &textureResource, 0), "Could not unmap texture resource");
	}
}

Color Film::getCumulativeColor(uint32_t x, uint32_t y) const
{
	return cumulativeImage.getPixel(x, y);
}

Color Film::getNormalizedColor(uint32_t x, uint32_t y) const
{
	return normalizedImage.getPixel(x, y);
}

Color Film::getTonemappedColor(uint32_t x, uint32_t y) const
{
	return tonemappedImage.getPixel(x, y);
}

CUDA_CALLABLE Image& Film::getCumulativeImage()
{
	return cumulativeImage;
}

CUDA_CALLABLE Image& Film::getNormalizedImage()
{
	return normalizedImage;
}

CUDA_CALLABLE Image& Film::getTonemappedImage()
{
	return tonemappedImage;
}

CUDA_CALLABLE uint32_t Film::getWidth() const
{
	return width;
}

CUDA_CALLABLE uint32_t Film::getHeight() const
{
	return height;
}

CUDA_CALLABLE uint32_t Film::getLength() const
{
	return length;
}

GLuint Film::getTextureId() const
{
	return textureId;
}
