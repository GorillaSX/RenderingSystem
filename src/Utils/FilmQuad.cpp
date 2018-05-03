#include "commonHeaders.h"
#include <GL/gl3w.h>
#include "Core/Film.h"
#include "Utils/FilmQuad.h"
#include "Utils/GLUtils.h"
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

using namespace Gorilla;

void FilmQuad::initialize() 
{
    programId = GLUtils::buildProgram("data/shaders/film.vert", "data/shaders/film.frag");

    textureUniformId = glGetUniformLocation(programId, "tex0");
    textureWidthUniformId = glGetUniformLocation(programId, "textureWidth");
    textureHeightUniformId = glGetUniformLocation(programId, "textureHeight");
    texelWidthUniformId = glGetUniformLocation(programId, "texelWidth");
    texelHeightUniformId = glGetUniformLocation(programId, "texelHeight");

    GLUtils::checkError("Could not get GLSL uniforms");

    const GLfloat vertexData[] = 
    {
        -1.0f, -1.0f, 0.0f, 0.0f,
        1.0f, -1.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 
        -1.0f, 1.0f, 0.0f, 1.0f,
    };


    glGenBuffers(1, &vboId);
    glBindBuffer(GL_ARRAY_BUFFER, vboId);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertexData), vertexData, GL_STATIC_DRAW);

    glGenVertexArrays(1, &vaoId);
    glBindVertexArray(vaoId);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), nullptr);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), reinterpret_cast<void*>(2 * sizeof(GLfloat)));
    glBindVertexArray(0);

    GLUtils::checkError("Could not set openGL buffer parameters");
}

void FilmQuad::render(const Film& film)
{
    /*unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int width, height, nrChannels;
    stbi_set_flip_vertically_on_load(true);
    unsigned char* data = stbi_load("tonemappedImage.png", &width, &height, &nrChannels, 0);
    if(data)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else
    {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);*/
    glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, film.getTextureId());

	glUseProgram(programId);
	glUniform1i(textureUniformId, 0);
	glUniform1f(textureWidthUniformId, float(film.getWidth()));
	glUniform1f(textureHeightUniformId, float(film.getHeight()));
	glUniform1f(texelWidthUniformId, 1.0f / film.getWidth());
	glUniform1f(texelHeightUniformId, 1.0f / film.getHeight());

	glBindVertexArray(vaoId);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glUseProgram(0);

	GLUtils::checkError("Could not render the film");
}