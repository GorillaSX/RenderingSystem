#ifndef __FILMQUAD_H__
#define __FILMQUAD_H__

#include <GL/glcorearb.h>

namespace Gorilla
{
    class Film;

    class FilmQuad 
    {
    public:
        void initialize();
        void render(const Film& film);

    private:

        GLuint vaoId = 0;
        GLuint vboId = 0;

        GLuint programId = 0;
        GLuint textureUniformId = 0;
        GLuint textureWidthUniformId = 0;
        GLuint textureHeightUniformId = 0;
        GLuint texelWidthUniformId = 0;
        GLuint texelHeightUniformId = 0;
    };
}

#endif //__FILMQUAD_H__