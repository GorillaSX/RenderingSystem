#ifndef __GLUTILS_H__
#define __GLUTILS_H__

#include <string>
#include <GL/glcorearb.h>

namespace Gorilla
{
    class GLUtils 
    {
    public:
        static GLuint buildProgram(const std::string& vertexShaderPath, const std::string& fragmentShaderPath);

        static void checkError(const std::string& message);
        static std::string getErrorMessage(int32_t result);
    };
}

#endif //__GLUTILS_H__