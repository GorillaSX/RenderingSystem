#include "commonHeaders.h"
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

#include "Utils/GLUtils.h"
#include "App.h"
#include "Utils/StringUtils.h"
#include "tinyformat/tinyformat.h"

using namespace Gorilla;

GLuint GLUtils::buildProgram(const std::string& vertexShaderPath, const std::string& fragmentShaderPath)
{
    std::string vertexShaderString = StringUtils::readFileToString(vertexShaderPath);
    const GLchar* vertexShaderStringPtr = vertexShaderString.c_str();

    std::string fragmentShaderString = StringUtils::readFileToString(fragmentShaderPath);
    const GLchar* fragmentShaderStringPtr = fragmentShaderString.c_str();

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderStringPtr, nullptr);
    glCompileShader(vertexShader);

    GLint isCompiled = 0;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &isCompiled);

    if(isCompiled == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &maxLength);
        std::vector<GLchar> infoLog(maxLength);
        glGetShaderInfoLog(vertexShader, maxLength, &maxLength, &infoLog[0]);
        glDeleteShader(vertexShader);

        throw std::runtime_error(tfm::format("Could not compile vertex shader: %s", &infoLog[0]));
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderStringPtr, nullptr);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &isCompiled);

    if(isCompiled == GL_FALSE)
    {
        GLint maxLength = 0;
        glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &maxLength);

        std::vector<GLchar> infoLog(maxLength);
        glGetShaderInfoLog(fragmentShader, maxLength, &maxLength, &infoLog[0]);
        glDeleteShader(fragmentShader);
        glDeleteShader(vertexShader);

        throw std::runtime_error(tfm::format("Could not compile fragment shader", &infoLog[0]));
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    GLint isLinked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &isLinked);

    if(isLinked == GL_FALSE)
    {
        GLint maxLength  = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

        std::vector<GLchar> infoLog(maxLength);
        glGetProgramInfoLog(program, maxLength, &maxLength, &infoLog[0]);
        glDeleteProgram(program);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        throw std::runtime_error(tfm::format("Could not link shader program", &infoLog[0]));
    }

    glDetachShader(program, vertexShader);
    glDetachShader(program, fragmentShader);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}

void GLUtils::checkError(const std::string& message)
{
    int32_t result = glGetError();
    if(result != GL_NO_ERROR)
        throw std::runtime_error(tfm::format("OpenGL error: %s: %s", message, getErrorMessage(result)));
}

std::string GLUtils::getErrorMessage(int32_t result)
{
    switch(result)
    {
        case GL_NO_ERROR: return "GL_NO_ERROR";
        case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
        case GL_STACK_OVERFLOW: return "GL_STACK_OVERFLOW";
        case GL_STACK_UNDERFLOW: return "GL_STACK_UNDERFLOW";
        case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
        default: return "Unknown error";
    }
}