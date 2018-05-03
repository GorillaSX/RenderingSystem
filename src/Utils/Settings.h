#ifndef __SETTINGS_H__
#define __SETTINGS_H__

namespace Gorilla
{
    class Settings 
    {
    public:

        bool load(int argc, char** argv);

        struct General
        {
            bool windowed;
            uint32_t cudaDeviceNumber;
        } general;

        struct Renderer
        {
            uint32_t pixelSamples;
        } renderer;

        struct Window 
        {
            uint32_t width;
            uint32_t height;
            bool fullscreen;
            bool vsync;
            bool hideCursor;
            float renderScale;
            bool checkGLErrors;
        } window;

        struct Scene 
        {
            std::string fileName;
        } scene;

        struct Image
        {
            uint32_t width;
            uint32_t height;
            bool write;
            std::string fileName;
            bool autoView;
        } image;
    };
}
#endif //__SETTINGS_H__