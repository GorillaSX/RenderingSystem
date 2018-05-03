#ifndef __APP_H__
#define __APP_H__

namespace Gorilla
{
    class Settings;
    class ConsoleRunner;
    class WindowRunner;
    class EyeTracker;

    class App 
    {
    public:
        static int run(int argc, char** argv);

        static Settings& getSettings();
        static ConsoleRunner& getConsoleRunner();
        static WindowRunner& getWindowRunner();
        static EyeTracker& getEyeTracker();
    };
}

#endif //__APP_H__