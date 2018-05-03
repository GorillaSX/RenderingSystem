#ifndef __SYSUTILS_H__
#define __SYSUTILS_H__

namespace Gorilla
{
    class SysUtils 
    {
    public:

        static void openFileExternally(const std::string& fileName);
        static uint64_t getFileSize(const std::string& fileName);
        static std::vector<std::string> getAllFiles(const std::string& dirName);
    };
}

#endif //__SYSUTILS_H__