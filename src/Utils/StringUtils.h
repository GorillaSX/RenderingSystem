#ifndef __STRINGUTILS_H__
#define __STRINGUTILS_H__

namespace Gorilla
{
    class StringUtils 
    {
    public:
        static bool endsWith(const std::string& input, const std::string& end);
        static std::string readFileToString(const std::string& fileName);
        static std::string humanizeNumber(double value, bool usePowerOfTwo = false);
    };
}

#endif //__STRINGUTILS_H__