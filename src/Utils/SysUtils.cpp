#include "commonHeaders.h"

#include <unistd.h>

#include <sys/stat.h>
#include <boost/filesystem.hpp>

#include "Utils/SysUtils.h"
#include "App.h"

using namespace Gorilla;
namespace bf = boost::filesystem;

void SysUtils::openFileExternally(const std::string& fileName)
{

	int32_t pid = fork();

	if (pid == 0)
	{
		char* arg[] = { (char*)"xdg-open", (char*)fileName.c_str(), (char*)nullptr };
		if (execvp(arg[0], arg) == -1)
			printf("Could not open file externally (%d)", errno);
	}
}

uint64_t SysUtils::getFileSize(const std::string& fileName)
{
	struct stat stat_buf;
	int rc = stat(fileName.c_str(), &stat_buf);
	return rc == 0 ? stat_buf.st_size : 0;
}

std::vector<std::string> SysUtils::getAllFiles(const std::string& dirName)
{
	std::vector<std::string> files;
	bf::path directory(dirName);
	bf::directory_iterator end;

	for (bf::directory_iterator it(directory); it != end; ++it)
	{
		if (bf::is_regular_file(it->path()))
			files.push_back(bf::absolute(it->path()).string());
	}

	return files;
}
