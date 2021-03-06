#include "commonHeaders.h"
#include <boost/program_options.hpp>
#include "Utils/Settings.h"

using namespace Gorilla;

namespace po = boost::program_options;

bool Settings::load(int argc, char** argv)
{
    po::options_description options("Options");

    options.add_options()
        ("help", "")
        ("general.windowed", po::value(&general.windowed)->default_value(true), "")
		("general.cudaDeviceNumber", po::value(&general.cudaDeviceNumber)->default_value(0), "")

		("renderer.pixelSamples", po::value(&renderer.pixelSamples)->default_value(1), "")

		("window.width", po::value(&window.width)->default_value(1280), "")
		("window.height", po::value(&window.height)->default_value(800), "")
		("window.fullscreen", po::value(&window.fullscreen)->default_value(false), "")
		("window.vsync", po::value(&window.vsync)->default_value(false), "")
		("window.hideCursor", po::value(&window.hideCursor)->default_value(false), "")
		("window.renderScale", po::value(&window.renderScale)->default_value(0.25f), "")
		("window.checkGLErrors", po::value(&window.checkGLErrors)->default_value(true), "")

		("scene.fileName", po::value(&scene.fileName)->default_value("../scene.xml"), "")

		("image.width", po::value(&image.width)->default_value(1280), "")
		("image.height", po::value(&image.height)->default_value(800), "")
		("image.write", po::value(&image.write)->default_value(true), "")
		("image.fileName", po::value(&image.fileName)->default_value("image.png"), "")
		("image.autoView", po::value(&image.autoView)->default_value(true), "");

    std::ifstream iniFile("Gorilla.ini");
    po::variables_map vm;

    try
    {
        po::store(po::parse_command_line(argc, argv, options),vm);
        po::store(po::parse_config_file(iniFile, options), vm);
        po::notify(vm);
    }
    catch(const po::error& e)
    {
        std::cout << "Command line / settings file parsing failed: " << e.what() << std::endl;
        std::cout << "Try '--help' for list of valid options" << std::endl;

        return false;
    }

    if(vm.count("help"))
    {
        std::cout << options;
        return false;
    }
    
    return true;
}