#include <vector>
#include <fstream>
#include <filesystem>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

std::filesystem::path sDatasetDirectory;
std::string GenerateSimilarText(const std::string& trainingText);

// Scan parent directories for a subdirectory datasets
std::filesystem::path DatasetDirectory(const char* exePath)
{
    std::filesystem::path p(exePath);
    while (true)
    {
        p = p.parent_path();
        if (std::filesystem::exists(p / "datasets"))
            break;
    }
    return p / "datasets";
}

int main(int argc, char** argv, char** env)
{
#ifdef _WIN32
    // console UTF-8
    SetConsoleOutputCP(CP_UTF8);
#endif

    sDatasetDirectory = DatasetDirectory(argv[0]);
    std::ifstream ts(sDatasetDirectory / "Shakespeare/tiny-shakespeare.txt");
    std::string str(std::istreambuf_iterator<char>{ts}, {});
    GenerateSimilarText(str);
	return 0;
}
