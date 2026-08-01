#include "ps2_runtime.h"
#include "games_database.h"
#if defined(PS2X_ENABLE_DEBUG_UI) && !defined(PLATFORM_VITA)
#include "ps2_debug_panel.h"
#endif

#ifdef _DEBUG
#include "ps2_log.h"
#endif

#include <iostream>
#include <string>
#include <filesystem>
#include <exception>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#if defined(__ANDROID__)
#include <android/log.h>
#include <unistd.h>
#include <thread>
#include <cstdio>
#include <cstring>
#endif

namespace
{
#if defined(__ANDROID__)
    void redirectStdioToLogcat()
    {
        static int pipeFds[2];
        if (pipe(pipeFds) != 0)
        {
            return;
        }

        setvbuf(stdout, nullptr, _IOLBF, 0);
        setvbuf(stderr, nullptr, _IONBF, 0);
        dup2(pipeFds[1], STDOUT_FILENO);
        dup2(pipeFds[1], STDERR_FILENO);

        std::thread([]()
                    {
                        FILE *reader = fdopen(pipeFds[0], "r");
                        if (!reader)
                        {
                            return;
                        }
                        char line[1024];
                        while (fgets(line, sizeof(line), reader))
                        {
                            size_t len = std::strlen(line);
                            if (len > 0 && line[len - 1] == '\n')
                            {
                                line[len - 1] = '\0';
                            }
                            __android_log_write(ANDROID_LOG_INFO, "ps2x", line);
                        }
                    })
            .detach();
    }
#endif

    void setupTerminateLogger() // to help on release build crashs
    {
        std::set_terminate([]()
                           {
                               std::cerr << "[terminate] unhandled exception" << std::endl;
                               const std::exception_ptr ep = std::current_exception();
                               if (ep)
                               {
                                   try
                                   {
                                       std::rethrow_exception(ep);
                                   }
                                   catch (const std::system_error &e)
                                   {
                                       std::cerr << "[terminate] std::system_error code=" << e.code().value()
                                                 << " category=" << e.code().category().name()
                                                 << " message=" << e.what() << std::endl;
                                   }
                                   catch (const std::exception &e)
                                   {
                                       std::cerr << "[terminate] std::exception: " << e.what() << std::endl;
                                   }
                                   catch (...)
                                   {
                                       std::cerr << "[terminate] non-std exception" << std::endl;
                                   }
                               }
                               std::abort(); });
    }

    std::string normalizeGameId(const std::string &folderName)
    {
        std::string result = folderName;

        size_t underscore = result.find('_');
        if (underscore != std::string::npos)
            result[underscore] = '-';

        size_t dot = result.find('.');
        if (dot != std::string::npos)
            result.erase(dot, 1);

        std::ranges::transform(result, result.begin(), [](unsigned char character)
                               { return static_cast<char>(std::toupper(character)); });

        return result;
    }

    bool parseRendererMode(
        const std::string &text,
        GsRendererMode &mode)
    {
        if (text == "software")
            mode = GsRendererMode::Software;
        else if (text == "hybrid")
            mode = GsRendererMode::Hybrid;
        else if (text == "verify")
            mode = GsRendererMode::Verify;
        else if (text == "gpu-strict")
            mode = GsRendererMode::GpuStrict;
        else
            return false;
        return true;
    }

    void printUsage(const char *program)
    {
        std::cout
            << "Usage: " << program
            << " [--renderer software|hybrid|verify|gpu-strict]"
               " [ELF [CD_IMAGE]]\n";
    }

    struct RunnerOptions
    {
        bool showHelp = false;
        GsRendererMode rendererMode = GsRendererMode::Software;
        std::vector<std::filesystem::path> positionalPaths;
    };

    RunnerOptions parseRunnerOptions(int argc, char *argv[])
    {
        RunnerOptions options{};
        bool optionsEnded = false;
        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i] ? argv[i] : "";
            if (!optionsEnded && (argument == "--help" || argument == "-h"))
            {
                options.showHelp = true;
                continue;
            }
            if (!optionsEnded && argument == "--")
            {
                optionsEnded = true;
                continue;
            }
            if (!optionsEnded && argument == "--renderer")
            {
                if (++i >= argc || !argv[i] ||
                    !parseRendererMode(argv[i], options.rendererMode))
                {
                    throw std::invalid_argument(
                        "--renderer requires software, hybrid, verify, or gpu-strict");
                }
                continue;
            }
            if (!optionsEnded && !argument.empty() && argument[0] == '-')
            {
                throw std::invalid_argument("unknown option: " + argument);
            }
            options.positionalPaths.emplace_back(argument);
        }
        if (options.positionalPaths.size() > 2u)
        {
            throw std::invalid_argument(
                "expected at most an ELF and optional CD image path");
        }
        return options;
    }

    std::filesystem::path getExecutablePath(const RunnerOptions &options)
    {
        if (!options.positionalPaths.empty())
        {
            std::cout << "Using argv boot path" << std::endl;
            return options.positionalPaths[0];
        }
#if defined(PS2X_DEFAULT_BOOT_ELF)
        std::cout << "Using default boot file" << std::endl;
        const std::filesystem::path configuredPath = std::filesystem::path(PS2X_DEFAULT_BOOT_ELF);
#if defined(PLATFORM_VITA)
        return configuredPath;
#endif
        if (configuredPath.is_absolute())
        {
            return configuredPath;
        }
        return (std::filesystem::current_path() / configuredPath).lexically_normal();
#else
        throw std::runtime_error("Unable to determine executable path. Pass the guest ELF as argv[1] or define PS2X_DEFAULT_BOOT_ELF.");
#endif
    }

    void configureOptionalCdImage(
        PS2Runtime &runtime,
        const RunnerOptions &options)
    {
        if (options.positionalPaths.size() < 2u)
        {
            return;
        }

        PS2Runtime::IoPaths ioPaths = runtime.ioPaths();
        ioPaths.cdImage = options.positionalPaths[1];
        runtime.configureIoPaths(ioPaths);
        std::cout << "Using argv CD image path: "
                  << runtime.ioPaths().cdImage << std::endl;
    }
}

int main(int argc, char *argv[])
{
#if defined(__ANDROID__)
    redirectStdioToLogcat();
#endif
    setupTerminateLogger();

    try
    {
        const RunnerOptions options = parseRunnerOptions(argc, argv);
        if (options.showHelp)
        {
            printUsage(argc > 0 && argv[0] ? argv[0] : "ps2EntryRunner");
            return 0;
        }

        std::filesystem::path pathObj = getExecutablePath(options);

        std::string filePathStr = pathObj.string();
        std::string elfName = pathObj.filename().string();
        std::string normalizedId = normalizeGameId(elfName);

        std::string windowTitle = "PS2-Recomp | ";
        const char *gameName = getGameName(normalizedId);

#if !defined(PLATFORM_VITA)
        if (gameName)
        {
            windowTitle += std::string(gameName) + " | " + elfName;
        }
        else
#endif
        {
            windowTitle += elfName;
        }

        PS2Runtime runtime;
        configureOptionalCdImage(runtime, options);
#if defined(PS2X_ENABLE_DEBUG_UI) && !defined(PLATFORM_VITA)
        // This hook is to prevent leak rlimgui deps to recompiler etc
        PS2DebugPanel debugPanel;
        runtime.setDebugUiCallbacks(
            [](PS2Runtime &rt, void *userData)
            {
                (void)rt;
                static_cast<PS2DebugPanel *>(userData)->initialize();
            },
            [](PS2Runtime &rt, void *userData)
            {
                static_cast<PS2DebugPanel *>(userData)->draw(rt);
            },
            [](PS2Runtime &rt, void *userData)
            {
                (void)rt;
                static_cast<PS2DebugPanel *>(userData)->shutdown();
            },
            &debugPanel);
#endif
        if (!runtime.initialize(windowTitle.c_str()))
        {
            std::cerr << "Failed to initialize PS2 runtime" << std::endl;
            return 1;
        }
        if (!runtime.gs().setRendererMode(options.rendererMode))
        {
            std::cerr
                << "Failed to select GS renderer '"
                << gsRendererModeName(options.rendererMode) << "': "
                << runtime.gs().rendererDiagnostic() << std::endl;
            return 1;
        }
        std::cout
            << "GS renderer: "
            << gsRendererModeName(runtime.gs().rendererMode())
            << std::endl;

        if (!runtime.loadELF(filePathStr))
        {
            std::cerr << "Failed to load ELF file: " << filePathStr << std::endl;
            return 1;
        }

        runtime.run();

#ifdef _DEBUG
        ps2_log::print_saved_location();
#endif
        std::cout.flush();
        std::cerr.flush();
        std::_Exit(0);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[main] fatal exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[main] fatal exception: unknown" << std::endl;
    }

    std::cout.flush();
    std::cerr.flush();
    std::_Exit(1);
}
