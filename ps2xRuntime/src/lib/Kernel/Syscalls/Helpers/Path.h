#include "FileRuntimeState.h"

namespace
{
    constexpr char kMc0Prefix[] = "mc0:";

    std::string toLowerAscii(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    std::string stripIsoVersionSuffix(std::string value)
    {
        const std::size_t semicolon = value.find(';');
        if (semicolon == std::string::npos)
        {
            return value;
        }

        bool numericSuffix = semicolon + 1 < value.size();
        for (std::size_t i = semicolon + 1; i < value.size(); ++i)
        {
            if (!std::isdigit(static_cast<unsigned char>(value[i])))
            {
                numericSuffix = false;
                break;
            }
        }

        if (numericSuffix)
        {
            value.erase(semicolon);
        }
        return value;
    }

    std::string normalizePs2PathSuffix(std::string suffix)
    {
        std::replace(suffix.begin(), suffix.end(), '\\', '/');
        suffix = stripIsoVersionSuffix(std::move(suffix));
        while (!suffix.empty() && (suffix.front() == '/' || suffix.front() == '\\'))
        {
            suffix.erase(suffix.begin());
        }
        return suffix;
    }

    bool guestPathSuffixIsAbsolute(
        const std::string &suffix)
    {
        return !suffix.empty() &&
               (suffix.front() == '/' ||
                suffix.front() == '\\');
    }

    std::filesystem::path fallbackHostDirectory()
    {
        std::error_code ec;
        const std::filesystem::path cwd =
            std::filesystem::current_path(ec);
        return ec ? std::filesystem::path(".")
                  : cwd.lexically_normal();
    }

    std::filesystem::path configuredHostRoot(
        const PS2Runtime::IoPaths &paths)
    {
        if (!paths.hostRoot.empty())
        {
            return paths.hostRoot;
        }
        if (!paths.elfDirectory.empty())
        {
            return paths.elfDirectory;
        }
        return fallbackHostDirectory();
    }

    std::filesystem::path configuredCdRoot(
        const PS2Runtime::IoPaths &paths)
    {
        if (!paths.cdRoot.empty())
        {
            return paths.cdRoot;
        }
        if (!paths.elfDirectory.empty())
        {
            return paths.elfDirectory;
        }
        return fallbackHostDirectory();
    }

    std::filesystem::path configuredMcRoot(
        const PS2Runtime::IoPaths &paths)
    {
        if (!paths.mcRoot.empty())
        {
            return paths.mcRoot;
        }
        if (!paths.elfDirectory.empty())
        {
            return paths.elfDirectory / "mc0";
        }
        return (fallbackHostDirectory() / "mc0")
            .lexically_normal();
    }

    struct GuestPathResolverSnapshot
    {
        std::filesystem::path hostBase;
        std::filesystem::path cdromBase;
        std::filesystem::path mcBase;
        std::filesystem::path hostCwd;
        std::filesystem::path cdromCwd;
        std::filesystem::path mcCwd;
        std::string cwdDevice = "cdrom0";
    };

    GuestPathResolverSnapshot
    guestPathResolverSnapshot(
        PS2Runtime *runtime)
    {
        GuestPathResolverSnapshot snapshot;
        if (!runtime)
        {
            return snapshot;
        }

        const PS2Runtime::IoPaths paths =
            runtime->ioPaths();
        const std::filesystem::path hostBase =
            configuredHostRoot(paths);
        const std::filesystem::path cdromBase =
            configuredCdRoot(paths);
        const std::filesystem::path mcBase =
            configuredMcRoot(paths);

        EeFileRuntimeState &state =
            runtime->eeFileRuntimeState();
        std::lock_guard<std::mutex> lock(
            state.pathMutex);
        if (!state.pathsInitialized ||
            state.hostBase != hostBase ||
            state.cdromBase != cdromBase ||
            state.mcBase != mcBase)
        {
            state.pathsInitialized = true;
            state.hostBase = hostBase;
            state.cdromBase = cdromBase;
            state.mcBase = mcBase;
            state.hostCwd = hostBase;
            state.cdromCwd = cdromBase;
            state.mcCwd = mcBase;
            // Preserve the old bare-path behavior: before any guest chdir,
            // unqualified paths resolve against the configured CD root.
            state.cwdDevice = "cdrom0";
        }

        snapshot.hostBase = state.hostBase;
        snapshot.cdromBase = state.cdromBase;
        snapshot.mcBase = state.mcBase;
        snapshot.hostCwd = state.hostCwd;
        snapshot.cdromCwd = state.cdromCwd;
        snapshot.mcCwd = state.mcCwd;
        snapshot.cwdDevice = state.cwdDevice;
        return snapshot;
    }

    void setGuestWorkingDirectory(
        PS2Runtime *runtime,
        const char *ps2Path,
        const std::filesystem::path &resolvedPath)
    {
        if (!runtime || !ps2Path || !*ps2Path)
        {
            return;
        }

        const GuestPathResolverSnapshot resolver =
            guestPathResolverSnapshot(runtime);
        const std::string lower =
            toLowerAscii(std::string(ps2Path));
        std::string device = resolver.cwdDevice;
        if (lower.rfind("host0:", 0) == 0 ||
            lower.rfind("host:", 0) == 0)
        {
            device = "host0";
        }
        else if (lower.rfind("cdrom0:", 0) == 0 ||
                 lower.rfind("cdrom:", 0) == 0)
        {
            device = "cdrom0";
        }
        else if (lower.rfind(kMc0Prefix, 0) == 0)
        {
            device = "mc0";
        }
        else if (lower.size() > 1u &&
                 lower[1] == ':')
        {
            // A native absolute host path has no modeled PS2 device.
            return;
        }

        EeFileRuntimeState &state =
            runtime->eeFileRuntimeState();
        std::lock_guard<std::mutex> lock(
            state.pathMutex);
        if (device == "host0")
        {
            state.hostCwd =
                resolvedPath.lexically_normal();
        }
        else if (device == "mc0")
        {
            state.mcCwd =
                resolvedPath.lexically_normal();
        }
        else
        {
            device = "cdrom0";
            state.cdromCwd =
                resolvedPath.lexically_normal();
        }
        state.cwdDevice = std::move(device);
    }
}
