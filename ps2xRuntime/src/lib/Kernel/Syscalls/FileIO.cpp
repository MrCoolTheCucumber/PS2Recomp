#include "Common.h"
#include "FileIO.h"
#include "Helpers/FileRuntimeState.h"

namespace ps2_syscalls
{
    namespace
    {
        constexpr size_t kVagAccumMaxBytes =
            16u * 1024u * 1024u;

        EeFileRuntimeState *getFileRuntimeState(
            PS2Runtime *runtime)
        {
            return runtime
                ? &runtime->eeFileRuntimeState()
                : nullptr;
        }

        int allocatePs2Fd(
            EeFileRuntimeState &state,
            FILE *file)
        {
            if (!file)
            {
                return -1;
            }

            std::lock_guard<std::mutex> lock(
                state.descriptorMutex);
            int fd = state.nextDescriptor++;
            state.descriptors[fd] = file;
            return fd;
        }

        const char *translateFioMode(int ps2Flags)
        {
            bool read = (ps2Flags & PS2_FIO_O_RDONLY) || (ps2Flags & PS2_FIO_O_RDWR);
            bool write = (ps2Flags & PS2_FIO_O_WRONLY) || (ps2Flags & PS2_FIO_O_RDWR);
            bool append = (ps2Flags & PS2_FIO_O_APPEND);
            bool create = (ps2Flags & PS2_FIO_O_CREAT);
            bool truncate = (ps2Flags & PS2_FIO_O_TRUNC);

            if (read && write)
            {
                if (create && truncate)
                    return "w+b";
                if (create)
                    return "a+b";
                return "r+b";
            }
            else if (write)
            {
                if (append)
                    return "ab";
                if (create && truncate)
                    return "wb";
                if (create)
                    return "wx";
                return "r+b";
            }
            else if (read)
            {
                return "rb";
            }
            return "rb";
        }
    }

    void resetFileIoState(PS2Runtime *runtime)
    {
        if (EeFileRuntimeState *const state =
                getFileRuntimeState(runtime))
        {
            state->closeAll();
        }
    }

    void fioOpen(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        EeFileRuntimeState *const state =
            getFileRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, -1);
            return;
        }

        uint32_t pathAddr = getRegU32(ctx, 4); // $a0
        int flags = (int)getRegU32(ctx, 5);    // $a1 (PS2 FIO flags)

        const char *ps2Path = reinterpret_cast<const char *>(getConstMemPtr(rdram, pathAddr));
        if (!ps2Path)
        {
            std::cerr << "fioOpen error: Invalid path address" << std::endl;
            setReturnS32(ctx, -1);
            return;
        }

        std::string hostPath =
            translatePs2Path(ps2Path, runtime);
        if (hostPath.empty())
        {
            std::cerr << "fioOpen error: Failed to translate path '" << ps2Path << "'" << std::endl;
            setReturnS32(ctx, -1);
            return;
        }

        const char *mode = translateFioMode(flags);
        RUNTIME_LOG("fioOpen: '" << hostPath << "' flags=0x" << std::hex << flags << std::dec << " mode='" << mode << "'");

        FILE *fp = ::fopen(hostPath.c_str(), mode);
        if (!fp)
        {
            std::cerr << "fioOpen error: fopen failed for '" << hostPath << "': " << strerror(errno) << std::endl;
            setReturnS32(ctx, -1); // e.g., -ENOENT, -EACCES
            return;
        }

        int ps2Fd = allocatePs2Fd(*state, fp);
        if (ps2Fd < 0)
        {
            std::cerr << "fioOpen error: Failed to allocate PS2 file descriptor" << std::endl;
            ::fclose(fp);
            setReturnS32(ctx, -1); // e.g., -EMFILE
            return;
        }

        // returns the PS2 file descriptor
        setReturnS32(ctx, ps2Fd);
    }

    void fioClose(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        EeFileRuntimeState *const state =
            getFileRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, -1);
            return;
        }

        const int ps2Fd =
            static_cast<int>(getRegU32(ctx, 4));
        int ret = -1;
        {
            std::lock_guard<std::mutex> lock(
                state->descriptorMutex);
            const auto it =
                state->descriptors.find(ps2Fd);
            if (it == state->descriptors.end() ||
                !it->second)
            {
                std::cerr
                    << "fioClose warning: Invalid PS2 file descriptor "
                    << ps2Fd << std::endl;
                setReturnS32(ctx, -1);
                return;
            }
            ret = ::fclose(it->second);
            state->descriptors.erase(it);
        }

        EeFileVagAccumEntry completedVag{};
        {
            std::lock_guard<std::mutex> lock(
                state->vagMutex);
            const auto it =
                state->vagAccum.find(ps2Fd);
            if (it != state->vagAccum.end())
            {
                completedVag = std::move(it->second);
                state->vagAccum.erase(it);
            }
        }

        if (completedVag.data.size() >= 48u)
        {
            const uint32_t magic =
                (static_cast<uint32_t>(
                     completedVag.data[0])
                 << 24) |
                (static_cast<uint32_t>(
                     completedVag.data[1])
                 << 16) |
                (static_cast<uint32_t>(
                     completedVag.data[2])
                 << 8) |
                static_cast<uint32_t>(
                    completedVag.data[3]);
            const uint32_t magicLE =
                (static_cast<uint32_t>(
                     completedVag.data[3])
                 << 24) |
                (static_cast<uint32_t>(
                     completedVag.data[2])
                 << 16) |
                (static_cast<uint32_t>(
                     completedVag.data[1])
                 << 8) |
                static_cast<uint32_t>(
                    completedVag.data[0]);
            if (magic == 0x56414770u ||
                magicLE == 0x56414770u)
            {
                runtime->audioBackend()
                    .onVagTransferFromBuffer(
                        completedVag.data.data(),
                        static_cast<uint32_t>(
                            completedVag.data.size()),
                        completedVag.firstBufAddr
                            ? completedVag.firstBufAddr
                            : 0u);
            }
        }

        setReturnS32(ctx, ret == 0 ? 0 : -1);
    }

    void fioRead(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        EeFileRuntimeState *const state =
            getFileRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, -1);
            return;
        }

        int ps2Fd = (int)getRegU32(ctx, 4);   // $a0
        uint32_t bufAddr = getRegU32(ctx, 5); // $a1
        size_t size = getRegU32(ctx, 6);      // $a2

        uint8_t *hostBuf = getMemPtr(rdram, bufAddr);

        if (!hostBuf)
        {
            std::cerr << "fioRead error: Invalid buffer address for fd " << ps2Fd << std::endl;
            setReturnS32(ctx, -1); // -EFAULT
            return;
        }
        size_t bytesRead = 0;
        bool readError = false;
        {
            std::lock_guard<std::mutex> lock(
                state->descriptorMutex);
            const auto it =
                state->descriptors.find(ps2Fd);
            if (it == state->descriptors.end() ||
                !it->second)
            {
                std::cerr
                    << "fioRead error: Invalid file descriptor "
                    << ps2Fd << std::endl;
                setReturnS32(ctx, -1);
                return;
            }
            if (size == 0u)
            {
                setReturnS32(ctx, 0);
                return;
            }

            bytesRead =
                std::fread(
                    hostBuf, 1u, size, it->second);
            readError =
                bytesRead < size &&
                std::ferror(it->second);
            if (readError)
            {
                std::clearerr(it->second);
            }
        }
        if (bytesRead > 0)
        {
            ps2TraceGuestRangeWrite(rdram, bufAddr, static_cast<uint32_t>(bytesRead), "fioRead", ctx);
        }

        if (readError)
        {
            std::cerr << "fioRead error: fread failed for fd " << ps2Fd << ": " << strerror(errno) << std::endl;
            setReturnS32(ctx, -1);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(
                state->vagMutex);
            auto it = state->vagAccum.find(ps2Fd);
            if (it != state->vagAccum.end())
            {
                EeFileVagAccumEntry &e = it->second;
                if (e.data.size() + bytesRead <= kVagAccumMaxBytes)
                    e.data.insert(e.data.end(), hostBuf, hostBuf + bytesRead);
            }
            else if (bytesRead >= 4)
            {
                const uint32_t magic = (static_cast<uint32_t>(hostBuf[0]) << 24) |
                                       (static_cast<uint32_t>(hostBuf[1]) << 16) |
                                       (static_cast<uint32_t>(hostBuf[2]) << 8) |
                                       static_cast<uint32_t>(hostBuf[3]);
                const uint32_t magicLE = (static_cast<uint32_t>(hostBuf[3]) << 24) |
                                         (static_cast<uint32_t>(hostBuf[2]) << 16) |
                                         (static_cast<uint32_t>(hostBuf[1]) << 8) |
                                         static_cast<uint32_t>(hostBuf[0]);
                if (magic == 0x56414770u || magicLE == 0x56414770u)
                {
                    EeFileVagAccumEntry &e =
                        state->vagAccum[ps2Fd];
                    e.firstBufAddr = bufAddr;
                    if (bytesRead <= kVagAccumMaxBytes)
                        e.data.assign(hostBuf, hostBuf + bytesRead);
                }
            }
        }

        setReturnS32(ctx, (int32_t)bytesRead);
    }

    void fioWrite(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        EeFileRuntimeState *const state =
            getFileRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, -1);
            return;
        }

        int ps2Fd = (int)getRegU32(ctx, 4);   // $a0
        uint32_t bufAddr = getRegU32(ctx, 5); // $a1
        size_t size = getRegU32(ctx, 6);      // $a2

        const uint8_t *hostBuf = getConstMemPtr(rdram, bufAddr);
        if (!hostBuf)
        {
            setReturnS32(ctx, -1);
            return;
        }

        size_t bytesWritten = 0;
        {
            std::lock_guard<std::mutex> lock(
                state->descriptorMutex);
            const auto it =
                state->descriptors.find(ps2Fd);
            if (it == state->descriptors.end() ||
                !it->second)
            {
                setReturnS32(ctx, -1);
                return;
            }
            if (size == 0u)
            {
                setReturnS32(ctx, 0);
                return;
            }

            bytesWritten =
                std::fwrite(
                    hostBuf, 1u, size, it->second);
            if (bytesWritten < size &&
                std::ferror(it->second))
            {
                std::clearerr(it->second);
                setReturnS32(ctx, -1); // -EIO, -ENOSPC etc.
                return;
            }
        }

        // returns number of bytes written
        setReturnS32(ctx, (int32_t)bytesWritten);
    }

    void fioLseek(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        EeFileRuntimeState *const state =
            getFileRuntimeState(runtime);
        if (!state)
        {
            setReturnS32(ctx, -1);
            return;
        }

        int ps2Fd = (int)getRegU32(ctx, 4);  // $a0
        int32_t offset = getRegU32(ctx, 5);  // $a1 (PS2 seems to use 32-bit offset here commonly)
        int whence = (int)getRegU32(ctx, 6); // $a2 (PS2 FIO_SEEK constants)

        int hostWhence;
        switch (whence)
        {
        case PS2_FIO_SEEK_SET:
            hostWhence = SEEK_SET;
            break;
        case PS2_FIO_SEEK_CUR:
            hostWhence = SEEK_CUR;
            break;
        case PS2_FIO_SEEK_END:
            hostWhence = SEEK_END;
            break;
        default:
            std::cerr << "fioLseek error: Invalid whence value " << whence << " for fd " << ps2Fd << std::endl;
            setReturnS32(ctx, -1); // -EINVAL
            return;
        }

        long newPos = -1;
        {
            std::lock_guard<std::mutex> lock(
                state->descriptorMutex);
            const auto it =
                state->descriptors.find(ps2Fd);
            if (it == state->descriptors.end() ||
                !it->second)
            {
                std::cerr
                    << "fioLseek error: Invalid file descriptor "
                    << ps2Fd << std::endl;
                setReturnS32(ctx, -1);
                return;
            }

            if (::fseek(
                    it->second,
                    static_cast<long>(offset),
                    hostWhence) != 0)
            {
                std::cerr
                    << "fioLseek error: fseek failed for fd "
                    << ps2Fd << ": "
                    << strerror(errno) << std::endl;
                setReturnS32(ctx, -1);
                return;
            }

            newPos = ::ftell(it->second);
        }

        if (newPos < 0)
        {
            std::cerr << "fioLseek error: ftell failed after fseek for fd " << ps2Fd << ": " << strerror(errno) << std::endl;
            setReturnS32(ctx, -1);
        }
        else
        {
            if (newPos > 0xFFFFFFFFL)
            {
                std::cerr << "fioLseek warning: New position exceeds 32-bit for fd " << ps2Fd << std::endl;
                setReturnS32(ctx, -1);
            }
            else
            {
                setReturnS32(ctx, (int32_t)newPos);
            }
        }
    }

    void fioMkdir(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t pathAddr = getRegU32(ctx, 4); // $a0
        // int mode = (int)getRegU32(ctx, 5);  // $a1 - ignored on host

        const char *ps2Path = reinterpret_cast<const char *>(getConstMemPtr(rdram, pathAddr));
        if (!ps2Path)
        {
            std::cerr << "fioMkdir error: Invalid path address" << std::endl;
            setReturnS32(ctx, -1); // -EFAULT
            return;
        }
        std::string hostPath =
            translatePs2Path(ps2Path, runtime);
        if (hostPath.empty())
        {
            std::cerr << "fioMkdir error: Failed to translate path '" << ps2Path << "'" << std::endl;
            setReturnS32(ctx, -1);
            return;
        }
        std::error_code ec;
        bool success = std::filesystem::create_directory(hostPath, ec);

        if (!success && ec)
        {
            std::cerr << "fioMkdir error: create_directory failed for '" << hostPath
                      << "': " << ec.message() << std::endl;
            setReturnS32(ctx, -1);
        }
        else
        {
            RUNTIME_LOG("fioMkdir: Created directory '" << hostPath << "'");
            setReturnS32(ctx, 0); // Success
        }
    }

    void fioChdir(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t pathAddr = getRegU32(ctx, 4); // $a0
        const char *ps2Path = reinterpret_cast<const char *>(getConstMemPtr(rdram, pathAddr));
        if (!ps2Path)
        {
            std::cerr << "fioChdir error: Invalid path address" << std::endl;
            setReturnS32(ctx, -1);
            return;
        }

        std::string hostPath =
            translatePs2Path(ps2Path, runtime);
        if (hostPath.empty())
        {
            std::cerr << "fioChdir error: Failed to translate path '" << ps2Path << "'" << std::endl;
            setReturnS32(ctx, -1);
            return;
        }

        std::error_code ec;
        std::filesystem::current_path(hostPath, ec);

        if (ec)
        {
            std::cerr << "fioChdir error: current_path failed for '" << hostPath
                      << "': " << ec.message() << std::endl;
            setReturnS32(ctx, -1);
        }
        else
        {
            RUNTIME_LOG("fioChdir: Changed directory to '" << hostPath << "'");
            setReturnS32(ctx, 0); // Success
        }
    }

    void fioRmdir(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t pathAddr = getRegU32(ctx, 4); // $a0
        const char *ps2Path = reinterpret_cast<const char *>(getConstMemPtr(rdram, pathAddr));
        if (!ps2Path)
        {
            std::cerr << "fioRmdir error: Invalid path address" << std::endl;
            setReturnS32(ctx, -1);
            return;
        }
        std::string hostPath =
            translatePs2Path(ps2Path, runtime);
        if (hostPath.empty())
        {
            std::cerr << "fioRmdir error: Failed to translate path '" << ps2Path << "'" << std::endl;
            setReturnS32(ctx, -1);
            return;
        }

        std::error_code ec;
        bool success = std::filesystem::remove(hostPath, ec);

        if (!success || ec)
        {
            std::cerr << "fioRmdir error: remove failed for '" << hostPath
                      << "': " << ec.message() << std::endl;
            setReturnS32(ctx, -1);
        }
        else
        {
            RUNTIME_LOG("fioRmdir: Removed directory '" << hostPath << "'");
            setReturnS32(ctx, 0); // Success
        }
    }

    void fioGetstat(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        // we wont implement this for now.
        uint32_t pathAddr = getRegU32(ctx, 4);    // $a0
        uint32_t statBufAddr = getRegU32(ctx, 5); // $a1

        const char *ps2Path = reinterpret_cast<const char *>(getConstMemPtr(rdram, pathAddr));
        uint8_t *ps2StatBuf = getMemPtr(rdram, statBufAddr);

        if (!ps2Path)
        {
            std::cerr << "fioGetstat error: Invalid path addr" << std::endl;
            setReturnS32(ctx, -1);
            return;
        }
        if (!ps2StatBuf)
        {
            std::cerr << "fioGetstat error: Invalid buffer addr" << std::endl;
            setReturnS32(ctx, -1);
            return;
        }

        std::string hostPath =
            translatePs2Path(ps2Path, runtime);
        if (hostPath.empty())
        {
            std::cerr << "fioGetstat error: Bad path translate" << std::endl;
            setReturnS32(ctx, -1);
            return;
        }

        setReturnS32(ctx, -1);
    }

    void fioRemove(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t pathAddr = getRegU32(ctx, 4); // $a0
        const char *ps2Path = reinterpret_cast<const char *>(getConstMemPtr(rdram, pathAddr));
        if (!ps2Path)
        {
            std::cerr << "fioRemove error: Invalid path" << std::endl;
            setReturnS32(ctx, -1);
            return;
        }

        std::string hostPath =
            translatePs2Path(ps2Path, runtime);
        if (hostPath.empty())
        {
            std::cerr << "fioRemove error: Path translate fail" << std::endl;
            setReturnS32(ctx, -1);
            return;
        }

        std::error_code ec;
        bool success = std::filesystem::remove(hostPath, ec);

        if (!success || ec)
        {
            std::cerr << "fioRemove error: remove failed for '" << hostPath
                      << "': " << ec.message() << std::endl;
            setReturnS32(ctx, -1);
        }
        else
        {
            RUNTIME_LOG("fioRemove: Removed file '" << hostPath << "'");
            setReturnS32(ctx, 0); // Success
        }
    }
}
