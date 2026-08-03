#include "MiniTest.h"
#include "ps2_runtime.h"

#include <elfio/elfio.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

namespace
{
    constexpr uint32_t kTextAddress = 0x00100000u;
    constexpr uint32_t kDataAddress = 0x00100100u;

    struct ScopedElfFile
    {
        std::filesystem::path path;

        explicit ScopedElfFile(const std::string &name)
        {
            const auto uniqueSuffix =
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                "_" + std::to_string(reinterpret_cast<std::uintptr_t>(this));
            path = std::filesystem::temp_directory_path() /
                   ("ps2recomp-" + name + "-" + uniqueSuffix + ".elf");
        }

        ~ScopedElfFile()
        {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    };

    bool writeMixedReadWriteExecuteElf(const std::filesystem::path &path)
    {
        ELFIO::elfio writer;
        writer.create(ELFIO::ELFCLASS32, ELFIO::ELFDATA2LSB);
        writer.set_os_abi(ELFIO::ELFOSABI_NONE);
        writer.set_type(ELFIO::ET_EXEC);
        writer.set_machine(ELFIO::EM_MIPS);
        writer.set_entry(kTextAddress);

        ELFIO::section *text = writer.sections.add(".text");
        text->set_type(ELFIO::SHT_PROGBITS);
        text->set_flags(ELFIO::SHF_ALLOC | ELFIO::SHF_EXECINSTR);
        text->set_addr_align(4);
        text->set_address(kTextAddress);
        const std::array<uint32_t, 2> textWords = {
            0x03E00008u, // jr $ra
            0x00000000u, // nop
        };
        text->set_data(reinterpret_cast<const char *>(textWords.data()),
                       static_cast<ELFIO::Elf_Word>(sizeof(textWords)));

        ELFIO::section *data = writer.sections.add(".data");
        data->set_type(ELFIO::SHT_PROGBITS);
        data->set_flags(ELFIO::SHF_ALLOC | ELFIO::SHF_WRITE);
        data->set_addr_align(4);
        data->set_address(kDataAddress);
        const std::array<uint32_t, 2> dataWords = {
            0x11223344u,
            0x55667788u,
        };
        data->set_data(reinterpret_cast<const char *>(dataWords.data()),
                       static_cast<ELFIO::Elf_Word>(sizeof(dataWords)));

        ELFIO::segment *mixedSegment = writer.segments.add();
        mixedSegment->set_type(ELFIO::PT_LOAD);
        mixedSegment->set_flags(ELFIO::PF_R | ELFIO::PF_W | ELFIO::PF_X);
        mixedSegment->set_align(0x1000);
        mixedSegment->set_virtual_address(kTextAddress);
        mixedSegment->set_physical_address(kTextAddress);
        mixedSegment->add_section_index(text->get_index(), text->get_addr_align());
        mixedSegment->add_section_index(data->get_index(), data->get_addr_align());

        return writer.save(path.string());
    }

    template <typename T>
    bool writeElfHeaderField(std::fstream &file, std::streamoff offset, T value)
    {
        file.seekp(offset, std::ios::beg);
        file.write(reinterpret_cast<const char *>(&value), sizeof(value));
        return file.good();
    }

    bool removeSectionTableFromElf(const std::filesystem::path &path)
    {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        if (!file)
        {
            return false;
        }

        // ELF32 e_shoff, e_shentsize, e_shnum, and e_shstrndx.
        return writeElfHeaderField<uint32_t>(file, 32, 0u) &&
               writeElfHeaderField<uint16_t>(file, 46, 0u) &&
               writeElfHeaderField<uint16_t>(file, 48, 0u) &&
               writeElfHeaderField<uint16_t>(file, 50, 0u);
    }

    bool initializeHeadlessRuntime(PS2Runtime &runtime)
    {
        return runtime.memory().initialize() && runtime.syncCoreSubsystems();
    }
}

void register_ps2_runtime_elf_tests()
{
    MiniTest::Case("PS2RuntimeELF", [](TestCase &tc)
    {
        tc.Run("direct ELF load recreates the BIOS TLB handoff", [](TestCase &t)
        {
            ScopedElfFile elf("post-bios-tlb");
            const bool wroteElf =
                writeMixedReadWriteExecuteElf(elf.path);
            t.IsTrue(
                wroteElf,
                "synthetic direct ELF should be generated");
            if (!wroteElf)
            {
                return;
            }

            PS2Runtime runtime;
            const bool initialized =
                initializeHeadlessRuntime(runtime);
            t.IsTrue(
                initialized,
                "headless runtime should initialize");
            if (!initialized)
            {
                return;
            }

            const bool loaded =
                runtime.loadELF(elf.path.string());
            t.IsTrue(
                loaded,
                "synthetic direct ELF should load");
            if (!loaded)
            {
                return;
            }
            t.Equals(
                runtime.cpu().cop0_wired,
                31u,
                "the direct ELF profile should retain the BIOS Wired boundary");
            t.Equals(
                runtime.cpu().cop0_index,
                38u,
                "the direct ELF profile should retain the current BIOS TLB index");
            t.Equals(
                runtime.cpu().cop0_pagemask,
                0x007fe000u,
                "the direct ELF profile should retain the current PageMask");
            t.Equals(
                runtime.cpu().cop0_entryhi,
                0x31800000u,
                "the direct ELF profile should retain the current EntryHi");
            t.Equals(
                runtime.cpu().cop0_entrylo0,
                0x0006003fu,
                "the direct ELF profile should retain the current EntryLo0");
            t.Equals(
                runtime.cpu().cop0_entrylo1,
                0x0007003fu,
                "the direct ELF profile should retain the current EntryLo1");

            EeTlbEntry entry{};
            t.IsTrue(
                runtime.memory().tlbRead(14u, entry),
                "the BIOS main-RAM entry should be readable");
            t.Equals(
                entry.pageMask,
                0x0007e000u,
                "the 1 MiB identity pair should retain its page size");
            t.Equals(
                entry.entryHi,
                0x00100000u,
                "the identity pair should retain its virtual base");
            t.Equals(
                entry.entryLo0,
                0x0000401fu,
                "the identity pair should retain its even PFN and attributes");
            t.Equals(
                entry.entryLo1,
                0x0000501fu,
                "the identity pair should retain its odd PFN and attributes");

            bool translated = false;
            uint32_t physicalAddress = 0u;
            try
            {
                physicalAddress =
                    runtime.memory().translateAddress(
                        0x00123456u,
                        EeAddressTranslationContext::fromCop0Status(
                            runtime.cpu().cop0_status,
                            static_cast<uint8_t>(
                                runtime.cpu().cop0_entryhi)));
                translated = true;
            }
            catch (const PS2TlbFaultException &)
            {
            }
            t.IsTrue(
                translated,
                "the BIOS handoff should map the direct ELF useg entry");
            t.Equals(
                physicalAddress,
                0x00123456u,
                "the BIOS handoff should identity-map ordinary main RAM");
        });

        tc.Run("mixed RWE segments track only executable sections", [](TestCase &t)
        {
            ScopedElfFile elf("mixed-rwe-sections");
            const bool wroteElf = writeMixedReadWriteExecuteElf(elf.path);
            t.IsTrue(wroteElf, "synthetic mixed RWE ELF should be generated");
            if (!wroteElf)
            {
                return;
            }

            PS2Runtime runtime;
            const bool initialized = initializeHeadlessRuntime(runtime);
            t.IsTrue(initialized, "headless runtime should initialize");
            if (!initialized)
            {
                return;
            }

            const bool loaded = runtime.loadELF(elf.path.string());
            t.IsTrue(loaded, "synthetic mixed RWE ELF should load");
            if (!loaded)
            {
                return;
            }

            t.IsTrue(runtime.memory().isCodeAddress(kTextAddress),
                     "allocated executable section should be tracked as code");
            t.IsTrue(runtime.memory().isCodeAddress(kTextAddress + 7u),
                     "the complete executable section should be tracked as code");
            t.IsFalse(runtime.memory().isCodeAddress(kTextAddress + 8u),
                      "padding in the mixed segment should not be tracked as code");
            t.IsFalse(runtime.memory().isCodeAddress(kDataAddress),
                      "writable data in the mixed segment should not be tracked as code");

            runtime.memory().write32(kDataAddress, 0xAABBCCDDu);
            t.IsFalse(runtime.memory().isCodeModified(kDataAddress, sizeof(uint32_t)),
                      "writes to data in a mixed segment should not mark code modified");

            runtime.memory().write32(kTextAddress, 0u);
            t.IsTrue(runtime.memory().isCodeModified(kTextAddress, sizeof(uint32_t)),
                     "writes to an executable section should still mark code modified");
        });

        tc.Run("sectionless ELFs fall back to executable segments", [](TestCase &t)
        {
            ScopedElfFile elf("sectionless-rwe-fallback");
            const bool wroteElf = writeMixedReadWriteExecuteElf(elf.path);
            t.IsTrue(wroteElf, "synthetic mixed RWE ELF should be generated");
            if (!wroteElf)
            {
                return;
            }

            const bool removedSectionTable = removeSectionTableFromElf(elf.path);
            t.IsTrue(removedSectionTable, "synthetic ELF section table should be removed");
            if (!removedSectionTable)
            {
                return;
            }

            PS2Runtime runtime;
            const bool initialized = initializeHeadlessRuntime(runtime);
            t.IsTrue(initialized, "headless runtime should initialize");
            if (!initialized)
            {
                return;
            }

            const bool loaded = runtime.loadELF(elf.path.string());
            t.IsTrue(loaded, "sectionless ELF should load");
            if (!loaded)
            {
                return;
            }

            t.IsTrue(runtime.memory().isCodeAddress(kTextAddress),
                     "sectionless ELF text should use the executable-segment fallback");
            t.IsTrue(runtime.memory().isCodeAddress(kDataAddress),
                     "the conservative fallback should retain the full executable segment");
        });
    });
}
