cmake_minimum_required(VERSION 3.20)

project(PS2Runtime VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PS2X_ENABLE_RUNNER_UNITY_BUILD "Build ps2EntryRunner with CMake unity build" ON)
set(PS2X_RUNNER_UNITY_BUILD_BATCH_SIZE 32 CACHE STRING "Unity build batch size for ps2EntryRunner")
set(PS2X_RUNNER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/runner" CACHE PATH
    "Directory containing generated ps2EntryRunner C++ sources")
set(PS2X_RUNNER_MODULE_SOURCE_DIRS "" CACHE STRING
    "Semicolon-separated generated fixed-address AOT module source directories")
option(PS2X_ENABLE_RUNNER_PCH "Precompile the heavy runtime headers for ps2EntryRunner" ON)
option(PS2X_ENABLE_SCCACHE "Use sccache as compiler launcher when available" ON)

option(PS2X_ENABLE_RUNTIME_LOGS "Enable PS2 runtime logs" OFF)
option(PS2X_ENABLE_AGRESSIVE_LOGS "Enable very verbose/agressive PS2 runtime logs" OFF)
option(PS2X_ENABLE_IOP_RPC_TRACE "Log unhandled IOP/SIF RPC trace suggestions" ON)
option(PS2X_STRICT_RETURN_DIAGNOSTICS "Route generated JR $ra returns through runtime branch diagnostics" OFF)
option(PS2X_SHOW_WINDOWS_CONSOLE "Show a console window for ps2EntryRunner on Windows release builds" ON)
option(PS2X_ENABLE_VU_X64_RECOMPILER
    "Build the production x86-64 VU dynamic recompiler when supported" ON)
option(PS2X_ENABLE_GS_AVX2_KERNEL
    "Build the runtime-dispatched AVX2 GS packed-sprite kernel" ON)
option(PS2X_ENABLE_GS_VULKAN
    "Build the optional dynamically loaded desktop Vulkan GS backend" ON)
option(PS2X_ENABLE_EE_CPP_FIBER_BACKEND
    "Build the legacy generated-C++ Boost.Context fiber backend when supported"
    ON)
set(PS2X_DEFAULT_EE_EXECUTION_BACKEND
    "legacy-host-thread"
    CACHE STRING
    "Default EE execution backend: legacy-host-thread or legacy-cpp-fiber")
set_property(
    CACHE PS2X_DEFAULT_EE_EXECUTION_BACKEND
    PROPERTY STRINGS
        legacy-host-thread
        legacy-cpp-fiber)

set(PS2X_EE_EXECUTION_BACKENDS
    legacy-host-thread
    legacy-cpp-fiber)
if(NOT PS2X_DEFAULT_EE_EXECUTION_BACKEND IN_LIST
   PS2X_EE_EXECUTION_BACKENDS)
    message(FATAL_ERROR
        "Unknown PS2X_DEFAULT_EE_EXECUTION_BACKEND="
        "${PS2X_DEFAULT_EE_EXECUTION_BACKEND}; expected "
        "legacy-host-thread or legacy-cpp-fiber")
endif()

if(PS2X_ENABLE_SCCACHE)
    find_program(PS2X_SCCACHE_PROGRAM sccache)

    if(PS2X_SCCACHE_PROGRAM)
        if(CMAKE_C_COMPILER)
            set(CMAKE_C_COMPILER_LAUNCHER "${PS2X_SCCACHE_PROGRAM}")
        endif()

        set(CMAKE_CXX_COMPILER_LAUNCHER "${PS2X_SCCACHE_PROGRAM}")
        message(STATUS "Using sccache: ${PS2X_SCCACHE_PROGRAM}")
    else()
        message(STATUS "sccache not found; continuing without compiler launcher")
    endif()
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ReleaseMode.cmake")

set(PS2X_IS_VITA OFF)

if((CMAKE_C_COMPILER MATCHES "arm-vita-eabi") OR
    (CMAKE_CXX_COMPILER MATCHES "arm-vita-eabi"))
    set(PS2X_IS_VITA ON)
endif()

set(PS2X_IS_ANDROID OFF)

if(ANDROID)
    set(PS2X_IS_ANDROID ON)
endif()

set(PS2X_HAS_GS_VULKAN OFF)
if(PS2X_ENABLE_GS_VULKAN AND
   NOT PS2X_IS_VITA AND
   NOT PS2X_IS_ANDROID)
    find_path(PS2X_GS_VULKAN_INCLUDE_DIR
        NAMES vulkan/vulkan.h)
    mark_as_advanced(PS2X_GS_VULKAN_INCLUDE_DIR)
    if(PS2X_GS_VULKAN_INCLUDE_DIR)
        set(PS2X_HAS_GS_VULKAN ON)
        message(STATUS
            "GS Vulkan capability path enabled: ${PS2X_GS_VULKAN_INCLUDE_DIR}")
    else()
        message(STATUS
            "Vulkan headers not found; GS Vulkan capability path disabled")
    endif()
elseif(PS2X_ENABLE_GS_VULKAN)
    message(STATUS
        "GS Vulkan capability path is currently desktop-only; using software GS")
endif()

set(PS2X_HAS_VU_X64_RECOMPILER OFF)
if(PS2X_ENABLE_VU_X64_RECOMPILER
   AND NOT PS2X_IS_VITA
   AND NOT PS2X_IS_ANDROID
   AND CMAKE_SIZEOF_VOID_P EQUAL 8
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(PS2X_HAS_VU_X64_RECOMPILER ON)
endif()

set(PS2X_DEBUG_SERVER_DEFAULT OFF)
if(UNIX AND NOT APPLE AND NOT PS2X_IS_VITA AND NOT PS2X_IS_ANDROID)
    set(PS2X_DEBUG_SERVER_DEFAULT ON)
endif()
option(PS2X_ENABLE_DEBUG_SERVER
    "Build the local ps2dbg JSON-RPC control server"
    ${PS2X_DEBUG_SERVER_DEFAULT})
option(PS2X_ENABLE_DEBUG_UI
    "Build the embedded ImGui runtime debugger panel"
    OFF)

set(PS2X_EE_FCONTEXT_PLATFORM_SUPPORTED OFF)
if(CMAKE_SIZEOF_VOID_P EQUAL 8
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$"
   AND (CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
        (WIN32 AND MSVC)))
    set(PS2X_EE_FCONTEXT_PLATFORM_SUPPORTED ON)
endif()
set(PS2X_HAS_EE_CPP_FIBER_BACKEND OFF)

if(NOT DEFINED PS2X_EE_CONTEXT_BUILD_MODE)
    if(PS2X_ENABLE_EE_CPP_FIBER_BACKEND
       AND PS2X_EE_FCONTEXT_PLATFORM_SUPPORTED)
        set(PS2X_EE_CONTEXT_BUILD_MODE "production-fcontext")
    else()
        set(PS2X_EE_CONTEXT_BUILD_MODE "host-fallback")
    endif()
endif()
set(PS2X_EE_CONTEXT_BUILD_MODE
    "${PS2X_EE_CONTEXT_BUILD_MODE}"
    CACHE STRING
    "EE continuation contract: production-fcontext, host-fallback, or sanitizer-no-fcontext")
set_property(
    CACHE PS2X_EE_CONTEXT_BUILD_MODE
    PROPERTY STRINGS
        production-fcontext
        host-fallback
        sanitizer-no-fcontext)

set(PS2X_EE_CONTEXT_BUILD_MODES
    production-fcontext
    host-fallback
    sanitizer-no-fcontext)
if(NOT PS2X_EE_CONTEXT_BUILD_MODE IN_LIST
   PS2X_EE_CONTEXT_BUILD_MODES)
    message(FATAL_ERROR
        "Unknown PS2X_EE_CONTEXT_BUILD_MODE="
        "${PS2X_EE_CONTEXT_BUILD_MODE}; expected production-fcontext, "
        "host-fallback, or sanitizer-no-fcontext")
endif()

if(PS2X_EE_CONTEXT_BUILD_MODE STREQUAL
   "production-fcontext")
    if(NOT PS2X_EE_FCONTEXT_PLATFORM_SUPPORTED)
        message(FATAL_ERROR
            "production-fcontext requires Linux x86-64 or Windows x64 MSVC")
    endif()
    if(NOT PS2X_ENABLE_EE_CPP_FIBER_BACKEND)
        message(FATAL_ERROR
            "production-fcontext requires "
            "PS2X_ENABLE_EE_CPP_FIBER_BACKEND=ON")
    endif()
elseif(PS2X_EE_CONTEXT_BUILD_MODE STREQUAL
       "sanitizer-no-fcontext")
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR
            "sanitizer-no-fcontext is supported only on Linux; "
            "Windows MSVC ASan with suspended Boost fcontext frames and "
            "Windows TSan are unsupported project configurations")
    endif()
    if(PS2X_ENABLE_EE_CPP_FIBER_BACKEND)
        message(FATAL_ERROR
            "sanitizer-no-fcontext requires "
            "PS2X_ENABLE_EE_CPP_FIBER_BACKEND=OFF; "
            "the context implementation is never changed silently")
    endif()
elseif(PS2X_EE_FCONTEXT_PLATFORM_SUPPORTED
       AND PS2X_ENABLE_EE_CPP_FIBER_BACKEND)
    message(FATAL_ERROR
        "host-fallback on a supported fcontext platform requires "
        "PS2X_ENABLE_EE_CPP_FIBER_BACKEND=OFF")
endif()

message(STATUS
    "PS2X EE context build mode: "
    "${PS2X_EE_CONTEXT_BUILD_MODE}")

option(PS2X_VITA_CREATE_PACKAGE "Create Vita eboot/vpk targets" ON)
set(PS2X_VITA_APP_NAME "PS2 Retro X" CACHE STRING "Display name for the Vita bubble")
set(PS2X_VITA_TITLEID "RANJ00001" CACHE STRING "9-character Vita title id")
set(PS2X_VITA_VERSION "01.00" CACHE STRING "Vita app version")
set(PS2X_DEFAULT_BOOT_ELF "" CACHE STRING "Guest ELF path passed directly to main() when argv is unavailable")

if(PS2X_IS_VITA)
    if(NOT DEFINED ENV{VITASDK})
        message(FATAL_ERROR "VITASDK is not defined. Configure inside the VitaSDK environment or pass the Vita toolchain file.")
    endif()

    set(PS2X_VITASDK "$ENV{VITASDK}")

    if(NOT EXISTS "${PS2X_VITASDK}/share/vita.cmake")
        message(FATAL_ERROR "Could not find vita.cmake under ${PS2X_VITASDK}/share.")
    endif()

    include("${PS2X_VITASDK}/share/vita.cmake" REQUIRED)
else()
    include(FetchContent)
    set(FETCHCONTENT_QUIET FALSE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_GAMES OFF CACHE BOOL "" FORCE)

    if(PS2X_ENABLE_EE_CPP_FIBER_BACKEND
       AND PS2X_EE_FCONTEXT_PLATFORM_SUPPORTED)
        # Use the exact official Boost release archive. In particular, do not
        # resolve Boost.Context through an unversioned system installation.
        set(BOOST_INCLUDE_LIBRARIES context)
        set(BOOST_CONTEXT_IMPLEMENTATION
            fcontext
            CACHE STRING
            "Boost.Context implementation used by the EE fiber backend"
            FORCE)
        FetchContent_Declare(
            Boost
            URL
                https://archives.boost.io/release/1.91.0/source/boost_1_91_0.tar.gz
            URL_HASH
                SHA256=5734305f40a76c30f951c9abd409a45a2a19fb546efe4162119250bbe4d3a463
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(Boost)

        if(NOT TARGET Boost::context)
            include(
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/BoostContextReleaseArchive.cmake")
            ps2x_add_boost_context_from_release_archive(
                "${boost_SOURCE_DIR}")
        endif()
        if(NOT BOOST_CONTEXT_IMPLEMENTATION STREQUAL "fcontext")
            message(FATAL_ERROR
                "Production EE fibers require Boost.Context fcontext")
        endif()

        set(PS2X_HAS_EE_CPP_FIBER_BACKEND ON)
        message(STATUS
            "PS2X EE fiber backend: Boost 1.91.0, "
            "architecture ${BOOST_CONTEXT_ARCHITECTURE}, "
            "binary format ${BOOST_CONTEXT_BINARY_FORMAT}, "
            "ABI ${BOOST_CONTEXT_ABI}, "
            "implementation ${BOOST_CONTEXT_IMPLEMENTATION}")
    elseif(PS2X_ENABLE_EE_CPP_FIBER_BACKEND)
        message(STATUS
            "PS2X EE fiber backend unavailable on "
            "${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}; "
            "retaining legacy-host-thread")
    else()
        message(STATUS
            "PS2X EE fiber backend disabled; "
            "retaining legacy-host-thread")
    endif()

    if(PS2X_HAS_VU_X64_RECOMPILER)
        FetchContent_Declare(
            xbyak
            GIT_REPOSITORY https://github.com/herumi/xbyak.git
            # Xbyak 7.37, byte-identical to the accepted PCSX2 snapshot.
            GIT_TAG 431abd865e70a46d56f5aa0e1f87572decb60169
        )
        FetchContent_MakeAvailable(xbyak)
    endif()

    if(PS2X_IS_ANDROID)
        set(PLATFORM "Android" CACHE STRING "" FORCE)
    endif()

    FetchContent_Declare(
        raylib
        GIT_REPOSITORY "https://github.com/raysan5/raylib.git"
        GIT_TAG "5.5" # we will migrate to 4.2.0 later, trust me it will be better
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(raylib)

    if(PS2X_ENABLE_DEBUG_SERVER)
        FetchContent_Declare(
            rapidjson
            GIT_REPOSITORY https://github.com/Tencent/rapidjson.git
            # Pinned post-1.1 revision with current C++20/GCC compatibility.
            GIT_TAG 24b5e7a8b27f42fa16b96fc70aade9106cf7102f
            GIT_SHALLOW TRUE
        )
        FetchContent_GetProperties(rapidjson)
        if(NOT rapidjson_POPULATED)
            FetchContent_Populate(rapidjson)
        endif()
    endif()

    if(NOT PS2X_IS_ANDROID AND PS2X_ENABLE_DEBUG_UI)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG "docking"
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(imgui)

    if(NOT imgui_POPULATED)
        FetchContent_Populate(imgui)
    endif()

    set(PS2X_IMGUI_SOURCE_DIR "${imgui_SOURCE_DIR}")

    add_library(imgui STATIC
        "${PS2X_IMGUI_SOURCE_DIR}/imgui.cpp"
        "${PS2X_IMGUI_SOURCE_DIR}/imgui_draw.cpp"
        "${PS2X_IMGUI_SOURCE_DIR}/imgui_tables.cpp"
        "${PS2X_IMGUI_SOURCE_DIR}/imgui_widgets.cpp"
        "${PS2X_IMGUI_SOURCE_DIR}/imgui_demo.cpp"
    )
    target_include_directories(imgui PUBLIC "${PS2X_IMGUI_SOURCE_DIR}")

    FetchContent_Declare(
        rlImGui
        GIT_REPOSITORY https://github.com/raylib-extras/rlImGui.git
        GIT_TAG "Raylib_5_5"
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(rlImGui)

    if(NOT rlimgui_POPULATED)
        FetchContent_Populate(rlImGui)
    endif()

    set(PS2X_RLIMGUI_SOURCE_DIR "${rlimgui_SOURCE_DIR}")

    add_library(rlImGui STATIC
        "${PS2X_RLIMGUI_SOURCE_DIR}/rlImGui.cpp"
    )
    target_include_directories(rlImGui PUBLIC "${PS2X_RLIMGUI_SOURCE_DIR}")
    target_link_libraries(rlImGui PUBLIC raylib imgui)
    endif() # NOT PS2X_IS_ANDROID AND PS2X_ENABLE_DEBUG_UI
endif()

add_library(ps2_host_backend INTERFACE)

set(PS2X_DEFAULT_BOOT_ELF_DEFINE "")

if(PS2X_DEFAULT_BOOT_ELF)
    set(PS2X_DEFAULT_BOOT_ELF_DEFINE "${PS2X_DEFAULT_BOOT_ELF}")
    string(REPLACE "\\" "\\\\" PS2X_DEFAULT_BOOT_ELF_DEFINE "${PS2X_DEFAULT_BOOT_ELF_DEFINE}")
    string(REPLACE "\"" "\\\"" PS2X_DEFAULT_BOOT_ELF_DEFINE "${PS2X_DEFAULT_BOOT_ELF_DEFINE}")
endif()

if(PS2X_IS_VITA)
    set(PS2X_VITA_PREFIX "${PS2X_VITASDK}/arm-vita-eabi")
    set(PS2X_VITA_EXTRA_INCLUDE_DIRS "" CACHE STRING "Extra include directories for Vita host dependencies")
    set(PS2X_VITA_LIBRARY_DIR "${PS2X_VITA_PREFIX}/lib")
    set(PS2X_VITA_REQUIRED_LIBRARY_NAMES
        raylib
        SDL2
        OpenSLES
        taihen_stub
        SceAppMgr_stub
        SceCtrl_stub
        SceGxm_stub
        SceCommonDialog_stub
        SceLibKernel_stub
        SceAudio_stub
        SceTouch_stub
        SceHid_stub
        SceMotion_stub
        SceSysmodule_stub
        SceIofilemgr_stub
        SceNetCtl_stub
        SceNet_stub
        SceDisplay_stub
        SceAppUtil_stub
        SceAudioIn_stub
        ScePower_stub
        SceProcessmgr_stub
        SceIme_stub
        libIMGEGL_stub_weak
        libgpu_es4_ext_stub_weak
        libGLESv2_stub_weak
        libGL_stub_weak
        libGLESv1_CM_stub_weak
    )

    set(PS2X_VITA_INCLUDE_DIRS
        "${PS2X_VITA_PREFIX}/include"
    )

    foreach(PS2X_VITA_OPTIONAL_INCLUDE_DIR IN ITEMS
        "${PS2X_VITA_PREFIX}/include/raylib"
        "${PS2X_VITA_PREFIX}/include/SDL2")
        if(EXISTS "${PS2X_VITA_OPTIONAL_INCLUDE_DIR}")
            list(APPEND PS2X_VITA_INCLUDE_DIRS "${PS2X_VITA_OPTIONAL_INCLUDE_DIR}")
        endif()
    endforeach()

    if(PS2X_VITA_EXTRA_INCLUDE_DIRS)
        list(APPEND PS2X_VITA_INCLUDE_DIRS ${PS2X_VITA_EXTRA_INCLUDE_DIRS})
    endif()

    if(NOT EXISTS "${PS2X_VITA_PREFIX}/include/SDL2/SDL.h")
        message(FATAL_ERROR
            "Missing SDL2 headers under ${PS2X_VITA_PREFIX}/include/SDL2. "
            "Quenom/raylib-5.5-vita requires SDL2 with PVR support installed into VitaSDK.")
    endif()

    if(NOT EXISTS "${PS2X_VITA_LIBRARY_DIR}")
        message(FATAL_ERROR "Missing Vita library directory: ${PS2X_VITA_LIBRARY_DIR}")
    endif()

    function(ps2x_find_vita_library out_var library_name)
        string(MAKE_C_IDENTIFIER "${library_name}" library_id)
        set(cache_var "PS2X_VITA_LIB_${library_id}")

        find_library(${cache_var}
            NAMES "${library_name}" "${library_name}.a" "lib${library_name}.a"
            PATHS "${PS2X_VITA_LIBRARY_DIR}"
            NO_DEFAULT_PATH
        )

        if(NOT ${cache_var})
            message(FATAL_ERROR
                "Could not find Vita library '${library_name}' under ${PS2X_VITA_LIBRARY_DIR}. "
                "Install Quenom/raylib-5.5-vita and its SDL2/PVR dependencies into VitaSDK first.")
        endif()

        set(${out_var} "${${cache_var}}" PARENT_SCOPE)
    endfunction()

    set(PS2X_VITA_RESOLVED_LIBRARIES "")

    foreach(PS2X_VITA_LIBRARY_NAME IN LISTS PS2X_VITA_REQUIRED_LIBRARY_NAMES)
        ps2x_find_vita_library(PS2X_VITA_LIBRARY_PATH "${PS2X_VITA_LIBRARY_NAME}")
        list(APPEND PS2X_VITA_RESOLVED_LIBRARIES "${PS2X_VITA_LIBRARY_PATH}")
    endforeach()

    target_compile_definitions(ps2_host_backend INTERFACE PLATFORM_VITA)
    target_include_directories(ps2_host_backend INTERFACE ${PS2X_VITA_INCLUDE_DIRS})
    target_link_libraries(ps2_host_backend INTERFACE
        ${PS2X_VITA_RESOLVED_LIBRARIES}
        m
        c
        pthread
    )
else()
    target_link_libraries(ps2_host_backend INTERFACE raylib)

    if(PS2X_IS_ANDROID)
        target_link_libraries(ps2_host_backend INTERFACE log android)
    endif()
endif()

if(PS2X_IS_ANDROID)
    set(PS2X_ENABLE_FFMPEG_DEFAULT OFF)
else()
    set(PS2X_ENABLE_FFMPEG_DEFAULT ON)
endif()

option(PS2X_ENABLE_FFMPEG "Enable FFmpeg-backed MPEG video decoding" ${PS2X_ENABLE_FFMPEG_DEFAULT})

if(NOT PS2X_ENABLE_FFMPEG)
    message(STATUS "FFmpeg disabled; MPEG video decode falls back to stub frames")
    add_library(ffmpeg INTERFACE)
elseif(WIN32)
    include(ExternalProject)

    set(FFMPEG_PREBUILT_TAG "n7.1-241205")
    set(FFMPEG_PREBUILT_URL
        "https://github.com/System233/ffmpeg-msvc-prebuilt/releases/download/${FFMPEG_PREBUILT_TAG}/ffmpeg-${FFMPEG_PREBUILT_TAG}-lgpl-amd64-shared.zip")

    set(FFMPEG_PREFIX_DIR "${CMAKE_BINARY_DIR}/ThirdParty/ffmpeg-prefix")
    set(FFMPEG_SOURCE_DIR "${FFMPEG_PREFIX_DIR}/src/ffmpeg_external")
    set(FFMPEG_INCLUDE_DIR "${FFMPEG_SOURCE_DIR}/include")
    set(FFMPEG_LIB_DIR "${FFMPEG_SOURCE_DIR}/bin")
    set(FFMPEG_BIN_DIR "${FFMPEG_SOURCE_DIR}/bin")

    ExternalProject_Add(ffmpeg_external
        URL "${FFMPEG_PREBUILT_URL}"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PREFIX "${FFMPEG_PREFIX_DIR}"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        BUILD_BYPRODUCTS
        "${FFMPEG_LIB_DIR}/avcodec.lib"
        "${FFMPEG_LIB_DIR}/avformat.lib"
        "${FFMPEG_LIB_DIR}/avutil.lib"
        "${FFMPEG_LIB_DIR}/swresample.lib"
        "${FFMPEG_LIB_DIR}/swscale.lib"
    )

    # Link against the import libraries shipped with the Windows DLLs.
    add_library(ffmpeg_avcodec STATIC IMPORTED GLOBAL)
    set_target_properties(ffmpeg_avcodec PROPERTIES
        IMPORTED_LOCATION "${FFMPEG_LIB_DIR}/avcodec.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIR}"
    )
    add_dependencies(ffmpeg_avcodec ffmpeg_external)

    add_library(ffmpeg_avformat STATIC IMPORTED GLOBAL)
    set_target_properties(ffmpeg_avformat PROPERTIES
        IMPORTED_LOCATION "${FFMPEG_LIB_DIR}/avformat.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIR}"
    )
    add_dependencies(ffmpeg_avformat ffmpeg_external)

    add_library(ffmpeg_avutil STATIC IMPORTED GLOBAL)
    set_target_properties(ffmpeg_avutil PROPERTIES
        IMPORTED_LOCATION "${FFMPEG_LIB_DIR}/avutil.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIR}"
    )
    add_dependencies(ffmpeg_avutil ffmpeg_external)

    add_library(ffmpeg_swresample STATIC IMPORTED GLOBAL)
    set_target_properties(ffmpeg_swresample PROPERTIES
        IMPORTED_LOCATION "${FFMPEG_LIB_DIR}/swresample.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIR}"
    )
    add_dependencies(ffmpeg_swresample ffmpeg_external)

    add_library(ffmpeg_swscale STATIC IMPORTED GLOBAL)
    set_target_properties(ffmpeg_swscale PROPERTIES
        IMPORTED_LOCATION "${FFMPEG_LIB_DIR}/swscale.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIR}"
    )
    add_dependencies(ffmpeg_swscale ffmpeg_external)

    add_library(ffmpeg INTERFACE)
    target_link_libraries(ffmpeg INTERFACE
        ffmpeg_avcodec
        ffmpeg_avformat
        ffmpeg_avutil
        ffmpeg_swresample
        ffmpeg_swscale
        bcrypt
        secur32
        ws2_32
        user32
        advapi32
        ole32
        shell32
    )

    set(PS2X_FFMPEG_BIN_DIR "${FFMPEG_BIN_DIR}" CACHE INTERNAL "Directory containing FFmpeg runtime DLLs")
    set(PS2X_COPY_FFMPEG_DLLS_SCRIPT
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/CopyFfmpegDlls.cmake"
        CACHE INTERNAL
        "Script used to stage FFmpeg runtime DLLs beside executables")
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET GLOBAL
        libavcodec
        libavformat
        libavutil
        libswresample
        libswscale
    )

    add_library(ffmpeg INTERFACE)
    target_link_libraries(ffmpeg INTERFACE PkgConfig::FFMPEG)
endif()

function(ps2x_stage_ffmpeg_runtime_dlls target_name)
    if(NOT WIN32)
        return()
    endif()

    if(NOT PS2X_FFMPEG_BIN_DIR OR NOT PS2X_COPY_FFMPEG_DLLS_SCRIPT)
        return()
    endif()

    if(TARGET ffmpeg_external)
        add_dependencies(${target_name} ffmpeg_external)
    endif()

    add_custom_command(
        TARGET ${target_name} POST_BUILD
        COMMAND "${CMAKE_COMMAND}"
        "-Dsource_dir=${PS2X_FFMPEG_BIN_DIR}"
        "-Ddest_dir=$<TARGET_FILE_DIR:${target_name}>"
        -P "${PS2X_COPY_FFMPEG_DLLS_SCRIPT}"
        VERBATIM
    )
endfunction()

add_library(ps2_runtime STATIC
    src/lib/boost_ee_fiber.cpp
    src/lib/cop0_timing.cpp
    src/lib/ee_cache.cpp
    src/lib/ee_counters.cpp
    src/lib/ee_execution_backend.cpp
    src/lib/ee_event_scheduler.cpp
    src/lib/ee_runtime_executor.cpp
    src/lib/ee_scheduler_executor.cpp
    src/lib/ee_thread_scheduler.cpp
    src/lib/game_overrides.cpp
    src/lib/ps2_gif_arbiter.cpp
    src/lib/ps2_989snd.cpp
    src/lib/ps2_audio.cpp
    src/lib/ps2_audio_vag.cpp
    src/lib/ps2_gs_backend.cpp
    src/lib/ps2_gs_coherency.cpp
    src/lib/ps2_gs_gpu.cpp
    src/lib/ps2_gs_memory.cpp
    src/lib/ps2_gs_rasterizer.cpp
    src/lib/ps2_gs_replay.cpp
    src/lib/ps2_gs_vulkan.cpp
    src/lib/ps2_gs_vulkan_backend.cpp
    src/lib/ps2_iop_host.cpp
    src/lib/ps2_memory.cpp
    src/lib/ps2_pad.cpp
    src/lib/ps2_perf_jitdump.cpp
    src/lib/ps2_runtime.cpp
    src/lib/ps2_vif1_interpreter.cpp
    src/lib/vu/ps2_vu_analysis.cpp
    src/lib/vu/ps2_vu_ir.cpp
    src/lib/vu/ps2_vu_ir_interpreter.cpp
    src/lib/vu/ps2_vu_executable_memory.cpp
    src/lib/vu/ps2_vu_program_cache.cpp
    src/lib/vu/ps2_vu_recompiler.cpp
    src/lib/vu/ps2_vu1_core.cpp
    src/lib/vu/ps2_vu1_upper.cpp
    src/lib/vu/ps2_vu1_lower.cpp
    src/lib/games_database.cpp
)

target_sources(ps2_runtime PRIVATE
    src/lib/shaders/ps2_gs_ct32_sprite.comp
    src/lib/shaders/ps2_gs_ct32_sprite_spv.inc
    src/lib/shaders/ps2_gs_depth_ct32_sprite.comp
    src/lib/shaders/ps2_gs_depth_ct32_sprite_spv.inc
    src/lib/shaders/ps2_gs_feedback_linear_depth_ct32_sprite.comp
    src/lib/shaders/ps2_gs_feedback_linear_depth_ct32_sprite_spv.inc
    src/lib/shaders/ps2_gs_gouraud_depth_ct32_triangle.comp
    src/lib/shaders/ps2_gs_gouraud_depth_ct32_triangle_spv.inc
    src/lib/shaders/ps2_gs_ct32_triangle.comp
    src/lib/shaders/ps2_gs_ct32_triangle_spv.inc
    src/lib/shaders/ps2_gs_linear_ct32_sprite.comp
    src/lib/shaders/ps2_gs_linear_ct32_sprite_spv.inc
    src/lib/shaders/ps2_gs_memory.glsl
    src/lib/shaders/ps2_gs_memory_cases.comp
    src/lib/shaders/ps2_gs_memory_cases_spv.inc
    src/lib/shaders/ps2_gs_nearest_ct32_sprite.comp
    src/lib/shaders/ps2_gs_nearest_ct32_sprite_spv.inc
    src/lib/shaders/ps2_gs_vram_noop.comp
    src/lib/shaders/ps2_gs_vram_noop_spv.inc
)
set_source_files_properties(
    src/lib/shaders/ps2_gs_ct32_sprite.comp
    src/lib/shaders/ps2_gs_ct32_sprite_spv.inc
    src/lib/shaders/ps2_gs_depth_ct32_sprite.comp
    src/lib/shaders/ps2_gs_depth_ct32_sprite_spv.inc
    src/lib/shaders/ps2_gs_feedback_linear_depth_ct32_sprite.comp
    src/lib/shaders/ps2_gs_feedback_linear_depth_ct32_sprite_spv.inc
    src/lib/shaders/ps2_gs_gouraud_depth_ct32_triangle.comp
    src/lib/shaders/ps2_gs_gouraud_depth_ct32_triangle_spv.inc
    src/lib/shaders/ps2_gs_ct32_triangle.comp
    src/lib/shaders/ps2_gs_ct32_triangle_spv.inc
    src/lib/shaders/ps2_gs_linear_ct32_sprite.comp
    src/lib/shaders/ps2_gs_linear_ct32_sprite_spv.inc
    src/lib/shaders/ps2_gs_memory.glsl
    src/lib/shaders/ps2_gs_memory_cases.comp
    src/lib/shaders/ps2_gs_memory_cases_spv.inc
    src/lib/shaders/ps2_gs_nearest_ct32_sprite.comp
    src/lib/shaders/ps2_gs_nearest_ct32_sprite_spv.inc
    src/lib/shaders/ps2_gs_vram_noop.comp
    src/lib/shaders/ps2_gs_vram_noop_spv.inc
    PROPERTIES HEADER_FILE_ONLY TRUE
)

if(PS2X_DEFAULT_EE_EXECUTION_BACKEND STREQUAL
   "legacy-cpp-fiber" AND
   NOT PS2X_HAS_EE_CPP_FIBER_BACKEND)
    message(FATAL_ERROR
        "PS2X_DEFAULT_EE_EXECUTION_BACKEND=legacy-cpp-fiber requires "
        "the validated Boost.Context fcontext backend")
endif()

if(PS2X_HAS_EE_CPP_FIBER_BACKEND)
    target_link_libraries(ps2_runtime PRIVATE Boost::context)
    target_compile_definitions(ps2_runtime PRIVATE
        PS2X_HAS_EE_CPP_FIBER_BACKEND=1
        PS2X_EE_BOOST_VERSION="1.91.0"
        PS2X_EE_BOOST_CONTEXT_ARCHITECTURE="${BOOST_CONTEXT_ARCHITECTURE}"
        PS2X_EE_BOOST_CONTEXT_BINARY_FORMAT="${BOOST_CONTEXT_BINARY_FORMAT}"
        PS2X_EE_BOOST_CONTEXT_ABI="${BOOST_CONTEXT_ABI}"
        PS2X_EE_BOOST_CONTEXT_IMPLEMENTATION="${BOOST_CONTEXT_IMPLEMENTATION}"
    )
else()
    target_compile_definitions(ps2_runtime PRIVATE
        PS2X_HAS_EE_CPP_FIBER_BACKEND=0
    )
endif()
target_compile_definitions(ps2_runtime PRIVATE
    PS2X_EE_CONTEXT_BUILD_MODE="${PS2X_EE_CONTEXT_BUILD_MODE}"
)
if(PS2X_DEFAULT_EE_EXECUTION_BACKEND STREQUAL
   "legacy-cpp-fiber")
    target_compile_definitions(ps2_runtime PRIVATE
        PS2X_DEFAULT_EE_EXECUTION_BACKEND_CPP_FIBER=1
    )
else()
    target_compile_definitions(ps2_runtime PRIVATE
        PS2X_DEFAULT_EE_EXECUTION_BACKEND_CPP_FIBER=0
    )
endif()

set(PS2X_HAS_GS_AVX2_KERNEL OFF)
if(PS2X_ENABLE_GS_AVX2_KERNEL
   AND NOT PS2X_IS_ARM_TARGET
   AND CMAKE_SYSTEM_PROCESSOR MATCHES
       "^(x86_64|AMD64|amd64|i[3-6]86)$")
    if(MSVC)
        set(PS2X_GS_AVX2_COMPILE_OPTION "/arch:AVX2")
        set(PS2X_HAS_GS_AVX2_KERNEL ON)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(
            "-mavx2" PS2X_COMPILER_SUPPORTS_MAVX2)
        if(PS2X_COMPILER_SUPPORTS_MAVX2)
            set(PS2X_GS_AVX2_COMPILE_OPTION "-mavx2")
            set(PS2X_HAS_GS_AVX2_KERNEL ON)
        endif()
    endif()
endif()

if(PS2X_HAS_GS_AVX2_KERNEL)
    target_sources(ps2_runtime PRIVATE
        src/lib/ps2_gs_rasterizer_avx2.cpp
    )
    set_source_files_properties(
        src/lib/ps2_gs_rasterizer_avx2.cpp
        PROPERTIES
            COMPILE_OPTIONS "${PS2X_GS_AVX2_COMPILE_OPTION}"
    )
    target_compile_definitions(ps2_runtime PRIVATE
        PS2X_GS_HAS_AVX2_KERNEL=1
    )
else()
    target_compile_definitions(ps2_runtime PRIVATE
        PS2X_GS_HAS_AVX2_KERNEL=0
    )
endif()

if(PS2X_HAS_GS_VULKAN)
    target_include_directories(ps2_runtime PRIVATE
        "${PS2X_GS_VULKAN_INCLUDE_DIR}"
    )
    target_compile_definitions(ps2_runtime PUBLIC
        PS2X_HAS_GS_VULKAN=1
    )
    if(CMAKE_DL_LIBS)
        target_link_libraries(ps2_runtime PRIVATE ${CMAKE_DL_LIBS})
    endif()
else()
    target_compile_definitions(ps2_runtime PUBLIC
        PS2X_HAS_GS_VULKAN=0
    )
endif()

if(PS2X_ENABLE_DEBUG_SERVER)
    target_sources(ps2_runtime PRIVATE
        src/lib/ps2_debug_server.cpp
    )
    target_include_directories(ps2_runtime PRIVATE
        "${rapidjson_SOURCE_DIR}/include"
    )
    target_compile_definitions(ps2_runtime PUBLIC
        PS2X_ENABLE_DEBUG_SERVER=1
    )
else()
    target_compile_definitions(ps2_runtime PUBLIC
        PS2X_ENABLE_DEBUG_SERVER=0
    )
endif()

if(PS2X_HAS_VU_X64_RECOMPILER)
    target_link_libraries(ps2_runtime PRIVATE xbyak::xbyak)
    target_compile_definitions(ps2_runtime PUBLIC
        PS2X_HAS_VU_X64_RECOMPILER=1
    )
else()
    target_compile_definitions(ps2_runtime PUBLIC
        PS2X_HAS_VU_X64_RECOMPILER=0
    )
endif()

# VU arithmetic observes a runtime-selected rounding mode. Prevent the
# compiler from assuming round-to-nearest or contracting multiply/add pairs.
if(MSVC)
    set_source_files_properties(
        src/lib/vu/ps2_vu1_core.cpp
        src/lib/vu/ps2_vu1_upper.cpp
        src/lib/vu/ps2_vu1_lower.cpp
        PROPERTIES COMPILE_OPTIONS "/fp:strict"
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        src/lib/vu/ps2_vu1_core.cpp
        src/lib/vu/ps2_vu1_upper.cpp
        src/lib/vu/ps2_vu1_lower.cpp
        PROPERTIES COMPILE_OPTIONS "-frounding-math;-ffp-contract=off"
    )
endif()

if(PS2X_ENABLE_RUNTIME_LOGS)
    target_compile_definitions(ps2_runtime PUBLIC
        PS2_RUNTIME_LOGS=1
    )
endif()

if(PS2X_ENABLE_AGRESSIVE_LOGS)
    target_compile_definitions(ps2_runtime PUBLIC
        AGRESSIVE_LOGS=1
        PS2_FUNCTION_LOG_TRACKER=1
    )
endif()

if(PS2X_ENABLE_IOP_RPC_TRACE)
    target_compile_definitions(ps2_runtime PUBLIC
        PS2X_ENABLE_IOP_RPC_TRACE=1
    )
else()
    target_compile_definitions(ps2_runtime PUBLIC
        PS2X_ENABLE_IOP_RPC_TRACE=0
    )
endif()

if(PS2X_STRICT_RETURN_DIAGNOSTICS)
    target_compile_definitions(ps2_runtime PUBLIC
        PS2X_STRICT_RETURN_DIAGNOSTICS=1
    )
endif()

target_compile_definitions(ps2_runtime PUBLIC
    PS2X_HAS_FFMPEG=$<BOOL:${PS2X_ENABLE_FFMPEG}>
)

file(GLOB_RECURSE KERNEL_SRC_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/lib/Kernel/*.cpp"
)

target_sources(ps2_runtime PRIVATE
    ${KERNEL_SRC_FILES}
)

get_filename_component(PS2X_RUNNER_SOURCE_DIR
    "${PS2X_RUNNER_SOURCE_DIR}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

if(NOT IS_DIRECTORY "${PS2X_RUNNER_SOURCE_DIR}")
    message(FATAL_ERROR "PS2X_RUNNER_SOURCE_DIR does not exist: ${PS2X_RUNNER_SOURCE_DIR}")
endif()

file(GLOB RUNNER_SRC_FILES CONFIGURE_DEPENDS
    "${PS2X_RUNNER_SOURCE_DIR}/*.cpp"
)

set(PS2X_RUNNER_MODULE_INCLUDE_DIRS)
set(PS2X_RUNNER_MODULE_SRC_FILES)
set(PS2X_RUNNER_MODULE_REGISTRATION_SOURCES)
foreach(PS2X_RUNNER_MODULE_SOURCE_DIR IN LISTS PS2X_RUNNER_MODULE_SOURCE_DIRS)
    get_filename_component(PS2X_RUNNER_MODULE_SOURCE_DIR
        "${PS2X_RUNNER_MODULE_SOURCE_DIR}" ABSOLUTE
        BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT IS_DIRECTORY "${PS2X_RUNNER_MODULE_SOURCE_DIR}")
        message(FATAL_ERROR
            "PS2X_RUNNER_MODULE_SOURCE_DIR does not exist: ${PS2X_RUNNER_MODULE_SOURCE_DIR}")
    endif()
    file(GLOB PS2X_RUNNER_MODULE_SOURCES CONFIGURE_DEPENDS
        "${PS2X_RUNNER_MODULE_SOURCE_DIR}/*.cpp")
    if(NOT PS2X_RUNNER_MODULE_SOURCES)
        message(FATAL_ERROR
            "PS2X_RUNNER_MODULE_SOURCE_DIR contains no C++ sources: ${PS2X_RUNNER_MODULE_SOURCE_DIR}")
    endif()
    list(APPEND PS2X_RUNNER_MODULE_INCLUDE_DIRS
        "${PS2X_RUNNER_MODULE_SOURCE_DIR}")
    list(APPEND PS2X_RUNNER_MODULE_SRC_FILES
        ${PS2X_RUNNER_MODULE_SOURCES})
    if(EXISTS "${PS2X_RUNNER_MODULE_SOURCE_DIR}/register_functions.cpp")
        list(APPEND PS2X_RUNNER_MODULE_REGISTRATION_SOURCES
            "${PS2X_RUNNER_MODULE_SOURCE_DIR}/register_functions.cpp")
    endif()
endforeach()
list(APPEND RUNNER_SRC_FILES ${PS2X_RUNNER_MODULE_SRC_FILES})

set(RUNNER_MAIN_CPP "${PS2X_RUNNER_SOURCE_DIR}/main.cpp")
set(ROOT_MAIN_CPP "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

if(EXISTS "${RUNNER_MAIN_CPP}")
    list(APPEND RUNNER_SRC_FILES "${RUNNER_MAIN_CPP}")
elseif(EXISTS "${ROOT_MAIN_CPP}")
    list(APPEND RUNNER_SRC_FILES "${ROOT_MAIN_CPP}")
endif()

if(PS2X_IS_ANDROID)
    add_library(ps2EntryRunner SHARED
        ${RUNNER_SRC_FILES}
    )

    target_link_options(ps2EntryRunner PRIVATE "-Wl,--undefined=ANativeActivity_onCreate")
else()
    add_executable(ps2EntryRunner
        ${RUNNER_SRC_FILES}
    )
endif()

target_include_directories(ps2EntryRunner PRIVATE
    "${PS2X_RUNNER_SOURCE_DIR}"
    ${PS2X_RUNNER_MODULE_INCLUDE_DIRS}
)

# Module function symbols carry a module-specific prefix and can safely use
# unity batches. Registration files contain same-named anonymous-namespace
# objects, so compile each of those as its own translation unit.
if(PS2X_RUNNER_MODULE_REGISTRATION_SOURCES)
    set_source_files_properties(${PS2X_RUNNER_MODULE_REGISTRATION_SOURCES}
        PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(PS2X_ENABLE_RUNNER_UNITY_BUILD)
    set_target_properties(ps2EntryRunner PROPERTIES
        UNITY_BUILD ON
        UNITY_BUILD_BATCH_SIZE "${PS2X_RUNNER_UNITY_BUILD_BATCH_SIZE}"
    )
endif()

if(PS2X_ENABLE_RUNNER_PCH)
    set(PS2X_RUNNER_PCH_HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ps2_runtime_macros.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ps2_runtime.h"
    )

    foreach(PS2X_RUNNER_GENERATED_HEADER IN ITEMS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ps2_recompiled_functions.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ps2_recompiled_stubs.h")
        if(EXISTS "${PS2X_RUNNER_GENERATED_HEADER}")
            list(APPEND PS2X_RUNNER_PCH_HEADERS "${PS2X_RUNNER_GENERATED_HEADER}")
        endif()
    endforeach()

    list(APPEND PS2X_RUNNER_PCH_HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ps2_syscalls.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/ps2_stubs.h"
    )

    target_precompile_headers(ps2EntryRunner PRIVATE ${PS2X_RUNNER_PCH_HEADERS})
endif()

if(PS2X_DEFAULT_BOOT_ELF_DEFINE)
    target_compile_definitions(ps2EntryRunner PRIVATE
        PS2X_DEFAULT_BOOT_ELF="${PS2X_DEFAULT_BOOT_ELF_DEFINE}"
    )
endif()

if(MSVC)
    target_compile_options(ps2EntryRunner PRIVATE /FS)
endif()

target_include_directories(ps2_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src/lib/Kernel
)

target_link_libraries(ps2_runtime PUBLIC ps2_host_backend ffmpeg ps2_iop)
target_link_libraries(ps2EntryRunner
    PRIVATE
    ps2_runtime
)

if(PS2X_ENABLE_DEBUG_UI AND NOT PS2X_IS_VITA AND NOT PS2X_IS_ANDROID)
    target_sources(ps2EntryRunner PRIVATE
        src/lib/ps2_debug_panel.cpp
    )
    target_link_libraries(ps2EntryRunner PRIVATE rlImGui imgui)
    target_compile_definitions(ps2EntryRunner PRIVATE PS2X_ENABLE_DEBUG_UI=1)
endif()

ps2x_stage_ffmpeg_runtime_dlls(ps2EntryRunner)

if(PS2X_IS_VITA)
    set(VITA_MKSFOEX_FLAGS "${VITA_MKSFOEX_FLAGS} -d PARENTAL_LEVEL=1") 
    set(VITA_MKSFOEX_FLAGS "${VITA_MKSFOEX_FLAGS} -d ATTRIBUTE2=12")

    if(PS2X_VITA_CREATE_PACKAGE)
        set(PS2X_VITA_VPK_FILES "")

        foreach(PS2X_VITA_MODULE IN ITEMS
            libgpu_es4_ext.suprx
            libIMGEGL.suprx
            libGLESv2.suprx
            libpvrPSP2_WSEGL.suprx
            libGL.suprx)
            set(PS2X_VITA_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vita/module/${PS2X_VITA_MODULE}")

            if(EXISTS "${PS2X_VITA_MODULE_PATH}")
                list(APPEND PS2X_VITA_VPK_FILES FILE "${PS2X_VITA_MODULE_PATH}" "module/${PS2X_VITA_MODULE}")
            else()
                message(WARNING
                    "Missing ${PS2X_VITA_MODULE_PATH}; the vpk will rely on ur0:data/external/ for it")
            endif()
        endforeach()

        vita_create_self(eboot.bin ps2EntryRunner UNSAFE)
        vita_create_vpk(ps2EntryRunner.vpk ${PS2X_VITA_TITLEID} eboot.bin
            VERSION ${PS2X_VITA_VERSION}
            NAME ${PS2X_VITA_APP_NAME}
            ${PS2X_VITA_VPK_FILES}
        )
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    EnableFastReleaseMode(ps2_runtime)
    EnableFastReleaseMode(ps2EntryRunner)

    if(WIN32 AND NOT PS2X_SHOW_WINDOWS_CONSOLE)
        if(MSVC)
            target_link_options(ps2EntryRunner PRIVATE
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:/SUBSYSTEM:WINDOWS>
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:/ENTRY:mainCRTStartup>
            )
        elseif(MINGW)
            target_link_options(ps2EntryRunner PRIVATE
                $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-mwindows>
            )
        endif()
    endif()
endif()

ConfigureEeFcontextSuspendedFrames(ps2_runtime)
ConfigureEeFcontextSuspendedFrames(ps2EntryRunner)

if(WIN32 AND PS2X_HAS_EE_CPP_FIBER_BACKEND)
    target_link_libraries(ps2_runtime PRIVATE Psapi)
endif()

# Work around WinAPI vs raylib symbol clash for CloseWindow on x64
if(MSVC)
    target_link_options(ps2EntryRunner PRIVATE "/FORCE:MULTIPLE")
endif()

# Vita: Work around for GCC 10's LTO plugin
if(PS2X_IS_VITA)
    target_link_options(ps2EntryRunner PRIVATE "-Wl,--allow-multiple-definition")
    target_sources(ps2EntryRunner PRIVATE src/lib/ps2_vita_runtime.cpp)
    set_source_files_properties(src/lib/ps2_vita_runtime.cpp PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION TRUE
        COMPILE_OPTIONS "-fno-lto"
    )
endif()

if(PS2X_IS_ANDROID)
    target_sources(ps2EntryRunner PRIVATE src/lib/ps2_android_runtime.cpp)
endif()

install(TARGETS ps2_runtime ps2EntryRunner
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if(WIN32 AND PS2X_ENABLE_FFMPEG)
    install(DIRECTORY "${FFMPEG_BIN_DIR}/"
        DESTINATION bin
        FILES_MATCHING
        PATTERN "*.dll"
    )
endif()

install(DIRECTORY include/
    DESTINATION include
)
