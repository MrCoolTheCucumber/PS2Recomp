cmake_minimum_required(VERSION 3.21)

project(ps2xTest LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(PS2X_IOP_ENABLE_PLUGINS AND (WIN32 OR (UNIX AND NOT APPLE)))
    add_library(ps2_iop_fake_plugin MODULE
        src/fake_iop_plugin.cpp
    )
    add_library(ps2_iop_bad_abi_plugin MODULE
        src/fake_iop_bad_abi.c
    )
    add_library(ps2_iop_missing_symbol_plugin MODULE
        src/fake_iop_missing_symbol.cpp
    )

    foreach(plugin_target IN ITEMS
            ps2_iop_fake_plugin
            ps2_iop_bad_abi_plugin
            ps2_iop_missing_symbol_plugin)
        target_compile_features(${plugin_target} PRIVATE cxx_std_20)
        target_include_directories(${plugin_target} PRIVATE
            ${CMAKE_SOURCE_DIR}/ps2xIOP/include
        )
        set_target_properties(${plugin_target} PROPERTIES
            PREFIX ""
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/iop_test_plugins"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/iop_test_plugins"
        )
    endforeach()
endif()

# Static library with test logic (no main), used by ps2xStudio
add_library(ps2_test_function_table OBJECT
    src/test_function_table.cpp
)

target_include_directories(ps2_test_function_table PRIVATE
    ${CMAKE_SOURCE_DIR}/ps2xRuntime/include
    ${CMAKE_SOURCE_DIR}/ps2xIOP/include
)

add_library(ps2_test_lib STATIC
    src/boost_ee_fiber_tests.cpp
    src/code_generator_tests.cpp
    src/cop0_badpaddr_tests.cpp
    src/cop0_breakpoint_register_tests.cpp
    src/cop0_hardware_breakpoint_tests.cpp
    src/cop0_cause_tests.cpp
    src/cop0_config_tests.cpp
    src/cop0_context_tests.cpp
    src/cop0_entryhi_tests.cpp
    src/cop0_entrylo_tests.cpp
    src/cop0_timing_tests.cpp
    src/ee_addi_zero_tests.cpp
    src/ee_address_mode_tests.cpp
    src/ee_bus_error_tests.cpp
    src/ee_coprocessor_unusable_tests.cpp
    src/ee_cycle_model_tests.cpp
    src/ee_cache_tests.cpp
    src/ee_counter_tests.cpp
    src/ee_execution_backend_tests.cpp
    src/ee_event_scheduler_tests.cpp
    src/ee_load_zero_tests.cpp
    src/ee_runtime_reset_tests.cpp
    src/ee_runtime_executor_tests.cpp
    src/ee_reserved_instruction_tests.cpp
    src/ee_syscall_delay_tests.cpp
    src/ee_thread_scheduler_tests.cpp
    src/r5900_decoder_tests.cpp
    src/elf_analyzer_tests.cpp
    src/ps2_audio_tests.cpp
    src/pad_input_tests.cpp
    src/ps2_runtime_elf_tests.cpp
    src/ps2_runtime_io_tests.cpp
    src/ps2_runtime_kernel_tests.cpp
    src/ps2_runtime_interrupt_tests.cpp
    src/ps2_memory_tests.cpp
    src/ps2_perf_jitdump_tests.cpp
    src/ps2_vu_executable_memory_tests.cpp
    src/ps2_vu_program_cache_tests.cpp
    src/ps2_vu_recompiler_tests.cpp
    src/ps2_vu1_tests.cpp
    src/ps2_gs_coherency_tests.cpp
    src/ps2_gs_tests.cpp
    src/ps2_gs_vulkan_tests.cpp
    src/ps2_iop_tests.cpp
    src/ps2_sif_rpc_tests.cpp
    src/ps2_sif_dma_tests.cpp
    src/ps2_recompiler_tests.cpp
    src/ps2_runtime_expansion_tests.cpp
    src/ps2_debug_control_tests.cpp
)

if(UNIX AND NOT APPLE AND NOT WIN32 AND
        CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    enable_language(ASM)
    target_sources(ps2_test_lib PRIVATE
        src/ps2_vu_recompiler_abi.S
    )
    target_compile_definitions(ps2_test_lib PRIVATE
        PS2X_TEST_HAS_VU_RECOMPILER_ABI_PROBE=1
    )
endif()

target_include_directories(ps2_test_lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ps2xRecomp/include
    ${CMAKE_SOURCE_DIR}/ps2xAnalyzer/include
    ${CMAKE_SOURCE_DIR}/ps2xRuntime/include
    ${CMAKE_SOURCE_DIR}/ps2xRuntime/src/lib
)

target_link_libraries(ps2_test_lib PRIVATE
    ps2_recomp_lib
    ps2_analyzer_lib
    ps2_runtime
    ps2_iop
)

if(UNIX AND NOT APPLE)
    target_link_libraries(ps2_test_lib PRIVATE ${CMAKE_DL_LIBS})
endif()

if(TARGET ps2_iop_fake_plugin)
    target_compile_definitions(ps2_test_lib PRIVATE
        PS2X_TEST_IOP_PLUGIN_DIR="$<TARGET_FILE_DIR:ps2_iop_fake_plugin>"
    )
    add_dependencies(ps2_test_lib
        ps2_iop_fake_plugin
        ps2_iop_bad_abi_plugin
        ps2_iop_missing_symbol_plugin
    )
endif()

add_executable(ps2x_tests
    src/main.cpp
    $<TARGET_OBJECTS:ps2_test_function_table>
)

add_executable(ee_runtime_executor_tests
    src/ee_runtime_executor_test_main.cpp
    $<TARGET_OBJECTS:ps2_test_function_table>
)

add_executable(ee_fiber_guard_probe
    src/ee_fiber_guard_probe.cpp
)

target_include_directories(ee_fiber_guard_probe PRIVATE
    ${CMAKE_SOURCE_DIR}/ps2xRuntime/include
)

target_link_libraries(ee_fiber_guard_probe PRIVATE
    ps2_runtime
)

add_dependencies(ps2x_tests ee_fiber_guard_probe)
target_compile_definitions(ps2_test_lib PRIVATE
    PS2X_TEST_EE_FIBER_GUARD_PROBE="$<TARGET_FILE:ee_fiber_guard_probe>"
)

add_executable(vu1_replay
    src/vu1_replay.cpp
    $<TARGET_OBJECTS:ps2_test_function_table>
)

add_executable(vu_backend_benchmark
    src/vu_backend_benchmark.cpp
    src/standalone_runtime_function_table.cpp
)

option(PS2X_BUILD_VU_NATIVE_EMITTER_SPIKE
    "Build the disposable Phase 0 x86-64 VU native-emitter spike" OFF)
set(PS2X_XBYAK_INCLUDE_DIR "" CACHE PATH
    "Directory containing xbyak/xbyak.h for the VU emitter spike")

if(PS2X_BUILD_VU_NATIVE_EMITTER_SPIKE)
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        message(FATAL_ERROR
            "The Phase 0 VU native-emitter spike only supports x86-64")
    endif()
    if(MSVC OR WIN32)
        message(FATAL_ERROR
            "The Phase 0 VU native-emitter spike currently uses the SysV x86-64 ABI")
    endif()
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR
            "The Phase 0 VU native-emitter spike currently validates W^X through Linux /proc")
    endif()
    if(NOT EXISTS "${PS2X_XBYAK_INCLUDE_DIR}/xbyak/xbyak.h")
        message(FATAL_ERROR
            "Set PS2X_XBYAK_INCLUDE_DIR to a pinned Xbyak checkout")
    endif()

    enable_language(ASM)
    add_executable(vu_native_emitter_spike
        src/vu_native_emitter_spike.cpp
        src/vu_native_emitter_spike_abi.S
        src/standalone_runtime_function_table.cpp
    )
    target_include_directories(vu_native_emitter_spike PRIVATE
        ${CMAKE_SOURCE_DIR}/ps2xRuntime/include
        ${PS2X_XBYAK_INCLUDE_DIR}
    )
    target_link_libraries(vu_native_emitter_spike PRIVATE
        ps2_runtime
    )
endif()

add_executable(ee_timing_benchmark
    src/ee_timing_benchmark.cpp
    $<TARGET_OBJECTS:ps2_test_function_table>
)

add_executable(ee_thread_scheduler_benchmark
    src/ee_thread_scheduler_benchmark.cpp
    $<TARGET_OBJECTS:ps2_test_function_table>
)

add_executable(gs_replay
    src/gs_replay.cpp
    src/standalone_runtime_function_table.cpp
)

option(PRINT_GENERATED_CODE "Print generated code in tests" OFF)
if(PRINT_GENERATED_CODE)

    target_compile_definitions(ps2x_tests PRIVATE PRINT_GENERATED_CODE)
    target_compile_definitions(ps2_test_lib PRIVATE PRINT_GENERATED_CODE)
endif()

target_include_directories(ps2x_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ps2xRecomp/include
    ${CMAKE_SOURCE_DIR}/ps2xAnalyzer/include
    ${CMAKE_SOURCE_DIR}/ps2xRuntime/include
)

target_include_directories(ee_runtime_executor_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/ps2xRecomp/include
    ${CMAKE_SOURCE_DIR}/ps2xAnalyzer/include
    ${CMAKE_SOURCE_DIR}/ps2xRuntime/include
)

target_link_libraries(ps2x_tests PRIVATE
    ps2_test_lib
    ps2_recomp_lib
    ps2_analyzer_lib
    ps2_runtime
)

target_link_libraries(ee_runtime_executor_tests PRIVATE
    ps2_test_lib
    ps2_recomp_lib
    ps2_analyzer_lib
    ps2_runtime
)

target_include_directories(vu1_replay PRIVATE
    ${CMAKE_SOURCE_DIR}/ps2xRuntime/include
)

target_link_libraries(vu1_replay PRIVATE
    ps2_runtime
)

target_include_directories(vu_backend_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/ps2xRuntime/include
)

target_link_libraries(vu_backend_benchmark PRIVATE
    ps2_runtime
)

target_include_directories(ee_timing_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/ps2xRuntime/include
)

target_link_libraries(ee_timing_benchmark PRIVATE
    ps2_runtime
)

target_include_directories(ee_thread_scheduler_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/ps2xRuntime/include
)

target_link_libraries(ee_thread_scheduler_benchmark PRIVATE
    ps2_runtime
)

target_include_directories(gs_replay PRIVATE
    ${CMAKE_SOURCE_DIR}/ps2xRuntime/include
)

target_link_libraries(gs_replay PRIVATE
    ps2_runtime
)

if(COMMAND ps2x_stage_ffmpeg_runtime_dlls)
    ps2x_stage_ffmpeg_runtime_dlls(ps2x_tests)
endif()

include("${CMAKE_SOURCE_DIR}/ps2xRuntime/cmake/ReleaseMode.cmake")

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    EnableFastReleaseMode(ps2_test_lib)
    EnableFastReleaseMode(ps2x_tests)
    EnableFastReleaseMode(ee_runtime_executor_tests)
    EnableFastReleaseMode(ee_timing_benchmark)
    EnableFastReleaseMode(ee_thread_scheduler_benchmark)
    EnableFastReleaseMode(vu_backend_benchmark)
    if(TARGET vu_native_emitter_spike)
        EnableFastReleaseMode(vu_native_emitter_spike)
    endif()
endif()

ConfigureEeFcontextSuspendedFrames(ps2_test_lib)
ConfigureEeFcontextSuspendedFrames(ps2x_tests)
ConfigureEeFcontextSuspendedFrames(ee_runtime_executor_tests)
ConfigureEeFcontextSuspendedFrames(ee_fiber_guard_probe)
