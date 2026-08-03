#include "MiniTest.h"
#include <cstdlib>
#include <iostream>

void register_code_generator_tests();
void register_boost_ee_fiber_tests();
void register_cop0_timing_tests();
void register_ee_cache_tests();
void register_ee_counter_tests();
void register_ee_execution_backend_tests();
void register_ee_event_scheduler_tests();
void register_ee_load_zero_tests();
void register_ee_runtime_executor_tests();
void register_ee_reserved_instruction_tests();
void register_ee_thread_scheduler_tests();
void register_r5900_decoder_tests();
void register_elf_analyzer_tests();
void register_ps2_audio_tests();
void register_pad_input_tests();
void register_ps2_runtime_elf_tests();
void register_ps2_runtime_io_tests();
void register_ps2_runtime_kernel_tests();
void register_ps2_runtime_interrupt_tests();
void register_ps2_memory_tests();
void register_ps2_perf_jitdump_tests();
void register_ps2_vu_executable_memory_tests();
void register_ps2_vu_program_cache_tests();
void register_ps2_vu_recompiler_tests();
void register_ps2_vu1_tests();
void register_ps2_gs_tests();
void register_ps2_gs_coherency_tests();
void register_ps2_gs_vulkan_tests();
void register_ps2_iop_tests();
void register_ps2_sif_rpc_tests();
void register_ps2_sif_dma_tests();
void register_ps2_recompiler_tests();
void register_ps2_runtime_expansion_tests();
void register_ps2_debug_control_tests();
void reset_ps2_test_function_table();

int main()
{
    MiniTest::BeforeEach(reset_ps2_test_function_table);

    register_code_generator_tests();
    register_boost_ee_fiber_tests();
    register_cop0_timing_tests();
    register_ee_cache_tests();
    register_ee_counter_tests();
    register_ee_execution_backend_tests();
    register_ee_event_scheduler_tests();
    register_ee_load_zero_tests();
    register_ee_runtime_executor_tests();
    register_ee_reserved_instruction_tests();
    register_ee_thread_scheduler_tests();
    register_r5900_decoder_tests();
    register_elf_analyzer_tests();
    register_ps2_audio_tests();
    register_pad_input_tests();
    register_ps2_runtime_elf_tests();
    register_ps2_runtime_io_tests();
    register_ps2_runtime_kernel_tests();
    register_ps2_runtime_interrupt_tests();
    register_ps2_memory_tests();
    register_ps2_perf_jitdump_tests();
    register_ps2_vu_executable_memory_tests();
    register_ps2_vu_program_cache_tests();
    register_ps2_vu_recompiler_tests();
    register_ps2_vu1_tests();
    register_ps2_gs_tests();
    register_ps2_gs_coherency_tests();
    register_ps2_gs_vulkan_tests();
    register_ps2_iop_tests();
    register_ps2_sif_rpc_tests();
    register_ps2_sif_dma_tests();
    register_ps2_recompiler_tests();
    register_ps2_runtime_expansion_tests();
    register_ps2_debug_control_tests();
    int res = MiniTest::Run();
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(res);
}
