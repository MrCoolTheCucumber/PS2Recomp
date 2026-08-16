#include "MiniTest.h"

#include <cstdlib>
#include <iostream>

void register_boost_ee_fiber_tests();
void register_ee_execution_backend_tests();
void register_ee_runtime_executor_tests();
void register_ee_thread_scheduler_tests();
void register_ps2_runtime_interrupt_tests();
void register_ps2_runtime_kernel_tests();
void reset_ps2_test_function_table();

int main()
{
    MiniTest::BeforeEach(
        reset_ps2_test_function_table);
    register_boost_ee_fiber_tests();
    register_ee_execution_backend_tests();
    register_ee_thread_scheduler_tests();
    register_ee_runtime_executor_tests();
    register_ps2_runtime_kernel_tests();
    register_ps2_runtime_interrupt_tests();

    const int result = MiniTest::Run();
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(result);
}
