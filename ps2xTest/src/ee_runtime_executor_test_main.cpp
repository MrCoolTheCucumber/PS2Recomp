#include "MiniTest.h"

#include <cstdlib>
#include <iostream>

void register_ee_runtime_executor_tests();
void reset_ps2_test_function_table();

int main()
{
    MiniTest::BeforeEach(
        reset_ps2_test_function_table);
    register_ee_runtime_executor_tests();

    const int result = MiniTest::Run();
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(result);
}
