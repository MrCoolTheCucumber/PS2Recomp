#include "MiniTest.h"

#include <cstdlib>
#include <iostream>

void register_ps2_vu1_command_stream_tests();
void reset_ps2_test_function_table();

int main()
{
    MiniTest::BeforeEach(reset_ps2_test_function_table);
    register_ps2_vu1_command_stream_tests();

    const int result = MiniTest::Run();
    std::cout.flush();
    std::cerr.flush();
    std::_Exit(result);
}
