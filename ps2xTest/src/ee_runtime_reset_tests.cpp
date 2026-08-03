#include "MiniTest.h"
#include "ps2_runtime.h"
#include "ps2_runtime_macros.h"

#include <array>
#include <cstdint>

namespace
{
    constexpr uint32_t kPostBiosStatus = 0x70020C11u;
    constexpr uint32_t kR5900PrId = 0x00002E20u;
    constexpr uint32_t kPostBiosConfig = 0x00073443u;

    std::array<float, 4> vectorLanes(__m128 value)
    {
        std::array<float, 4> lanes{};
        _mm_storeu_ps(lanes.data(), value);
        return lanes;
    }

    void expectVf0(TestCase &t, const R5900Context &ctx, const char *prefix)
    {
        const std::array<float, 4> lanes = vectorLanes(ctx.vu0_vf[0]);
        t.Equals(lanes[0], 0.0f, std::string(prefix) + "VF0.x should be zero");
        t.Equals(lanes[1], 0.0f, std::string(prefix) + "VF0.y should be zero");
        t.Equals(lanes[2], 0.0f, std::string(prefix) + "VF0.z should be zero");
        t.Equals(lanes[3], 1.0f, std::string(prefix) + "VF0.w should be one");
        t.Equals(ctx.vi[0], static_cast<uint16_t>(0),
                 std::string(prefix) + "VI0 should be zero");
    }
}

void register_ee_runtime_reset_tests()
{
    MiniTest::Case("EE runtime reset state", [](TestCase &tc)
    {
        tc.Run("standalone contexts retain their deterministic defaults", [](TestCase &t)
        {
            const R5900Context ctx{};

            expectVf0(t, ctx, "standalone: ");
            t.Equals(ctx.vu0_q, 1.0f,
                     "standalone VU0 Q should retain its deterministic default");
            t.Equals(ctx.cop0_random, 47u,
                     "standalone COP0 Random should start at its reset upper bound");
            t.Equals(ctx.cop0_prid, kR5900PrId,
                     "standalone PRId should identify the R5900");
            t.Equals(ctx.cop0_config, kPostBiosConfig,
                     "standalone Config should retain the runtime configuration");
        });

        tc.Run("fresh runtimes expose the direct ELF launch profile", [](TestCase &t)
        {
            const PS2Runtime runtime;
            const R5900Context &ctx = runtime.cpu();

            expectVf0(t, ctx, "runtime: ");
            t.Equals(ctx.cop0_status, kPostBiosStatus,
                     "Status should match the standard post-BIOS ELF handoff");
            t.Equals(ctx.cop0_prid, kR5900PrId,
                     "PRId should identify the R5900 before the first MFC0");
            t.Equals(ctx.cop0_config, kPostBiosConfig,
                     "Config should expose normal post-BIOS EE modes");
            t.Equals(ctx.cop0_random, 0u,
                     "post-BIOS Random should reflect the standard loader handoff");
            t.Equals(ctx.vu0_q, 0.0f,
                     "indeterminate Q should use the captured deterministic handoff value");
        });

        tc.Run("first coprocessor uses observe the launch state", [](TestCase &t)
        {
            PS2Runtime runtime;
            R5900Context &ctx = runtime.cpu();

            bool usable = true;
            for (uint32_t coprocessor = 0u; coprocessor <= 2u; ++coprocessor)
            {
                try
                {
                    runtime.RequireCoprocessorUsable(&ctx, coprocessor);
                }
                catch (const PS2GuestException &)
                {
                    usable = false;
                    break;
                }
            }
            t.IsTrue(usable,
                     "COP0, COP1, and COP2 should be usable at direct ELF entry");

            const __m128 firstVaddq =
                PS2_VADD(ctx.vu0_vf[0], _mm_set1_ps(ctx.vu0_q));
            const std::array<float, 4> lanes = vectorLanes(firstVaddq);
            t.Equals(lanes[0], 0.0f, "first VADDq should observe VF0.x and Q");
            t.Equals(lanes[1], 0.0f, "first VADDq should observe VF0.y and Q");
            t.Equals(lanes[2], 0.0f, "first VADDq should observe VF0.z and Q");
            t.Equals(lanes[3], 1.0f, "first VADDq should observe constant VF0.w");
        });
    });
}
