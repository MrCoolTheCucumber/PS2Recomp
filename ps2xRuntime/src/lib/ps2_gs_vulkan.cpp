#include "runtime/ps2_gs_vulkan.h"
#include "ThreadNaming.h"
#include "runtime/ps2_gs_common.h"
#include "runtime/ps2_gs_memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <sstream>
#include <thread>
#include <tuple>
#include <utility>

#if PS2X_HAS_GS_VULKAN
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

std::string_view gsVulkanProbeStatusName(
    GsVulkanProbeStatus status) noexcept
{
    switch (status)
    {
    case GsVulkanProbeStatus::CompiledOut:
        return "compiled-out";
    case GsVulkanProbeStatus::LoaderUnavailable:
        return "loader-unavailable";
    case GsVulkanProbeStatus::LoaderInvalid:
        return "loader-invalid";
    case GsVulkanProbeStatus::ValidationUnavailable:
        return "validation-unavailable";
    case GsVulkanProbeStatus::InstanceCreationFailed:
        return "instance-creation-failed";
    case GsVulkanProbeStatus::DeviceEnumerationFailed:
        return "device-enumeration-failed";
    case GsVulkanProbeStatus::NoPhysicalDevices:
        return "no-physical-devices";
    case GsVulkanProbeStatus::NoSuitableDevice:
        return "no-suitable-device";
    case GsVulkanProbeStatus::DeviceCreationFailed:
        return "device-creation-failed";
    case GsVulkanProbeStatus::ResourceCreationFailed:
        return "resource-creation-failed";
    case GsVulkanProbeStatus::ExecutionTimeout:
        return "execution-timeout";
    case GsVulkanProbeStatus::DeviceLost:
        return "device-lost";
    case GsVulkanProbeStatus::ValidationError:
        return "validation-error";
    case GsVulkanProbeStatus::Ready:
        return "ready";
    }
    return "unknown";
}

std::string_view gsVulkanDeviceKindName(
    GsVulkanDeviceKind kind) noexcept
{
    switch (kind)
    {
    case GsVulkanDeviceKind::Other:
        return "other";
    case GsVulkanDeviceKind::IntegratedGpu:
        return "integrated-gpu";
    case GsVulkanDeviceKind::DiscreteGpu:
        return "discrete-gpu";
    case GsVulkanDeviceKind::VirtualGpu:
        return "virtual-gpu";
    case GsVulkanDeviceKind::Cpu:
        return "cpu";
    }
    return "unknown";
}

std::string gsVulkanVersionString(uint32_t version)
{
    const uint32_t variant = version >> 29u;
    const uint32_t major = (version >> 22u) & 0x7Fu;
    const uint32_t minor = (version >> 12u) & 0x3FFu;
    const uint32_t patch = version & 0xFFFu;
    std::ostringstream output;
    if (variant != 0u)
        output << variant << ':';
    output << major << '.' << minor << '.' << patch;
    return output.str();
}

namespace
{
    bool ct32RectangleHasUniqueWords(
        uint32_t x0,
        uint32_t y0,
        uint32_t x1,
        uint32_t y1,
        uint32_t framebufferWidth) noexcept;

    const char *ct32SpriteValidationError(
        const GsVulkanCt32Sprite &sprite) noexcept
    {
        if (sprite.reserved != 0u)
            return "Vulkan CT32 sprite has non-zero reserved data";
        if (sprite.framebufferBaseBlock > 0x3FFFu)
            return "Vulkan CT32 sprite framebuffer base is outside GS VRAM";
        if (sprite.framebufferWidth == 0u ||
            sprite.framebufferWidth > 0x3Fu)
            return "Vulkan CT32 sprite framebuffer width is outside FRAME range";
        if (sprite.x0 >= sprite.x1 || sprite.y0 >= sprite.y1)
            return "Vulkan CT32 sprite bounds are empty";
        if (sprite.x1 > 2048u || sprite.y1 > 2048u)
            return "Vulkan CT32 sprite bounds are outside GS scissor range";
        return nullptr;
    }

    const char *depthCt32SpriteShapeValidationError(
        const GsVulkanDepthCt32Sprite &sprite) noexcept
    {
        const bool usesDepth = sprite.depthTestMethod != 0u;
        if (sprite.framebufferBaseBlock > 0x3FFFu ||
            (usesDepth && sprite.depthBaseBlock > 0x3FFFu))
        {
            return "Vulkan depth CT32 sprite base is outside GS VRAM";
        }
        if (sprite.framebufferWidth == 0u ||
            sprite.framebufferWidth > 0x3Fu)
        {
            return "Vulkan depth CT32 sprite width is outside FRAME range";
        }
        if (sprite.boundsX0 >= sprite.boundsX1 ||
            sprite.boundsY0 >= sprite.boundsY1)
        {
            return "Vulkan depth CT32 sprite bounds are empty";
        }
        if (sprite.boundsX1 > 2048u || sprite.boundsY1 > 2048u)
            return "Vulkan depth CT32 sprite bounds are outside GS scissor range";
        if (usesDepth &&
            sprite.depthPsm != GS_PSM_Z32 &&
            sprite.depthPsm != GS_PSM_Z24)
        {
            return "Vulkan depth CT32 sprite has unsupported depth format";
        }
        if (sprite.depthTestMethod > 3u ||
            sprite.depthWrite > 1u ||
            (!usesDepth &&
             (sprite.depthBaseBlock != 0u || sprite.depthPsm != 0u ||
              sprite.depth != 0u || sprite.depthWrite != 0u)))
        {
            return "Vulkan depth CT32 sprite has unsupported depth function";
        }
        if (!ct32RectangleHasUniqueWords(
                sprite.boundsX0,
                sprite.boundsY0,
                sprite.boundsX1,
                sprite.boundsY1,
                sprite.framebufferWidth))
        {
            return "Vulkan depth CT32 sprite surface aliases itself";
        }

        const uint32_t width = sprite.boundsX1 - sprite.boundsX0;
        const uint32_t height = sprite.boundsY1 - sprite.boundsY0;
        const GsVramPageMask framebufferPages =
            gsVramPagesForSurfaceRect(
                sprite.framebufferBaseBlock,
                sprite.framebufferWidth,
                static_cast<uint8_t>(GSMem::C32),
                sprite.boundsX0,
                sprite.boundsY0,
                width,
                height);
        if (usesDepth)
        {
            const GsVramPageMask depthPages =
                gsVramPagesForSurfaceRect(
                    sprite.depthBaseBlock,
                    sprite.framebufferWidth,
                    static_cast<uint8_t>(sprite.depthPsm),
                    sprite.boundsX0,
                    sprite.boundsY0,
                    width,
                    height);
            if (framebufferPages.intersects(depthPages))
                return "Vulkan depth CT32 sprite framebuffer aliases depth";
        }
        return nullptr;
    }

    const char *depthCt32SpriteValidationError(
        const GsVulkanDepthCt32Sprite &sprite) noexcept
    {
        if ((sprite.reserved1 | sprite.reserved2 |
             sprite.reserved3) != 0u)
        {
            return "Vulkan depth CT32 sprite has non-zero reserved data";
        }
        const uint32_t descriptor = sprite.colorBlendMode;
        if ((descriptor &
             ~GS_VULKAN_DEPTH_CT32_COLOR_DESCRIPTOR_MASK) != 0u)
        {
            return "Vulkan depth CT32 sprite has unsupported color operation";
        }
        const uint32_t operation =
            descriptor & GS_VULKAN_DEPTH_CT32_COLOR_OPERATION_MASK;
        if (operation > GS_VULKAN_DEPTH_CT32_COLOR_FIXED_ALPHA)
            return "Vulkan depth CT32 sprite has unsupported color operation";
        if (operation == GS_VULKAN_DEPTH_CT32_COLOR_FIXED_ALPHA)
        {
            if ((descriptor & GS_VULKAN_DEPTH_CT32_COLOR_DATM) != 0u &&
                (descriptor & GS_VULKAN_DEPTH_CT32_COLOR_DATE) == 0u)
            {
                return "Vulkan depth CT32 sprite has DATM without DATE";
            }
        }
        else if (descriptor != operation)
        {
            return "Vulkan depth CT32 sprite has color parameters for a literal operation";
        }
        return depthCt32SpriteShapeValidationError(sprite);
    }

    struct DepthCt32SpriteAccessPages
    {
        GsVramPageMask readPages;
        GsVramPageMask writePages;
    };

    DepthCt32SpriteAccessPages depthCt32SpriteAccessPages(
        const GsVulkanDepthCt32Sprite &sprite) noexcept
    {
        const uint32_t width = sprite.boundsX1 - sprite.boundsX0;
        const uint32_t height = sprite.boundsY1 - sprite.boundsY0;
        DepthCt32SpriteAccessPages access{};
        const GsVramPageMask framebufferPages =
            gsVramPagesForSurfaceRect(
                sprite.framebufferBaseBlock,
                sprite.framebufferWidth,
                static_cast<uint8_t>(GSMem::C32),
                sprite.boundsX0,
                sprite.boundsY0,
                width,
                height);
        access.writePages = framebufferPages;
        const uint32_t colorOperation =
            sprite.colorBlendMode &
            GS_VULKAN_DEPTH_CT32_COLOR_OPERATION_MASK;
        if (colorOperation !=
                GS_VULKAN_DEPTH_CT32_COLOR_SOURCE_COPY ||
            (sprite.colorBlendMode &
             GS_VULKAN_DEPTH_CT32_COLOR_DATE) != 0u)
        {
            access.readPages.unionWith(framebufferPages);
        }
        if (sprite.depthTestMethod != 0u)
        {
            const GsVramPageMask depthPages =
                gsVramPagesForSurfaceRect(
                    sprite.depthBaseBlock,
                    sprite.framebufferWidth,
                    static_cast<uint8_t>(sprite.depthPsm),
                    sprite.boundsX0,
                    sprite.boundsY0,
                    width,
                    height);
            if (sprite.depthTestMethod >= 2u)
                access.readPages.unionWith(depthPages);
            if (sprite.depthWrite != 0u)
                access.writePages.unionWith(depthPages);
        }
        return access;
    }

    bool isPowerOfTwoMask(uint32_t mask) noexcept
    {
        return mask <= 1023u && (mask & (mask + 1u)) == 0u;
    }

    bool ct32RectangleHasUniqueWords(
        uint32_t x0,
        uint32_t y0,
        uint32_t x1,
        uint32_t y1,
        uint32_t framebufferWidth) noexcept
    {
        const uint32_t minimumColumn = x0 / 64u;
        const uint32_t maximumColumn = (x1 - 1u) / 64u;
        const uint32_t minimumRow = y0 / 32u;
        const uint32_t maximumRow = (y1 - 1u) / 32u;
        const uint32_t columns = maximumColumn - minimumColumn + 1u;
        if (columns > framebufferWidth)
            return false;

        const uint64_t logicalPageSpan =
            static_cast<uint64_t>(maximumRow - minimumRow) *
                framebufferWidth +
            (maximumColumn - minimumColumn);
        return logicalPageSpan < GS_VRAM_PAGE_COUNT;
    }

    uint32_t maximumNearestCt32TextureCoordinate(
        uint32_t textureMask,
        uint32_t wrap) noexcept
    {
        return GSInternal::maximumWrappedTextureCoordinate(
            textureMask + 1u,
            gsVulkanTextureWrapMode(wrap),
            gsVulkanTextureRegionMin(wrap),
            gsVulkanTextureRegionMax(wrap));
    }

    GsVramPageMask nearestCt32TexturePages(
        const GsVulkanNearestCt32Sprite &sprite) noexcept
    {
        return gsVramPagesForSurfaceRect(
            sprite.textureBaseBlock,
            sprite.textureWidth,
            static_cast<uint8_t>(GSMem::C32),
            0u,
            0u,
            maximumNearestCt32TextureCoordinate(
                sprite.textureMaskU, sprite.textureWrapU) + 1u,
            maximumNearestCt32TextureCoordinate(
                sprite.textureMaskV, sprite.textureWrapV) + 1u);
    }

    const char *nearestCt32SpriteValidationError(
        const GsVulkanNearestCt32Sprite &sprite) noexcept
    {
        if ((sprite.textureWrapU &
             ~GS_VULKAN_TEXTURE_WRAP_DESCRIPTOR_MASK) != 0u ||
            (sprite.textureWrapV &
             ~GS_VULKAN_TEXTURE_WRAP_DESCRIPTOR_MASK) != 0u)
        {
            return "Vulkan nearest CT32 sprite wrap descriptor is invalid";
        }
        if ((gsVulkanTextureWrapMode(sprite.textureWrapU) == 2u &&
             gsVulkanTextureRegionMin(sprite.textureWrapU) >
                 gsVulkanTextureRegionMax(sprite.textureWrapU)) ||
            (gsVulkanTextureWrapMode(sprite.textureWrapV) == 2u &&
             gsVulkanTextureRegionMin(sprite.textureWrapV) >
                 gsVulkanTextureRegionMax(sprite.textureWrapV)))
        {
            return "Vulkan nearest CT32 sprite region clamp is reversed";
        }
        if (sprite.framebufferBaseBlock > 0x3FFFu ||
            sprite.textureBaseBlock > 0x3FFFu)
        {
            return "Vulkan nearest CT32 sprite base is outside GS VRAM";
        }
        if (sprite.framebufferWidth == 0u ||
            sprite.framebufferWidth > 0x3Fu ||
            sprite.textureWidth == 0u || sprite.textureWidth > 0x3Fu)
        {
            return "Vulkan nearest CT32 sprite width is outside GS register range";
        }
        if (sprite.boundsX0 >= sprite.boundsX1 ||
            sprite.boundsY0 >= sprite.boundsY1)
        {
            return "Vulkan nearest CT32 sprite bounds are empty";
        }
        if (sprite.boundsX1 > 2048u || sprite.boundsY1 > 2048u)
            return "Vulkan nearest CT32 sprite bounds are outside GS scissor range";
        if (!ct32RectangleHasUniqueWords(
                sprite.boundsX0,
                sprite.boundsY0,
                sprite.boundsX1,
                sprite.boundsY1,
                sprite.framebufferWidth))
        {
            return "Vulkan nearest CT32 sprite destination aliases itself";
        }
        if (!isPowerOfTwoMask(sprite.textureMaskU) ||
            !isPowerOfTwoMask(sprite.textureMaskV))
        {
            return "Vulkan nearest CT32 sprite texture mask is invalid";
        }
        if (sprite.textureOriginU < 0 || sprite.textureOriginU > 1023 ||
            sprite.textureOriginV < 0 || sprite.textureOriginV > 1023)
        {
            return "Vulkan nearest CT32 sprite texture origin is outside UV range";
        }
        if ((sprite.textureStepU != -1 && sprite.textureStepU != 1) ||
            (sprite.textureStepV != -1 && sprite.textureStepV != 1))
        {
            return "Vulkan nearest CT32 sprite texture step is not unit length";
        }
        const GsVramPageMask texturePages =
            nearestCt32TexturePages(sprite);
        const GsVramPageMask framebufferPages = gsVramPagesForSurfaceRect(
            sprite.framebufferBaseBlock,
            sprite.framebufferWidth,
            static_cast<uint8_t>(GSMem::C32),
            sprite.boundsX0,
            sprite.boundsY0,
            sprite.boundsX1 - sprite.boundsX0,
            sprite.boundsY1 - sprite.boundsY0);
        if (texturePages.intersects(framebufferPages))
            return "Vulkan nearest CT32 sprite source aliases destination";
        return nullptr;
    }

    bool linearDdaFloatFitsInt32(float value) noexcept
    {
        constexpr float kInt32Minimum = -2147483648.0f;
        constexpr float kInt32Limit = 2147483648.0f;
        return std::isfinite(value) &&
               value >= kInt32Minimum && value < kInt32Limit;
    }

    GsVramPageMask linearCt32TexturePages(
        const GsVulkanLinearCt32Sprite &sprite) noexcept
    {
        return gsVramPagesForSurfaceRect(
            sprite.textureBaseBlock,
            sprite.textureWidth,
            static_cast<uint8_t>(GSMem::C32),
            0u,
            0u,
            sprite.textureMaskU + 1u,
            sprite.textureMaskV + 1u);
    }

    const char *linearCt32SpriteRecordValidationError(
        const GsVulkanLinearCt32Sprite &sprite,
        bool exactFramebufferFeedback = false) noexcept
    {
        if ((sprite.textureWrapU &
             ~GS_VULKAN_TEXTURE_WRAP_DESCRIPTOR_MASK) != 0u ||
            (sprite.textureWrapV &
             ~GS_VULKAN_TEXTURE_WRAP_DESCRIPTOR_MASK) != 0u ||
            gsVulkanTextureWrapMode(sprite.textureWrapU) > 1u ||
            gsVulkanTextureWrapMode(sprite.textureWrapV) > 1u)
        {
            return "Vulkan linear CT32 sprite requires repeat or clamp wrap";
        }
        if (sprite.framebufferBaseBlock > 0x3FFFu ||
            sprite.textureBaseBlock > 0x3FFFu)
        {
            return "Vulkan linear CT32 sprite base is outside GS VRAM";
        }
        if (sprite.framebufferWidth == 0u ||
            sprite.framebufferWidth > 0x3Fu ||
            sprite.textureWidth == 0u || sprite.textureWidth > 0x3Fu)
        {
            return "Vulkan linear CT32 sprite width is outside GS register range";
        }
        if (sprite.boundsX0 >= sprite.boundsX1 ||
            sprite.boundsY0 >= sprite.boundsY1)
        {
            return "Vulkan linear CT32 sprite bounds are empty";
        }
        if (sprite.boundsX1 > 2048u || sprite.boundsY1 > 2048u)
            return "Vulkan linear CT32 sprite bounds are outside GS scissor range";
        if (!ct32RectangleHasUniqueWords(
                sprite.boundsX0,
                sprite.boundsY0,
                sprite.boundsX1,
                sprite.boundsY1,
                sprite.framebufferWidth))
        {
            return "Vulkan linear CT32 sprite destination aliases itself";
        }
        if (!isPowerOfTwoMask(sprite.textureMaskU) ||
            !isPowerOfTwoMask(sprite.textureMaskV))
        {
            return "Vulkan linear CT32 sprite texture mask is invalid";
        }

        float fixedScanV = std::bit_cast<float>(sprite.fixedScanVBits);
        const float fixedStepV =
            std::bit_cast<float>(sprite.fixedStepVBits);
        if (!std::isfinite(fixedStepV))
            return "Vulkan linear CT32 sprite V step is not finite";
        for (uint32_t y = sprite.boundsY0; y < sprite.boundsY1; ++y)
        {
            if (!linearDdaFloatFitsInt32(fixedScanV))
            {
                return "Vulkan linear CT32 sprite V coordinate is outside signed fixed range";
            }
            fixedScanV += fixedStepV;
        }

        const GsVramPageMask texturePages =
            linearCt32TexturePages(sprite);
        const GsVramPageMask framebufferPages = gsVramPagesForSurfaceRect(
            sprite.framebufferBaseBlock,
            sprite.framebufferWidth,
            static_cast<uint8_t>(GSMem::C32),
            sprite.boundsX0,
            sprite.boundsY0,
            sprite.boundsX1 - sprite.boundsX0,
            sprite.boundsY1 - sprite.boundsY0);
        const bool aliasesDestination =
            texturePages.intersects(framebufferPages);
        if (!exactFramebufferFeedback && aliasesDestination)
            return "Vulkan linear CT32 sprite source aliases destination";
        if (exactFramebufferFeedback &&
            (!aliasesDestination ||
             sprite.textureBaseBlock != sprite.framebufferBaseBlock ||
             sprite.textureWidth != sprite.framebufferWidth))
        {
            return "Vulkan feedback linear CT32 sprite requires one exact source surface";
        }
        return nullptr;
    }

    const char *linearCt32SpriteValidationError(
        const GsVulkanLinearCt32Sprite &sprite) noexcept
    {
        return linearCt32SpriteRecordValidationError(sprite);
    }

    struct FixedTriangleVertex
    {
        int32_t x;
        int32_t y;
    };

    int64_t triangleEdge(
        const FixedTriangleVertex &a,
        const FixedTriangleVertex &b,
        const FixedTriangleVertex &sample) noexcept
    {
        return
            (static_cast<int64_t>(b.x) - a.x) *
                (static_cast<int64_t>(sample.y) - a.y) -
            (static_cast<int64_t>(b.y) - a.y) *
                (static_cast<int64_t>(sample.x) - a.x);
    }

    bool isTopLeftEdge(
        const FixedTriangleVertex &a,
        const FixedTriangleVertex &b) noexcept
    {
        const int64_t dx =
            static_cast<int64_t>(b.x) - a.x;
        const int64_t dy =
            static_cast<int64_t>(b.y) - a.y;
        return dy < 0 || (dy == 0 && dx > 0);
    }

    struct GouraudColorDdaBits
    {
        std::array<uint32_t, 4> dx{};
        std::array<uint32_t, 4> dy{};
    };

    GouraudColorDdaBits prepareGouraudColorDdaBits(
        const std::array<FixedTriangleVertex, 3> &fixedVertices,
        const std::array<uint32_t, 3> &rgba) noexcept
    {
        struct Vertex
        {
            float x;
            float y;
            uint32_t rgba;
        };
        std::array<Vertex, 3> sorted{{
            {static_cast<float>(fixedVertices[0].x) / 16.0f,
             static_cast<float>(fixedVertices[0].y) / 16.0f,
             rgba[0]},
            {static_cast<float>(fixedVertices[1].x) / 16.0f,
             static_cast<float>(fixedVertices[1].y) / 16.0f,
             rgba[1]},
            {static_cast<float>(fixedVertices[2].x) / 16.0f,
             static_cast<float>(fixedVertices[2].y) / 16.0f,
             rgba[2]},
        }};
        if (sorted[1].y < sorted[0].y)
            std::swap(sorted[0], sorted[1]);
        if (sorted[2].y < sorted[1].y)
            std::swap(sorted[1], sorted[2]);
        if (sorted[1].y < sorted[0].y)
            std::swap(sorted[0], sorted[1]);

        const float gradientDx01 = sorted[1].x - sorted[0].x;
        const float gradientDy01 = sorted[1].y - sorted[0].y;
        const float gradientDx02 = sorted[2].x - sorted[0].x;
        const float gradientDy02 = sorted[2].y - sorted[0].y;
        const float gradientCross = std::fma(
            gradientDy01,
            gradientDx02,
            -(gradientDx01 * gradientDy02));
        GouraudColorDdaBits result{};
        if (gradientCross == 0.0f)
            return result;

        const float gradientX01OverCross =
            gradientDx01 / gradientCross;
        const float gradientY01OverCross =
            gradientDy01 / gradientCross;
        const float gradientX02OverCross =
            gradientDx02 / gradientCross;
        const float gradientY02OverCross =
            gradientDy02 / gradientCross;
        const auto channel = [](uint32_t color, size_t index)
        {
            return static_cast<float>(
                       (color >> (index * 8u)) & 0xFFu) *
                   128.0f;
        };
        for (size_t index = 0u; index < result.dx.size(); ++index)
        {
            const float top = channel(sorted[0].rgba, index);
            const float delta01 =
                channel(sorted[1].rgba, index) - top;
            const float delta02 =
                channel(sorted[2].rgba, index) - top;
            const float dx = std::fma(
                delta02,
                gradientY01OverCross,
                -(delta01 * gradientY02OverCross));
            const float dy = std::fma(
                delta01,
                gradientX02OverCross,
                -(delta02 * gradientX01OverCross));
            result.dx[index] = std::bit_cast<uint32_t>(dx);
            result.dy[index] = std::bit_cast<uint32_t>(dy);
        }
        return result;
    }

    const char *ct32TriangleValidationError(
        const GsVulkanCt32Triangle &triangle) noexcept
    {
        if (triangle.reserved0 != 0u || triangle.reserved1 != 0u)
            return "Vulkan CT32 triangle has non-zero reserved data";
        if (triangle.framebufferBaseBlock > 0x3FFFu)
            return "Vulkan CT32 triangle framebuffer base is outside GS VRAM";
        if (triangle.framebufferWidth == 0u ||
            triangle.framebufferWidth > 0x3Fu)
        {
            return "Vulkan CT32 triangle framebuffer width is outside FRAME range";
        }
        if (triangle.boundsX0 >= triangle.boundsX1 ||
            triangle.boundsY0 >= triangle.boundsY1)
        {
            return "Vulkan CT32 triangle bounds are empty";
        }
        if (triangle.boundsX1 > 2048u || triangle.boundsY1 > 2048u)
            return "Vulkan CT32 triangle bounds are outside GS scissor range";
        if ((triangle.topLeftEdgeMask & ~0x7u) != 0u)
            return "Vulkan CT32 triangle edge mask has reserved bits";

        constexpr int32_t kMinimumWindowFixed = -65535;
        constexpr int32_t kMaximumWindowFixed = 65535;
        const std::array<FixedTriangleVertex, 3> vertices{{
            {triangle.vertex0X12_4, triangle.vertex0Y12_4},
            {triangle.vertex1X12_4, triangle.vertex1Y12_4},
            {triangle.vertex2X12_4, triangle.vertex2Y12_4},
        }};
        for (const FixedTriangleVertex &vertex : vertices)
        {
            if (vertex.x < kMinimumWindowFixed ||
                vertex.x > kMaximumWindowFixed ||
                vertex.y < kMinimumWindowFixed ||
                vertex.y > kMaximumWindowFixed)
            {
                return "Vulkan CT32 triangle vertex is outside GS window range";
            }
        }

        if (triangleEdge(vertices[0], vertices[1], vertices[2]) <= 0)
            return "Vulkan CT32 triangle winding is not normalized";
        const uint32_t expectedTopLeftMask =
            (isTopLeftEdge(vertices[1], vertices[2]) ? 1u : 0u) |
            (isTopLeftEdge(vertices[2], vertices[0]) ? 2u : 0u) |
            (isTopLeftEdge(vertices[0], vertices[1]) ? 4u : 0u);
        if (triangle.topLeftEdgeMask != expectedTopLeftMask)
            return "Vulkan CT32 triangle edge mask is inconsistent";
        return nullptr;
    }

    const char *gouraudDepthCt32TriangleValidationError(
        const GsVulkanGouraudDepthCt32Triangle &triangle) noexcept
    {
        if (triangle.framebufferBaseBlock > 0x3FFFu ||
            triangle.depthBaseBlock > 0x3FFFu)
        {
            return "Vulkan Gouraud depth CT32 triangle base is outside GS VRAM";
        }
        if (triangle.framebufferWidth == 0u ||
            triangle.framebufferWidth > 0x3Fu)
        {
            return "Vulkan Gouraud depth CT32 triangle width is outside FRAME range";
        }
        if (triangle.boundsX0 >= triangle.boundsX1 ||
            triangle.boundsY0 >= triangle.boundsY1)
        {
            return "Vulkan Gouraud depth CT32 triangle bounds are empty";
        }
        if (triangle.boundsX1 > 2048u || triangle.boundsY1 > 2048u)
        {
            return "Vulkan Gouraud depth CT32 triangle bounds are outside GS scissor range";
        }
        if (triangle.depthPsm != GS_PSM_Z32 &&
            triangle.depthPsm != GS_PSM_Z24)
        {
            return "Vulkan Gouraud depth CT32 triangle has unsupported depth format";
        }
        if (triangle.positiveArea > 1u)
        {
            return "Vulkan Gouraud depth CT32 triangle has invalid winding data";
        }
        if ((triangle.topLeftEdgeMask & ~0x7u) != 0u)
        {
            return "Vulkan Gouraud depth CT32 triangle edge mask has reserved bits";
        }
        if ((triangle.rgba0 >> 24u) != 128u ||
            (triangle.rgba1 >> 24u) != 128u ||
            (triangle.rgba2 >> 24u) != 128u)
        {
            return "Vulkan Gouraud depth CT32 triangle alpha is not normalized";
        }
        if (!ct32RectangleHasUniqueWords(
                triangle.boundsX0,
                triangle.boundsY0,
                triangle.boundsX1,
                triangle.boundsY1,
                triangle.framebufferWidth))
        {
            return "Vulkan Gouraud depth CT32 triangle surface aliases itself";
        }

        constexpr int32_t kMinimumWindowFixed = -65535;
        constexpr int32_t kMaximumWindowFixed = 65535;
        const std::array<FixedTriangleVertex, 3> vertices{{
            {triangle.vertex0X12_4, triangle.vertex0Y12_4},
            {triangle.vertex1X12_4, triangle.vertex1Y12_4},
            {triangle.vertex2X12_4, triangle.vertex2Y12_4},
        }};
        for (const FixedTriangleVertex &vertex : vertices)
        {
            if (vertex.x < kMinimumWindowFixed ||
                vertex.x > kMaximumWindowFixed ||
                vertex.y < kMinimumWindowFixed ||
                vertex.y > kMaximumWindowFixed)
            {
                return "Vulkan Gouraud depth CT32 triangle vertex is outside GS window range";
            }
        }

        const int64_t signedArea =
            triangleEdge(vertices[0], vertices[1], vertices[2]);
        if (signedArea == 0 ||
            (signedArea > 0) != (triangle.positiveArea != 0u))
        {
            return "Vulkan Gouraud depth CT32 triangle winding is inconsistent";
        }
        const bool positiveArea = signedArea > 0;
        const auto orientedTopLeft =
            [positiveArea](
                const FixedTriangleVertex &a,
                const FixedTriangleVertex &b)
        {
            return positiveArea
                ? isTopLeftEdge(a, b)
                : isTopLeftEdge(b, a);
        };
        const uint32_t expectedTopLeftMask =
            (orientedTopLeft(vertices[1], vertices[2]) ? 1u : 0u) |
            (orientedTopLeft(vertices[2], vertices[0]) ? 2u : 0u) |
            (orientedTopLeft(vertices[0], vertices[1]) ? 4u : 0u);
        if (triangle.topLeftEdgeMask != expectedTopLeftMask)
        {
            return "Vulkan Gouraud depth CT32 triangle edge mask is inconsistent";
        }
        const GouraudColorDdaBits expectedColorDda =
            prepareGouraudColorDdaBits(
                vertices,
                {triangle.rgba0, triangle.rgba1, triangle.rgba2});
        if (triangle.colorDxBits != expectedColorDda.dx ||
            triangle.colorDyBits != expectedColorDda.dy)
        {
            return "Vulkan Gouraud depth CT32 triangle color DDA is inconsistent";
        }

        const uint32_t width = triangle.boundsX1 - triangle.boundsX0;
        const uint32_t height = triangle.boundsY1 - triangle.boundsY0;
        const GsVramPageMask framebufferPages =
            gsVramPagesForSurfaceRect(
                triangle.framebufferBaseBlock,
                triangle.framebufferWidth,
                static_cast<uint8_t>(GSMem::C32),
                triangle.boundsX0,
                triangle.boundsY0,
                width,
                height);
        const GsVramPageMask depthPages =
            gsVramPagesForSurfaceRect(
                triangle.depthBaseBlock,
                triangle.framebufferWidth,
                static_cast<uint8_t>(triangle.depthPsm),
                triangle.boundsX0,
                triangle.boundsY0,
                width,
                height);
        if (framebufferPages.intersects(depthPages))
        {
            return "Vulkan Gouraud depth CT32 triangle framebuffer aliases depth";
        }
        return nullptr;
    }

    template <typename Triangle>
    const char *t8GouraudDepthCt32TriangleValidationError(
        const Triangle &triangle,
        bool validateResourceViews = true,
        bool residentRecord = false) noexcept
    {
        if (triangle.framebufferBaseBlock > 0x3FFFu ||
            triangle.depthBaseBlock > 0x3FFFu)
        {
            return "Vulkan T8 Gouraud triangle surface base is outside GS VRAM";
        }
        if (triangle.framebufferWidth == 0u ||
            triangle.framebufferWidth > 0x3Fu)
        {
            return "Vulkan T8 Gouraud triangle framebuffer width is outside FRAME range";
        }
        if (triangle.boundsX0 >= triangle.boundsX1 ||
            triangle.boundsY0 >= triangle.boundsY1)
        {
            return "Vulkan T8 Gouraud triangle bounds are empty";
        }
        if (triangle.boundsX1 > 2048u || triangle.boundsY1 > 2048u)
            return "Vulkan T8 Gouraud triangle bounds are outside GS scissor range";
        if (triangle.depthPsm != GS_PSM_Z32 &&
            triangle.depthPsm != GS_PSM_Z24)
        {
            return "Vulkan T8 Gouraud triangle has unsupported depth format";
        }
        if (triangle.positiveArea > 1u ||
            (triangle.topLeftEdgeMask & ~0x7u) != 0u)
        {
            return "Vulkan T8 Gouraud triangle has invalid winding data";
        }
        if ((!residentRecord && triangle.paletteIndex != 0u) ||
            (triangle.rasterFlags &
             ~GS_VULKAN_T8_GOURAUD_VALID_FLAGS) != 0u ||
            triangle.reserved != 0u)
        {
            return "Vulkan T8 Gouraud triangle has non-zero reserved data";
        }
        const bool textureDisabled =
            (triangle.rasterFlags &
             GS_VULKAN_T8_GOURAUD_FLAG_TEXTURE_DISABLED) != 0u;
        const auto allZero = [](const auto &values)
        {
            return std::all_of(
                values.begin(), values.end(),
                [](const auto value) { return value == 0; });
        };
        if (textureDisabled &&
            (((triangle.rasterFlags &
               GS_VULKAN_T8_GOURAUD_FLAG_CONSTANT_Q_FIXED) != 0u) ||
             triangle.maximumMipLevel != 0u ||
             triangle.textureWidthLog2 != 0u ||
             triangle.textureHeightLog2 != 0u ||
             triangle.textureWrapU != 0u ||
             triangle.textureWrapV != 0u ||
             triangle.lodK != 0 || triangle.lodL != 0u ||
             !allZero(triangle.textureBaseBlocks) ||
             triangle.textureWidths[0] != 1u ||
             !std::all_of(
                 triangle.textureWidths.begin() + 1u,
                 triangle.textureWidths.end(),
                 [](uint32_t width) { return width == 0u; }) ||
             !allZero(triangle.sBits) ||
             !allZero(triangle.tBits) ||
             !std::all_of(
                 triangle.qBits.begin(), triangle.qBits.end(),
                 [](uint32_t bits)
                 {
                     return bits == std::bit_cast<uint32_t>(1.0f);
                 }) ||
             !allZero(triangle.textureDxBits) ||
             !allZero(triangle.textureDyBits)))
        {
            return "Vulkan untextured triangle has non-canonical texture data";
        }
        if (triangle.textureWidthLog2 > 10u ||
            triangle.textureHeightLog2 > 10u ||
            triangle.maximumMipLevel > 3u ||
            (((triangle.rasterFlags &
               GS_VULKAN_T8_GOURAUD_FLAG_CONSTANT_Q_FIXED) != 0u) &&
             triangle.maximumMipLevel != 0u) ||
            triangle.textureWrapU > 1u ||
            triangle.textureWrapV > 1u ||
            triangle.lodK < -2048 || triangle.lodK > 2047 ||
            triangle.lodL > 3u || triangle.alphaReference > 255u)
        {
            return "Vulkan T8 Gouraud triangle texture state is invalid";
        }
        for (uint32_t level = 0u;
             level <= triangle.maximumMipLevel;
             ++level)
        {
            if (triangle.textureBaseBlocks[level] > 0x3FFFu ||
                triangle.textureWidths[level] == 0u ||
                triangle.textureWidths[level] > 0x3Fu)
            {
                return "Vulkan T8 Gouraud triangle mip surface is invalid";
            }
        }

        constexpr int32_t kMinimumWindowFixed = -65535;
        constexpr int32_t kMaximumWindowFixed = 65535;
        std::array<FixedTriangleVertex, 3> vertices{};
        for (size_t index = 0u; index < vertices.size(); ++index)
        {
            vertices[index] = {
                triangle.vertexX12_4[index],
                triangle.vertexY12_4[index]};
            if (vertices[index].x < kMinimumWindowFixed ||
                vertices[index].x > kMaximumWindowFixed ||
                vertices[index].y < kMinimumWindowFixed ||
                vertices[index].y > kMaximumWindowFixed)
            {
                return "Vulkan T8 Gouraud triangle vertex is outside GS window range";
            }
        }
        const int64_t signedArea =
            triangleEdge(vertices[0], vertices[1], vertices[2]);
        if (signedArea == 0 ||
            (signedArea > 0) != (triangle.positiveArea != 0u))
        {
            return "Vulkan T8 Gouraud triangle winding is inconsistent";
        }
        const bool positiveArea = signedArea > 0;
        const auto orientedTopLeft =
            [positiveArea](
                const FixedTriangleVertex &a,
                const FixedTriangleVertex &b)
        {
            return positiveArea
                ? isTopLeftEdge(a, b)
                : isTopLeftEdge(b, a);
        };
        const uint32_t expectedTopLeftMask =
            (orientedTopLeft(vertices[1], vertices[2]) ? 1u : 0u) |
            (orientedTopLeft(vertices[2], vertices[0]) ? 2u : 0u) |
            (orientedTopLeft(vertices[0], vertices[1]) ? 4u : 0u);
        if (triangle.topLeftEdgeMask != expectedTopLeftMask)
            return "Vulkan T8 Gouraud triangle edge mask is inconsistent";

        const auto finiteFloatBits = [](uint32_t bits)
        {
            return std::isfinite(std::bit_cast<float>(bits));
        };
        if (!std::all_of(
                triangle.sBits.begin(), triangle.sBits.end(),
                finiteFloatBits) ||
            !std::all_of(
                triangle.tBits.begin(), triangle.tBits.end(),
                finiteFloatBits) ||
            !std::all_of(
                triangle.qBits.begin(), triangle.qBits.end(),
                finiteFloatBits) ||
            !std::all_of(
                triangle.colorDxBits.begin(),
                triangle.colorDxBits.end(), finiteFloatBits) ||
            !std::all_of(
                triangle.colorDyBits.begin(),
                triangle.colorDyBits.end(), finiteFloatBits) ||
            !finiteFloatBits(triangle.fogDxBits) ||
            !finiteFloatBits(triangle.fogDyBits) ||
            !std::all_of(
                triangle.textureDxBits.begin(),
                triangle.textureDxBits.end(), finiteFloatBits) ||
            !std::all_of(
                triangle.textureDyBits.begin(),
                triangle.textureDyBits.end(), finiteFloatBits))
        {
            return "Vulkan T8 Gouraud triangle contains non-finite setup data";
        }
        const auto unpackDouble = [](const std::array<uint32_t, 2> &bits)
        {
            const uint64_t packed =
                static_cast<uint64_t>(bits[0]) |
                (static_cast<uint64_t>(bits[1]) << 32u);
            return std::bit_cast<double>(packed);
        };
        if (!std::isfinite(unpackDouble(triangle.depthDxBits)) ||
            !std::isfinite(unpackDouble(triangle.depthDyBits)))
        {
            return "Vulkan T8 Gouraud triangle contains non-finite depth setup";
        }
        if (!ct32RectangleHasUniqueWords(
                triangle.boundsX0,
                triangle.boundsY0,
                triangle.boundsX1,
                triangle.boundsY1,
                triangle.framebufferWidth))
        {
            return "Vulkan T8 Gouraud triangle destination aliases itself";
        }
        if (!validateResourceViews)
            return nullptr;

        const uint32_t width = triangle.boundsX1 - triangle.boundsX0;
        const uint32_t height = triangle.boundsY1 - triangle.boundsY0;
        const GsVramPageMask framebufferPages =
            gsVramPagesForSurfaceRect(
                triangle.framebufferBaseBlock,
                triangle.framebufferWidth,
                GS_PSM_CT32,
                triangle.boundsX0,
                triangle.boundsY0,
                width,
                height);
        const GsVramPageMask depthPages =
            gsVramPagesForSurfaceRect(
                triangle.depthBaseBlock,
                triangle.framebufferWidth,
                static_cast<uint8_t>(triangle.depthPsm),
                triangle.boundsX0,
                triangle.boundsY0,
                width,
                height);
        if (framebufferPages.intersects(depthPages))
            return "Vulkan T8 Gouraud triangle color and depth alias";
        if (textureDisabled)
            return nullptr;
        GsVramPageMask texturePages;
        for (uint32_t level = 0u;
             level <= triangle.maximumMipLevel;
             ++level)
        {
            texturePages.unionWith(gsVramPagesForSurfaceRect(
                triangle.textureBaseBlocks[level],
                triangle.textureWidths[level],
                GS_PSM_T8,
                0u,
                0u,
                std::max(1u, (1u << triangle.textureWidthLog2) >> level),
                std::max(1u, (1u << triangle.textureHeightLog2) >> level)));
        }
        if (texturePages.intersects(framebufferPages) ||
            texturePages.intersects(depthPages))
        {
            return "Vulkan T8 Gouraud triangle texture aliases output";
        }
        return nullptr;
    }

    const char *feedbackNearestDepthCt32TriangleValidationError(
        const GsVulkanFeedbackNearestDepthCt32Triangle &triangle) noexcept
    {
        if (triangle.framebufferBaseBlock > 0x3FFFu ||
            triangle.depthBaseBlock > 0x3FFFu ||
            triangle.textureBaseBlock > 0x3FFFu)
        {
            return "Vulkan feedback nearest triangle surface base is outside GS VRAM";
        }
        if (triangle.framebufferWidth == 0u ||
            triangle.framebufferWidth > 0x3Fu ||
            triangle.textureWidth == 0u ||
            triangle.textureWidth > 0x3Fu)
        {
            return "Vulkan feedback nearest triangle surface width is invalid";
        }
        if (triangle.boundsX0 >= triangle.boundsX1 ||
            triangle.boundsY0 >= triangle.boundsY1 ||
            triangle.boundsX1 > 2048u || triangle.boundsY1 > 2048u)
        {
            return "Vulkan feedback nearest triangle bounds are invalid";
        }
        if (triangle.depthPsm != GS_PSM_Z32 &&
            triangle.depthPsm != GS_PSM_Z24)
        {
            return "Vulkan feedback nearest triangle depth format is invalid";
        }
        if (triangle.positiveArea > 1u ||
            (triangle.topLeftEdgeMask & ~0x7u) != 0u)
        {
            return "Vulkan feedback nearest triangle winding data is invalid";
        }
        if (triangle.textureWidthLog2 > 10u ||
            triangle.textureHeightLog2 > 10u ||
            triangle.textureWrapU > 1u ||
            triangle.textureWrapV > 1u)
        {
            return "Vulkan feedback nearest triangle texture state is invalid";
        }
        if (triangle.textureSource !=
                GS_VULKAN_TEXTURE_SOURCE_FEEDBACK_SNAPSHOT ||
            std::any_of(
                triangle.reserved.begin(), triangle.reserved.end(),
                [](uint32_t value) { return value != 0u; }))
        {
            return "Vulkan feedback nearest triangle snapshot state is invalid";
        }
        if (triangle.textureBaseBlock !=
                triangle.framebufferBaseBlock ||
            triangle.textureWidth != triangle.framebufferWidth)
        {
            return "Vulkan feedback nearest triangle does not name one exact recursive surface";
        }

        constexpr int32_t kMinimumWindowFixed = -65535;
        constexpr int32_t kMaximumWindowFixed = 65535;
        std::array<FixedTriangleVertex, 3> vertices{};
        for (size_t index = 0u; index < vertices.size(); ++index)
        {
            vertices[index] = {
                triangle.vertexX12_4[index],
                triangle.vertexY12_4[index]};
            if (vertices[index].x < kMinimumWindowFixed ||
                vertices[index].x > kMaximumWindowFixed ||
                vertices[index].y < kMinimumWindowFixed ||
                vertices[index].y > kMaximumWindowFixed)
            {
                return "Vulkan feedback nearest triangle vertex is outside GS window range";
            }
        }
        const int64_t signedArea =
            triangleEdge(vertices[0], vertices[1], vertices[2]);
        if (signedArea == 0 ||
            (signedArea > 0) != (triangle.positiveArea != 0u))
        {
            return "Vulkan feedback nearest triangle winding is inconsistent";
        }
        const bool positiveArea = signedArea > 0;
        const auto orientedTopLeft =
            [positiveArea](
                const FixedTriangleVertex &a,
                const FixedTriangleVertex &b)
        {
            return positiveArea
                ? isTopLeftEdge(a, b)
                : isTopLeftEdge(b, a);
        };
        const uint32_t expectedTopLeftMask =
            (orientedTopLeft(vertices[1], vertices[2]) ? 1u : 0u) |
            (orientedTopLeft(vertices[2], vertices[0]) ? 2u : 0u) |
            (orientedTopLeft(vertices[0], vertices[1]) ? 4u : 0u);
        if (triangle.topLeftEdgeMask != expectedTopLeftMask)
            return "Vulkan feedback nearest triangle edge mask is inconsistent";

        const auto finiteFloatBits = [](uint32_t bits)
        {
            return std::isfinite(std::bit_cast<float>(bits));
        };
        if (!std::all_of(
                triangle.sBits.begin(), triangle.sBits.end(),
                finiteFloatBits) ||
            !std::all_of(
                triangle.tBits.begin(), triangle.tBits.end(),
                finiteFloatBits) ||
            !std::all_of(
                triangle.textureDxBits.begin(),
                triangle.textureDxBits.end(), finiteFloatBits) ||
            !std::all_of(
                triangle.textureDyBits.begin(),
                triangle.textureDyBits.end(), finiteFloatBits))
        {
            return "Vulkan feedback nearest triangle contains non-finite texture setup";
        }
        const auto unpackDouble = [](const std::array<uint32_t, 2> &bits)
        {
            const uint64_t packed =
                static_cast<uint64_t>(bits[0]) |
                (static_cast<uint64_t>(bits[1]) << 32u);
            return std::bit_cast<double>(packed);
        };
        if (!std::isfinite(unpackDouble(triangle.depthDxBits)) ||
            !std::isfinite(unpackDouble(triangle.depthDyBits)))
        {
            return "Vulkan feedback nearest triangle contains non-finite depth setup";
        }
        if (!ct32RectangleHasUniqueWords(
                triangle.boundsX0,
                triangle.boundsY0,
                triangle.boundsX1,
                triangle.boundsY1,
                triangle.framebufferWidth))
        {
            return "Vulkan feedback nearest triangle destination aliases itself";
        }

        const uint32_t width = triangle.boundsX1 - triangle.boundsX0;
        const uint32_t height = triangle.boundsY1 - triangle.boundsY0;
        const GsVramPageMask framebufferPages =
            gsVramPagesForSurfaceRect(
                triangle.framebufferBaseBlock,
                triangle.framebufferWidth,
                GS_PSM_CT32,
                triangle.boundsX0,
                triangle.boundsY0,
                width,
                height);
        const GsVramPageMask depthPages =
            gsVramPagesForSurfaceRect(
                triangle.depthBaseBlock,
                triangle.framebufferWidth,
                static_cast<uint8_t>(triangle.depthPsm),
                triangle.boundsX0,
                triangle.boundsY0,
                width,
                height);
        if (framebufferPages.intersects(depthPages))
            return "Vulkan feedback nearest triangle color and depth alias";
        return nullptr;
    }

    bool validateResidentCt32SpriteBatch(
        std::span<const GsVulkanCt32Sprite> sprites,
        std::string &error)
    {
        if (sprites.empty() ||
            sprites.size() > GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH)
        {
            error = "Vulkan resident CT32 sprite batches require between 1 and " +
                    std::to_string(GS_VULKAN_MAX_RESIDENT_SPRITE_BATCH) +
                    " records";
            return false;
        }

        GsVramPageMask priorWritePages;
        for (size_t index = 0u; index < sprites.size(); ++index)
        {
            const GsVulkanCt32Sprite &sprite = sprites[index];
            if (const char *validationError =
                    ct32SpriteValidationError(sprite))
            {
                error = "Vulkan resident CT32 sprite " +
                        std::to_string(index) + ": " + validationError;
                return false;
            }

            const GsVramPageMask writePages =
                gsVramPagesForSurfaceRect(
                    sprite.framebufferBaseBlock,
                    sprite.framebufferWidth,
                    static_cast<uint8_t>(GSMem::C32),
                    sprite.x0,
                    sprite.y0,
                    sprite.x1 - sprite.x0,
                    sprite.y1 - sprite.y0);
            if (priorWritePages.intersects(writePages))
            {
                error = "Vulkan resident CT32 sprite " +
                        std::to_string(index) +
                        " overlaps an earlier batch member";
                return false;
            }
            priorWritePages.unionWith(writePages);
        }
        error.clear();
        return true;
    }

    bool validateResidentNearestCt32SpriteBatch(
        std::span<const GsVulkanNearestCt32Sprite> sprites,
        std::string &error)
    {
        if (sprites.empty() ||
            sprites.size() > GS_VULKAN_MAX_RESIDENT_NEAREST_CT32_BATCH)
        {
            error =
                "Vulkan resident nearest CT32 sprite batches require between 1 and " +
                std::to_string(
                    GS_VULKAN_MAX_RESIDENT_NEAREST_CT32_BATCH) +
                " records";
            return false;
        }

        GsVramPageMask priorReadPages;
        GsVramPageMask priorWritePages;
        for (size_t index = 0u; index < sprites.size(); ++index)
        {
            const GsVulkanNearestCt32Sprite &sprite = sprites[index];
            if (const char *validationError =
                    nearestCt32SpriteValidationError(sprite))
            {
                error = "Vulkan resident nearest CT32 sprite " +
                        std::to_string(index) + ": " + validationError;
                return false;
            }

            const GsVramPageMask readPages =
                nearestCt32TexturePages(sprite);
            const GsVramPageMask writePages =
                gsVramPagesForSurfaceRect(
                    sprite.framebufferBaseBlock,
                    sprite.framebufferWidth,
                    static_cast<uint8_t>(GSMem::C32),
                    sprite.boundsX0,
                    sprite.boundsY0,
                    sprite.boundsX1 - sprite.boundsX0,
                    sprite.boundsY1 - sprite.boundsY0);
            if (priorWritePages.intersects(readPages) ||
                priorWritePages.intersects(writePages) ||
                priorReadPages.intersects(writePages))
            {
                error = "Vulkan resident nearest CT32 sprite " +
                        std::to_string(index) +
                        " has a memory dependency on an earlier batch member";
                return false;
            }
            priorReadPages.unionWith(readPages);
            priorWritePages.unionWith(writePages);
        }
        error.clear();
        return true;
    }

    bool validateResidentDepthCt32SpriteBatch(
        std::span<const GsVulkanDepthCt32Sprite> sprites,
        std::string &error)
    {
        if (sprites.empty() ||
            sprites.size() > GS_VULKAN_MAX_RESIDENT_DEPTH_CT32_BATCH)
        {
            error =
                "Vulkan resident depth CT32 sprite batches require between 1 and " +
                std::to_string(
                    GS_VULKAN_MAX_RESIDENT_DEPTH_CT32_BATCH) +
                " records";
            return false;
        }

        for (size_t index = 0u; index < sprites.size(); ++index)
        {
            if (const char *validationError =
                    depthCt32SpriteValidationError(sprites[index]))
            {
                error = "Vulkan resident depth CT32 sprite " +
                        std::to_string(index) + ": " + validationError;
                return false;
            }
        }
        error.clear();
        return true;
    }

    bool validateResidentLinearCt32SpriteBatch(
        std::span<const GsVulkanLinearCt32Sprite> sprites,
        std::string &error)
    {
        if (sprites.empty() ||
            sprites.size() > GS_VULKAN_MAX_RESIDENT_LINEAR_CT32_BATCH)
        {
            error =
                "Vulkan resident linear CT32 sprite batches require between 1 and " +
                std::to_string(
                    GS_VULKAN_MAX_RESIDENT_LINEAR_CT32_BATCH) +
                " records";
            return false;
        }

        for (size_t index = 0u; index < sprites.size(); ++index)
        {
            const GsVulkanLinearCt32Sprite &sprite = sprites[index];
            if (const char *validationError =
                    linearCt32SpriteValidationError(sprite))
            {
                error = "Vulkan resident linear CT32 sprite " +
                        std::to_string(index) + ": " + validationError;
                return false;
            }
        }
        error.clear();
        return true;
    }

    bool validateResidentCt32TriangleBatch(
        std::span<const GsVulkanCt32Triangle> triangles,
        std::string &error)
    {
        if (triangles.empty() ||
            triangles.size() > GS_VULKAN_MAX_RESIDENT_TRIANGLE_BATCH)
        {
            error =
                "Vulkan resident CT32 triangle batches require between 1 and " +
                std::to_string(GS_VULKAN_MAX_RESIDENT_TRIANGLE_BATCH) +
                " records";
            return false;
        }

        GsVramPageMask priorWritePages;
        for (size_t index = 0u; index < triangles.size(); ++index)
        {
            const GsVulkanCt32Triangle &triangle = triangles[index];
            if (const char *validationError =
                    ct32TriangleValidationError(triangle))
            {
                error = "Vulkan resident CT32 triangle " +
                        std::to_string(index) + ": " + validationError;
                return false;
            }

            const GsVramPageMask writePages =
                gsVramPagesForSurfaceRect(
                    triangle.framebufferBaseBlock,
                    triangle.framebufferWidth,
                    static_cast<uint8_t>(GSMem::C32),
                    triangle.boundsX0,
                    triangle.boundsY0,
                    triangle.boundsX1 - triangle.boundsX0,
                    triangle.boundsY1 - triangle.boundsY0);
            if (priorWritePages.intersects(writePages))
            {
                error = "Vulkan resident CT32 triangle " +
                        std::to_string(index) +
                        " overlaps an earlier batch member";
                return false;
            }
            priorWritePages.unionWith(writePages);
        }
        error.clear();
        return true;
    }

    bool validateResidentT8GouraudDepthCt32TriangleBatch(
        std::span<const GsVulkanResidentT8GouraudDepthCt32Triangle> triangles,
        std::span<const GsVulkanT8Palette> palettes,
        bool validateRecords,
        bool validateResourceViews,
        std::string &error)
    {
        if (triangles.empty() ||
            triangles.size() > GS_VULKAN_MAX_RESIDENT_T8_TRIANGLE_BATCH)
        {
            error =
                "Vulkan resident T8 Gouraud depth CT32 triangle batches require between 1 and " +
                std::to_string(
                    GS_VULKAN_MAX_RESIDENT_T8_TRIANGLE_BATCH) +
                " records";
            return false;
        }

        for (size_t index = 0u; index < triangles.size(); ++index)
        {
            if (triangles[index].paletteIndex >= palettes.size())
            {
                error =
                    "Vulkan resident T8 Gouraud depth CT32 triangle " +
                    std::to_string(index) +
                    " has an invalid palette index";
                return false;
            }
            if (!validateRecords)
                continue;
            if (const char *validationError =
                    t8GouraudDepthCt32TriangleValidationError(
                        triangles[index], validateResourceViews, true))
            {
                error =
                    "Vulkan resident T8 Gouraud depth CT32 triangle " +
                    std::to_string(index) + ": " + validationError;
                return false;
            }
        }
        error.clear();
        return true;
    }

    bool validateResidentFeedbackNearestDepthCt32TriangleBatch(
        std::span<
            const GsVulkanFeedbackNearestDepthCt32Triangle> triangles,
        std::string &error)
    {
        if (triangles.empty() ||
            triangles.size() >
                GS_VULKAN_MAX_RESIDENT_FEEDBACK_NEAREST_DEPTH_CT32_TRIANGLE_BATCH)
        {
            error =
                "Vulkan resident feedback nearest depth CT32 triangle batches require between 1 and " +
                std::to_string(
                    GS_VULKAN_MAX_RESIDENT_FEEDBACK_NEAREST_DEPTH_CT32_TRIANGLE_BATCH) +
                " records";
            return false;
        }
        for (size_t index = 0u; index < triangles.size(); ++index)
        {
            if (const char *validationError =
                    feedbackNearestDepthCt32TriangleValidationError(
                        triangles[index]))
            {
                error =
                    "Vulkan resident feedback nearest depth CT32 triangle " +
                    std::to_string(index) + ": " + validationError;
                return false;
            }
        }
        error.clear();
        return true;
    }

    bool validateResidentGouraudDepthCt32TriangleBatch(
        std::span<const GsVulkanGouraudDepthCt32Triangle> triangles,
        std::string &error)
    {
        if (triangles.empty() ||
            triangles.size() >
                GS_VULKAN_MAX_RESIDENT_GOURAUD_DEPTH_CT32_TRIANGLE_BATCH)
        {
            error =
                "Vulkan resident Gouraud depth CT32 triangle batches require between 1 and " +
                std::to_string(
                    GS_VULKAN_MAX_RESIDENT_GOURAUD_DEPTH_CT32_TRIANGLE_BATCH) +
                " records";
            return false;
        }
        for (size_t index = 0u; index < triangles.size(); ++index)
        {
            if (const char *validationError =
                    gouraudDepthCt32TriangleValidationError(
                        triangles[index]))
            {
                error = "Vulkan resident Gouraud depth CT32 triangle " +
                        std::to_string(index) + ": " + validationError;
                return false;
            }
        }
        error.clear();
        return true;
    }

    struct GsVulkanTriangleTileWorkPlan
    {
        std::vector<uint32_t> words;
        uint32_t tileCount = 0u;
        uint32_t indexOffsetWords = 0u;
        uint64_t triangleReferences = 0u;
    };

    template <typename Triangle>
    bool sameTriangleRenderTarget(
        const Triangle &left,
        const Triangle &right) noexcept
    {
        return
            left.framebufferBaseBlock == right.framebufferBaseBlock &&
            left.framebufferWidth == right.framebufferWidth &&
            left.depthBaseBlock == right.depthBaseBlock &&
            left.depthPsm == right.depthPsm;
    }

    template <typename Triangle>
    bool buildTriangleTileWork(
        std::span<const Triangle> triangles,
        GsVulkanTriangleTileWorkPlan &plan)
    {
        constexpr uint32_t tileSize = 8u;
        constexpr size_t capacityWords =
            GS_VULKAN_AUXILIARY_STORAGE_SIZE / sizeof(uint32_t);
        if (triangles.empty())
            return false;
        for (size_t index = 1u; index < triangles.size(); ++index)
        {
            if (!sameTriangleRenderTarget(
                    triangles.front(), triangles[index]))
                return false;
        }

        uint32_t minimumTileX = UINT32_MAX;
        uint32_t minimumTileY = UINT32_MAX;
        uint32_t maximumTileX = 0u;
        uint32_t maximumTileY = 0u;
        for (const auto &triangle : triangles)
        {
            minimumTileX = std::min(minimumTileX, triangle.boundsX0 / tileSize);
            minimumTileY = std::min(minimumTileY, triangle.boundsY0 / tileSize);
            maximumTileX = std::max(
                maximumTileX, (triangle.boundsX1 - 1u) / tileSize);
            maximumTileY = std::max(
                maximumTileY, (triangle.boundsY1 - 1u) / tileSize);
        }
        const uint32_t gridWidth = maximumTileX - minimumTileX + 1u;
        const uint32_t gridHeight = maximumTileY - minimumTileY + 1u;
        std::vector<uint32_t> tileCounts(
            static_cast<size_t>(gridWidth) * gridHeight, 0u);
        const auto tileIndex = [&](uint32_t tileX, uint32_t tileY)
        {
            return static_cast<size_t>(tileY - minimumTileY) * gridWidth +
                   (tileX - minimumTileX);
        };

        uint64_t referenceCount = 0u;
        for (size_t index = 0u; index < triangles.size(); ++index)
        {
            const auto &triangle = triangles[index];
            const uint32_t tileX0 = triangle.boundsX0 / tileSize;
            const uint32_t tileY0 = triangle.boundsY0 / tileSize;
            const uint32_t tileX1 =
                (triangle.boundsX1 - 1u) / tileSize;
            const uint32_t tileY1 =
                (triangle.boundsY1 - 1u) / tileSize;
            const uint64_t triangleTileCount =
                static_cast<uint64_t>(tileX1 - tileX0 + 1u) *
                static_cast<uint64_t>(tileY1 - tileY0 + 1u);
            referenceCount += triangleTileCount;
            if (referenceCount > capacityWords)
                return false;
            for (uint32_t tileY = tileY0; tileY <= tileY1; ++tileY)
            {
                for (uint32_t tileX = tileX0; tileX <= tileX1; ++tileX)
                {
                    const uint64_t key =
                        tileIndex(tileX, tileY);
                    ++tileCounts[static_cast<size_t>(key)];
                }
            }
        }

        constexpr size_t descriptorWords = 4u;
        const size_t tileCount = static_cast<size_t>(std::count_if(
            tileCounts.begin(), tileCounts.end(),
            [](uint32_t count) { return count != 0u; }));
        const size_t indexOffsetWords = tileCount * descriptorWords;
        if (tileCount == 0u ||
            indexOffsetWords + referenceCount > capacityWords ||
            tileCount >
                static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        {
            return false;
        }

        GsVulkanTriangleTileWorkPlan candidate;
        candidate.words.resize(indexOffsetWords + referenceCount);
        candidate.tileCount =
            static_cast<uint32_t>(tileCount);
        candidate.indexOffsetWords =
            static_cast<uint32_t>(indexOffsetWords);
        candidate.triangleReferences = referenceCount;
        size_t descriptor = 0u;
        uint32_t first = 0u;
        for (uint32_t tileY = minimumTileY;
             tileY <= maximumTileY; ++tileY)
        {
            for (uint32_t tileX = minimumTileX;
                 tileX <= maximumTileX; ++tileX)
            {
                const size_t index = tileIndex(tileX, tileY);
                const uint32_t count = tileCounts[index];
                if (count == 0u)
                    continue;
                candidate.words[descriptor] = tileX * tileSize;
                candidate.words[descriptor + 1u] = tileY * tileSize;
                candidate.words[descriptor + 2u] = first;
                candidate.words[descriptor + 3u] = count;
                tileCounts[index] =
                    static_cast<uint32_t>(indexOffsetWords) + first;
                first += count;
                descriptor += descriptorWords;
            }
        }
        for (size_t triangleIndex = 0u;
             triangleIndex < triangles.size(); ++triangleIndex)
        {
            const auto &triangle = triangles[triangleIndex];
            const uint32_t tileX0 = triangle.boundsX0 / tileSize;
            const uint32_t tileY0 = triangle.boundsY0 / tileSize;
            const uint32_t tileX1 =
                (triangle.boundsX1 - 1u) / tileSize;
            const uint32_t tileY1 =
                (triangle.boundsY1 - 1u) / tileSize;
            for (uint32_t tileY = tileY0; tileY <= tileY1; ++tileY)
            {
                for (uint32_t tileX = tileX0; tileX <= tileX1; ++tileX)
                {
                    uint32_t &write = tileCounts[tileIndex(tileX, tileY)];
                    candidate.words[write++] =
                        static_cast<uint32_t>(triangleIndex);
                }
            }
        }
        plan = std::move(candidate);
        return true;
    }

    struct GsVulkanT8TriangleParameters
    {
        uint32_t recordIndex = 0u;
        uint32_t tileIndexWords = 0u;
        uint32_t paletteIndexWords = 0u;
    };

    static_assert(sizeof(GsVulkanT8TriangleParameters) == 12u);

    struct GsVulkanGouraudTriangleParameters
    {
        uint32_t recordIndex = 0u;
        uint32_t tileIndexWords = 0u;
    };

    static_assert(sizeof(GsVulkanGouraudTriangleParameters) == 8u);

    GsBackendDecision populateLinearCt32SpriteRecord(
        const GsDrawCommand &command,
        bool exactFramebufferFeedback,
        GsVulkanLinearCt32Sprite &sprite) noexcept
    {
        const GSContext &context = command.context();
        const GsDrawBounds &bounds = command.bounds();
        int32_t fixedX0 = command.fixedX()[0];
        int32_t fixedX1 = command.fixedX()[1];
        int32_t fixedY0 = command.fixedY()[0];
        int32_t fixedY1 = command.fixedY()[1];
        float textureU0 =
            static_cast<float>(command.vertices()[0].u) / 16.0f;
        float textureU1 =
            static_cast<float>(command.vertices()[1].u) / 16.0f;
        float textureV0 =
            static_cast<float>(command.vertices()[0].v) / 16.0f;
        float textureV1 =
            static_cast<float>(command.vertices()[1].v) / 16.0f;
        if (fixedX0 > fixedX1)
        {
            std::swap(fixedX0, fixedX1);
            std::swap(textureU0, textureU1);
        }
        if (fixedY0 > fixedY1)
        {
            std::swap(fixedY0, fixedY1);
            std::swap(textureV0, textureV1);
        }

        constexpr float kLinearBias = 32768.0f;
        const float windowX0 = static_cast<float>(fixedX0) / 16.0f;
        const float windowY0 = static_cast<float>(fixedY0) / 16.0f;
        const float spriteWidth =
            static_cast<float>(fixedX1 - fixedX0) / 16.0f;
        const float spriteHeight =
            static_cast<float>(fixedY1 - fixedY0) / 16.0f;
        const float fixedU0 = textureU0 * 65536.0f - kLinearBias;
        const float fixedV0 = textureV0 * 65536.0f - kLinearBias;
        const float fixedU1 = textureU1 * 65536.0f - kLinearBias;
        const float fixedV1 = textureV1 * 65536.0f - kLinearBias;
        const float fixedStepU = (fixedU1 - fixedU0) / spriteWidth;
        const float fixedStepV = (fixedV1 - fixedV0) / spriteHeight;
        const float fixedPrestepX =
            static_cast<float>(bounds.x0) - windowX0;
        const float fixedPrestepY =
            static_cast<float>(bounds.y0) - windowY0;
        const float fixedScanU = fixedU0 + fixedStepU * fixedPrestepX;
        const float fixedScanV = fixedV0 + fixedStepV * fixedPrestepY;
        constexpr int kPixelsPerLaneGroup = 8;
        const int laneSkip = bounds.x0 & (kPixelsPerLaneGroup - 1);
        const float fixedBlockStepU =
            fixedStepU * static_cast<float>(kPixelsPerLaneGroup);
        if (!linearDdaFloatFitsInt32(fixedScanU) ||
            !linearDdaFloatFitsInt32(fixedBlockStepU) ||
            !std::isfinite(fixedStepV))
        {
            return {
                false,
                GsFallbackReason::UnsupportedTextureCoordinates};
        }

        GsVulkanLinearCt32Sprite prepared{};
        prepared.framebufferBaseBlock = context.frame.fbp << 5u;
        prepared.framebufferWidth =
            std::max<uint32_t>(context.frame.fbw, 1u);
        prepared.boundsX0 = static_cast<uint32_t>(bounds.x0);
        prepared.boundsY0 = static_cast<uint32_t>(bounds.y0);
        prepared.boundsX1 = static_cast<uint32_t>(bounds.x1);
        prepared.boundsY1 = static_cast<uint32_t>(bounds.y1);
        prepared.textureBaseBlock = context.tex0.tbp0;
        prepared.textureWidth = context.tex0.tbw;
        prepared.textureMaskU = (1u << context.tex0.tw) - 1u;
        prepared.textureMaskV = (1u << context.tex0.th) - 1u;
        prepared.fixedBaseU = static_cast<int32_t>(fixedScanU);
        prepared.fixedBlockStepU =
            static_cast<int32_t>(fixedBlockStepU);
        for (int lane = 0; lane < kPixelsPerLaneGroup; ++lane)
        {
            const float fixedLaneU = fixedStepU *
                static_cast<float>(lane - laneSkip);
            if (!linearDdaFloatFitsInt32(fixedLaneU))
            {
                return {
                    false,
                    GsFallbackReason::UnsupportedTextureCoordinates};
            }
            prepared.fixedLaneU[static_cast<size_t>(lane)] =
                static_cast<int32_t>(fixedLaneU);
        }
        prepared.fixedScanVBits = std::bit_cast<uint32_t>(fixedScanV);
        prepared.fixedStepVBits = std::bit_cast<uint32_t>(fixedStepV);
        prepared.textureWrapU = packGsVulkanTextureWrap(
            static_cast<uint8_t>(context.clamp & 0x3u),
            static_cast<uint16_t>((context.clamp >> 4u) & 0x3FFu),
            static_cast<uint16_t>((context.clamp >> 14u) & 0x3FFu));
        prepared.textureWrapV = packGsVulkanTextureWrap(
            static_cast<uint8_t>((context.clamp >> 2u) & 0x3u),
            static_cast<uint16_t>((context.clamp >> 24u) & 0x3FFu),
            static_cast<uint16_t>((context.clamp >> 34u) & 0x3FFu));
        if (linearCt32SpriteRecordValidationError(
                prepared, exactFramebufferFeedback))
        {
            return {false, GsFallbackReason::UnknownMemoryLayout};
        }

        sprite = prepared;
        return {true, GsFallbackReason::Supported};
    }

    GsVulkanLinearCt32Sprite feedbackLinearPart(
        const GsVulkanFeedbackLinearDepthCt32Sprite &sprite) noexcept
    {
        return {
            sprite.framebufferBaseBlock,
            sprite.framebufferWidth,
            sprite.boundsX0,
            sprite.boundsY0,
            sprite.boundsX1,
            sprite.boundsY1,
            sprite.textureBaseBlock,
            sprite.textureWidth,
            sprite.textureMaskU,
            sprite.textureMaskV,
            sprite.fixedBaseU,
            sprite.fixedBlockStepU,
            sprite.fixedLaneU,
            sprite.fixedScanVBits,
            sprite.fixedStepVBits,
            sprite.textureWrapU,
            sprite.textureWrapV,
        };
    }

    const char *feedbackLinearDepthCt32SpriteValidationError(
        const GsVulkanFeedbackLinearDepthCt32Sprite &sprite) noexcept
    {
        if ((sprite.reserved0 | sprite.reserved1) != 0u)
        {
            return "Vulkan feedback linear depth CT32 sprite has non-zero reserved data";
        }
        if (sprite.textureSource !=
            GS_VULKAN_TEXTURE_SOURCE_FEEDBACK_SNAPSHOT)
        {
            return "Vulkan feedback linear depth CT32 sprite requires a snapshot source";
        }
        const GsVulkanLinearCt32Sprite linear =
            feedbackLinearPart(sprite);
        if (const char *error =
                linearCt32SpriteRecordValidationError(linear, true))
        {
            return error;
        }
        const GsVulkanDepthCt32Sprite depth{
            sprite.framebufferBaseBlock,
            sprite.framebufferWidth,
            sprite.depthBaseBlock,
            sprite.depthPsm,
            sprite.boundsX0,
            sprite.boundsY0,
            sprite.boundsX1,
            sprite.boundsY1,
            0u,
            sprite.depth,
            sprite.depthTestMethod,
            sprite.depthWrite,
            GS_VULKAN_DEPTH_CT32_COLOR_SOURCE_COPY,
            0u,
            0u,
            0u,
        };
        return depthCt32SpriteShapeValidationError(depth);
    }

    DepthCt32SpriteAccessPages
    feedbackLinearDepthCt32SpriteAccessPages(
        const GsVulkanFeedbackLinearDepthCt32Sprite &sprite) noexcept
    {
        const GsVulkanDepthCt32Sprite depth{
            sprite.framebufferBaseBlock,
            sprite.framebufferWidth,
            sprite.depthBaseBlock,
            sprite.depthPsm,
            sprite.boundsX0,
            sprite.boundsY0,
            sprite.boundsX1,
            sprite.boundsY1,
            0u,
            sprite.depth,
            sprite.depthTestMethod,
            sprite.depthWrite,
            GS_VULKAN_DEPTH_CT32_COLOR_SOURCE_COPY,
            0u,
            0u,
            0u,
        };
        return depthCt32SpriteAccessPages(depth);
    }

    bool validateResidentFeedbackLinearDepthCt32SpriteBatch(
        std::span<const GsVulkanFeedbackLinearDepthCt32Sprite> sprites,
        std::string &error)
    {
        if (sprites.empty() ||
            sprites.size() >
                GS_VULKAN_MAX_RESIDENT_FEEDBACK_LINEAR_DEPTH_CT32_BATCH)
        {
            error =
                "Vulkan resident feedback linear depth CT32 sprite batches require between 1 and " +
                std::to_string(
                    GS_VULKAN_MAX_RESIDENT_FEEDBACK_LINEAR_DEPTH_CT32_BATCH) +
                " records";
            return false;
        }

        for (size_t index = 0u; index < sprites.size(); ++index)
        {
            if (const char *validationError =
                    feedbackLinearDepthCt32SpriteValidationError(
                        sprites[index]))
            {
                error =
                    "Vulkan resident feedback linear depth CT32 sprite " +
                    std::to_string(index) + ": " + validationError;
                return false;
            }
        }
        error.clear();
        return true;
    }

    struct GsVulkanElementParameters
    {
        uint32_t elementCount;
        uint32_t reserved;
    };

    static_assert(sizeof(GsVulkanElementParameters) == 8u);
}

GsBackendDecision prepareGsVulkanCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanCt32Sprite &sprite) noexcept
{
    const GsBackendDecision decision =
        classifyGsInitialCt32Sprite(command);
    if (!decision.supported)
        return decision;

    const GSContext &context = command.context();
    const GsDrawBounds &bounds = command.bounds();
    const GSVertex &color = command.vertices()[1];
    GsVulkanCt32Sprite prepared{};
    prepared.framebufferBaseBlock = context.frame.fbp << 5u;
    prepared.framebufferWidth =
        std::max<uint32_t>(context.frame.fbw, 1u);
    prepared.x0 = static_cast<uint32_t>(bounds.x0);
    prepared.y0 = static_cast<uint32_t>(bounds.y0);
    prepared.x1 = static_cast<uint32_t>(bounds.x1);
    prepared.y1 = static_cast<uint32_t>(bounds.y1);
    prepared.rgba =
        static_cast<uint32_t>(color.r) |
        (static_cast<uint32_t>(color.g) << 8u) |
        (static_cast<uint32_t>(color.b) << 16u) |
        (static_cast<uint32_t>(color.a) << 24u);
    if (ct32SpriteValidationError(prepared))
    {
        return {false, GsFallbackReason::UnknownMemoryLayout};
    }

    sprite = prepared;
    return decision;
}

GsBackendDecision prepareGsVulkanDepthCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanDepthCt32Sprite &sprite) noexcept
{
    const GsBackendDecision decision =
        classifyGsDepthCt32Sprite(command);
    if (!decision.supported)
        return decision;

    const GSContext &context = command.context();
    const GsDrawBounds &bounds = command.bounds();
    const GSVertex &vertex = command.vertices()[1];
    GsVulkanDepthCt32Sprite prepared{};
    prepared.framebufferBaseBlock = context.frame.fbp << 5u;
    prepared.framebufferWidth =
        std::max<uint32_t>(context.frame.fbw, 1u);
    prepared.depthBaseBlock = context.zbuf.zbp << 5u;
    prepared.depthPsm = context.zbuf.psm;
    prepared.boundsX0 = static_cast<uint32_t>(bounds.x0);
    prepared.boundsY0 = static_cast<uint32_t>(bounds.y0);
    prepared.boundsX1 = static_cast<uint32_t>(bounds.x1);
    prepared.boundsY1 = static_cast<uint32_t>(bounds.y1);
    prepared.rgba =
        static_cast<uint32_t>(vertex.r) |
        (static_cast<uint32_t>(vertex.g) << 8u) |
        (static_cast<uint32_t>(vertex.b) << 16u) |
        (static_cast<uint32_t>(vertex.a) << 24u);
    prepared.depth = vertex.zInteger;
    prepared.depthTestMethod =
        static_cast<uint32_t>((context.test >> 17u) & 0x3u);
    prepared.depthWrite = context.zbuf.zmask ? 0u : 1u;
    prepared.colorBlendMode =
        GS_VULKAN_DEPTH_CT32_COLOR_SOURCE_COPY;
    if (depthCt32SpriteValidationError(prepared))
        return {false, GsFallbackReason::UnknownMemoryLayout};

    sprite = prepared;
    return decision;
}

GsBackendDecision prepareGsVulkanSourceOverCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanDepthCt32Sprite &sprite) noexcept
{
    const GsBackendDecision decision =
        classifyGsSourceOverCt32Sprite(command);
    if (!decision.supported)
        return decision;

    const GSContext &context = command.context();
    const GsDrawBounds &bounds = command.bounds();
    const GSVertex &vertex = command.vertices()[1];
    GsVulkanDepthCt32Sprite prepared{};
    prepared.framebufferBaseBlock = context.frame.fbp << 5u;
    prepared.framebufferWidth =
        std::max<uint32_t>(context.frame.fbw, 1u);
    prepared.boundsX0 = static_cast<uint32_t>(bounds.x0);
    prepared.boundsY0 = static_cast<uint32_t>(bounds.y0);
    prepared.boundsX1 = static_cast<uint32_t>(bounds.x1);
    prepared.boundsY1 = static_cast<uint32_t>(bounds.y1);
    prepared.rgba =
        static_cast<uint32_t>(vertex.r) |
        (static_cast<uint32_t>(vertex.g) << 8u) |
        (static_cast<uint32_t>(vertex.b) << 16u) |
        (static_cast<uint32_t>(vertex.a) << 24u);
    prepared.colorBlendMode =
        GS_VULKAN_DEPTH_CT32_COLOR_SOURCE_OVER;
    if (depthCt32SpriteValidationError(prepared))
        return {false, GsFallbackReason::UnknownMemoryLayout};

    sprite = prepared;
    return decision;
}

GsBackendDecision prepareGsVulkanSourceOverDepthCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanDepthCt32Sprite &sprite) noexcept
{
    const GsBackendDecision decision =
        classifyGsSourceOverDepthCt32Sprite(command);
    if (!decision.supported)
        return decision;

    const GSContext &context = command.context();
    const GsDrawBounds &bounds = command.bounds();
    const GSVertex &vertex = command.vertices()[1];
    GsVulkanDepthCt32Sprite prepared{};
    prepared.framebufferBaseBlock = context.frame.fbp << 5u;
    prepared.framebufferWidth =
        std::max<uint32_t>(context.frame.fbw, 1u);
    prepared.depthBaseBlock = context.zbuf.zbp << 5u;
    prepared.depthPsm = context.zbuf.psm;
    prepared.boundsX0 = static_cast<uint32_t>(bounds.x0);
    prepared.boundsY0 = static_cast<uint32_t>(bounds.y0);
    prepared.boundsX1 = static_cast<uint32_t>(bounds.x1);
    prepared.boundsY1 = static_cast<uint32_t>(bounds.y1);
    prepared.rgba =
        static_cast<uint32_t>(vertex.r) |
        (static_cast<uint32_t>(vertex.g) << 8u) |
        (static_cast<uint32_t>(vertex.b) << 16u) |
        (static_cast<uint32_t>(vertex.a) << 24u);
    prepared.depth = vertex.zInteger;
    prepared.depthTestMethod =
        static_cast<uint32_t>((context.test >> 17u) & 0x3u);
    prepared.depthWrite = context.zbuf.zmask ? 0u : 1u;
    prepared.colorBlendMode =
        GS_VULKAN_DEPTH_CT32_COLOR_SOURCE_OVER;
    if (depthCt32SpriteShapeValidationError(prepared))
        return {false, GsFallbackReason::UnknownMemoryLayout};

    sprite = prepared;
    return decision;
}

GsBackendDecision
prepareGsVulkanFramebufferOnlyAlphaFailDepthCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanDepthCt32Sprite &sprite) noexcept
{
    const GsBackendDecision decision =
        classifyGsFramebufferOnlyAlphaFailDepthCt32Sprite(command);
    if (!decision.supported)
        return decision;

    const GSContext &context = command.context();
    const GsDrawBounds &bounds = command.bounds();
    const GSVertex &vertex = command.vertices()[1];
    GsVulkanDepthCt32Sprite prepared{};
    prepared.framebufferBaseBlock = context.frame.fbp << 5u;
    prepared.framebufferWidth =
        std::max<uint32_t>(context.frame.fbw, 1u);
    prepared.depthBaseBlock = context.zbuf.zbp << 5u;
    prepared.depthPsm = context.zbuf.psm;
    prepared.boundsX0 = static_cast<uint32_t>(bounds.x0);
    prepared.boundsY0 = static_cast<uint32_t>(bounds.y0);
    prepared.boundsX1 = static_cast<uint32_t>(bounds.x1);
    prepared.boundsY1 = static_cast<uint32_t>(bounds.y1);
    prepared.rgba =
        static_cast<uint32_t>(vertex.r) |
        (static_cast<uint32_t>(vertex.g) << 8u) |
        (static_cast<uint32_t>(vertex.b) << 16u) |
        (static_cast<uint32_t>(vertex.a) << 24u);
    prepared.depth = vertex.zInteger;
    prepared.depthTestMethod = 1u;
    // A failed alpha test with AFAIL=FB_ONLY suppresses depth writes even
    // when ZMASK is clear.
    prepared.depthWrite = 0u;
    const uint32_t fixedAlpha = static_cast<uint32_t>(
        (context.alpha >> 32u) & 0xFFu);
    prepared.colorBlendMode =
        GS_VULKAN_DEPTH_CT32_COLOR_FIXED_ALPHA |
        (fixedAlpha <<
         GS_VULKAN_DEPTH_CT32_COLOR_FIXED_ALPHA_SHIFT);
    if (((context.test >> 14u) & 1u) != 0u)
    {
        prepared.colorBlendMode |= GS_VULKAN_DEPTH_CT32_COLOR_DATE;
        if (((context.test >> 15u) & 1u) != 0u)
            prepared.colorBlendMode |= GS_VULKAN_DEPTH_CT32_COLOR_DATM;
    }
    if (depthCt32SpriteValidationError(prepared))
        return {false, GsFallbackReason::UnknownMemoryLayout};

    sprite = prepared;
    return decision;
}

GsBackendDecision prepareGsVulkanNearestCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanNearestCt32Sprite &sprite) noexcept
{
    const GsBackendDecision decision =
        classifyGsNearestCt32TexturedSprite(command);
    if (!decision.supported)
        return decision;

    const GSContext &context = command.context();
    const GsDrawBounds &bounds = command.bounds();
    int32_t fixedX0 = command.fixedX()[0];
    int32_t fixedX1 = command.fixedX()[1];
    int32_t fixedY0 = command.fixedY()[0];
    int32_t fixedY1 = command.fixedY()[1];
    int32_t textureU0 = command.vertices()[0].u;
    int32_t textureU1 = command.vertices()[1].u;
    int32_t textureV0 = command.vertices()[0].v;
    int32_t textureV1 = command.vertices()[1].v;
    if (fixedX0 > fixedX1)
    {
        std::swap(fixedX0, fixedX1);
        std::swap(textureU0, textureU1);
    }
    if (fixedY0 > fixedY1)
    {
        std::swap(fixedY0, fixedY1);
        std::swap(textureV0, textureV1);
    }

    const int32_t textureStepU = textureU1 > textureU0 ? 1 : -1;
    const int32_t textureStepV = textureV1 > textureV0 ? 1 : -1;
    const int32_t unclippedX0 = fixedX0 / 16;
    const int32_t unclippedY0 = fixedY0 / 16;

    GsVulkanNearestCt32Sprite prepared{};
    prepared.framebufferBaseBlock = context.frame.fbp << 5u;
    prepared.framebufferWidth =
        std::max<uint32_t>(context.frame.fbw, 1u);
    prepared.boundsX0 = static_cast<uint32_t>(bounds.x0);
    prepared.boundsY0 = static_cast<uint32_t>(bounds.y0);
    prepared.boundsX1 = static_cast<uint32_t>(bounds.x1);
    prepared.boundsY1 = static_cast<uint32_t>(bounds.y1);
    prepared.textureBaseBlock = context.tex0.tbp0;
    prepared.textureWidth = context.tex0.tbw;
    prepared.textureMaskU = (1u << context.tex0.tw) - 1u;
    prepared.textureMaskV = (1u << context.tex0.th) - 1u;
    prepared.textureOriginU =
        textureU0 / 16 +
        textureStepU * (bounds.x0 - unclippedX0);
    prepared.textureOriginV =
        textureV0 / 16 +
        textureStepV * (bounds.y0 - unclippedY0);
    prepared.textureStepU = textureStepU;
    prepared.textureStepV = textureStepV;
    prepared.textureWrapU = packGsVulkanTextureWrap(
        static_cast<uint8_t>(context.clamp & 0x3u),
        static_cast<uint16_t>((context.clamp >> 4u) & 0x3FFu),
        static_cast<uint16_t>((context.clamp >> 14u) & 0x3FFu));
    prepared.textureWrapV = packGsVulkanTextureWrap(
        static_cast<uint8_t>((context.clamp >> 2u) & 0x3u),
        static_cast<uint16_t>((context.clamp >> 24u) & 0x3FFu),
        static_cast<uint16_t>((context.clamp >> 34u) & 0x3FFu));
    if (nearestCt32SpriteValidationError(prepared))
        return {false, GsFallbackReason::UnknownMemoryLayout};

    sprite = prepared;
    return decision;
}

GsBackendDecision prepareGsVulkanLinearCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanLinearCt32Sprite &sprite) noexcept
{
    const GsBackendDecision decision =
        classifyGsLinearCt32TexturedSprite(command);
    if (!decision.supported)
        return decision;
    const GsBackendDecision recordDecision =
        populateLinearCt32SpriteRecord(command, false, sprite);
    return recordDecision.supported ? decision : recordDecision;
}

GsBackendDecision prepareGsVulkanFeedbackLinearDepthCt32Sprite(
    const GsDrawCommand &command,
    GsVulkanFeedbackLinearDepthCt32Sprite &sprite) noexcept
{
    const GsBackendDecision decision =
        classifyGsFeedbackLinearDepthCt32Sprite(command);
    if (!decision.supported)
        return decision;

    GsVulkanLinearCt32Sprite linear{};
    const GsBackendDecision linearDecision =
        populateLinearCt32SpriteRecord(command, true, linear);
    if (!linearDecision.supported)
        return linearDecision;

    const GSContext &context = command.context();
    const GSVertex &vertex = command.vertices()[1];
    GsVulkanFeedbackLinearDepthCt32Sprite prepared{
        linear.framebufferBaseBlock,
        linear.framebufferWidth,
        linear.boundsX0,
        linear.boundsY0,
        linear.boundsX1,
        linear.boundsY1,
        linear.textureBaseBlock,
        linear.textureWidth,
        linear.textureMaskU,
        linear.textureMaskV,
        linear.fixedBaseU,
        linear.fixedBlockStepU,
        linear.fixedLaneU,
        linear.fixedScanVBits,
        linear.fixedStepVBits,
        linear.textureWrapU,
        linear.textureWrapV,
        context.zbuf.zbp << 5u,
        context.zbuf.psm,
        vertex.zInteger,
        static_cast<uint32_t>((context.test >> 17u) & 0x3u),
        context.zbuf.zmask ? 0u : 1u,
        GS_VULKAN_TEXTURE_SOURCE_FEEDBACK_SNAPSHOT,
        0u,
        0u,
    };
    if (feedbackLinearDepthCt32SpriteValidationError(prepared))
        return {false, GsFallbackReason::UnknownMemoryLayout};

    sprite = prepared;
    return decision;
}

GsBackendDecision prepareGsVulkanCt32Triangle(
    const GsDrawCommand &command,
    GsVulkanCt32Triangle &triangle) noexcept
{
    const GsBackendDecision decision =
        classifyGsFlatCt32Triangle(command);
    if (!decision.supported)
        return decision;

    const GSContext &context = command.context();
    const GsDrawBounds &bounds = command.bounds();
    const GSVertex &color = command.vertices()[2];
    std::array<FixedTriangleVertex, 3> vertices{{
        {command.fixedX()[0], command.fixedY()[0]},
        {command.fixedX()[1], command.fixedY()[1]},
        {command.fixedX()[2], command.fixedY()[2]},
    }};
    if (triangleEdge(vertices[0], vertices[1], vertices[2]) < 0)
        std::swap(vertices[1], vertices[2]);

    GsVulkanCt32Triangle prepared{};
    prepared.framebufferBaseBlock = context.frame.fbp << 5u;
    prepared.framebufferWidth =
        std::max<uint32_t>(context.frame.fbw, 1u);
    prepared.boundsX0 = static_cast<uint32_t>(bounds.x0);
    prepared.boundsY0 = static_cast<uint32_t>(bounds.y0);
    prepared.boundsX1 = static_cast<uint32_t>(bounds.x1);
    prepared.boundsY1 = static_cast<uint32_t>(bounds.y1);
    prepared.vertex0X12_4 = vertices[0].x;
    prepared.vertex0Y12_4 = vertices[0].y;
    prepared.vertex1X12_4 = vertices[1].x;
    prepared.vertex1Y12_4 = vertices[1].y;
    prepared.vertex2X12_4 = vertices[2].x;
    prepared.vertex2Y12_4 = vertices[2].y;
    prepared.rgba =
        static_cast<uint32_t>(color.r) |
        (static_cast<uint32_t>(color.g) << 8u) |
        (static_cast<uint32_t>(color.b) << 16u) |
        (static_cast<uint32_t>(color.a) << 24u);
    prepared.topLeftEdgeMask =
        (isTopLeftEdge(vertices[1], vertices[2]) ? 1u : 0u) |
        (isTopLeftEdge(vertices[2], vertices[0]) ? 2u : 0u) |
        (isTopLeftEdge(vertices[0], vertices[1]) ? 4u : 0u);
    if (ct32TriangleValidationError(prepared))
        return {false, GsFallbackReason::UnknownMemoryLayout};

    triangle = prepared;
    return decision;
}

GsBackendDecision prepareGsVulkanGouraudSourceOverDepthCt32Triangle(
    const GsDrawCommand &command,
    GsVulkanGouraudDepthCt32Triangle &triangle) noexcept
{
    const GsBackendDecision decision =
        classifyGsGouraudSourceOverDepthCt32Triangle(command);
    if (!decision.supported)
        return decision;

    const GSContext &context = command.context();
    const GsDrawBounds &bounds = command.bounds();
    const auto &vertices = command.vertices();
    const std::array<FixedTriangleVertex, 3> fixedVertices{{
        {command.fixedX()[0], command.fixedY()[0]},
        {command.fixedX()[1], command.fixedY()[1]},
        {command.fixedX()[2], command.fixedY()[2]},
    }};
    const int64_t signedArea = triangleEdge(
        fixedVertices[0], fixedVertices[1], fixedVertices[2]);
    const bool positiveArea = signedArea > 0;
    const auto orientedTopLeft =
        [positiveArea](
            const FixedTriangleVertex &a,
            const FixedTriangleVertex &b)
    {
        return positiveArea
            ? isTopLeftEdge(a, b)
            : isTopLeftEdge(b, a);
    };
    const auto packRgba = [](const GSVertex &vertex)
    {
        return static_cast<uint32_t>(vertex.r) |
               (static_cast<uint32_t>(vertex.g) << 8u) |
               (static_cast<uint32_t>(vertex.b) << 16u) |
               (static_cast<uint32_t>(vertex.a) << 24u);
    };

    GsVulkanGouraudDepthCt32Triangle prepared{};
    prepared.framebufferBaseBlock = context.frame.fbp << 5u;
    prepared.framebufferWidth =
        std::max<uint32_t>(context.frame.fbw, 1u);
    prepared.boundsX0 = static_cast<uint32_t>(bounds.x0);
    prepared.boundsY0 = static_cast<uint32_t>(bounds.y0);
    prepared.boundsX1 = static_cast<uint32_t>(bounds.x1);
    prepared.boundsY1 = static_cast<uint32_t>(bounds.y1);
    prepared.depthBaseBlock = context.zbuf.zbp << 5u;
    prepared.depthPsm = context.zbuf.psm;
    prepared.depth = vertices[0].zInteger;
    prepared.positiveArea = positiveArea ? 1u : 0u;
    prepared.vertex0X12_4 = fixedVertices[0].x;
    prepared.vertex0Y12_4 = fixedVertices[0].y;
    prepared.vertex1X12_4 = fixedVertices[1].x;
    prepared.vertex1Y12_4 = fixedVertices[1].y;
    prepared.vertex2X12_4 = fixedVertices[2].x;
    prepared.vertex2Y12_4 = fixedVertices[2].y;
    prepared.rgba0 = packRgba(vertices[0]);
    prepared.rgba1 = packRgba(vertices[1]);
    prepared.rgba2 = packRgba(vertices[2]);
    prepared.topLeftEdgeMask =
        (orientedTopLeft(fixedVertices[1], fixedVertices[2]) ? 1u : 0u) |
        (orientedTopLeft(fixedVertices[2], fixedVertices[0]) ? 2u : 0u) |
        (orientedTopLeft(fixedVertices[0], fixedVertices[1]) ? 4u : 0u);
    const GouraudColorDdaBits colorDda =
        prepareGouraudColorDdaBits(
            fixedVertices,
            {prepared.rgba0, prepared.rgba1, prepared.rgba2});
    prepared.colorDxBits = colorDda.dx;
    prepared.colorDyBits = colorDda.dy;
    if (gouraudDepthCt32TriangleValidationError(prepared))
        return {false, GsFallbackReason::UnknownMemoryLayout};

    triangle = prepared;
    return decision;
}

GsBackendDecision
prepareGsVulkanFeedbackNearestDepthCt32Triangle(
    const GsDrawCommand &command,
    GsVulkanFeedbackNearestDepthCt32Triangle &triangle,
    const GsDrawResources *classifiedResources) noexcept
{
    const GsBackendDecision decision = classifiedResources
        ? GsBackendDecision{true, GsFallbackReason::Supported}
        : classifyGsFeedbackNearestDepthCt32Triangle(command);
    if (!decision.supported)
        return decision;

    const GSContext &context = command.context();
    const GsDrawBounds &bounds = command.bounds();
    const auto &vertices = command.vertices();
    const std::array<FixedTriangleVertex, 3> fixedVertices{{
        {command.fixedX()[0], command.fixedY()[0]},
        {command.fixedX()[1], command.fixedY()[1]},
        {command.fixedX()[2], command.fixedY()[2]},
    }};
    const int64_t signedArea = triangleEdge(
        fixedVertices[0], fixedVertices[1], fixedVertices[2]);
    if (signedArea == 0)
        return {false, GsFallbackReason::UnsupportedPrimitiveState};
    const bool positiveArea = signedArea > 0;
    const auto orientedTopLeft =
        [positiveArea](
            const FixedTriangleVertex &a,
            const FixedTriangleVertex &b)
    {
        return positiveArea
            ? isTopLeftEdge(a, b)
            : isTopLeftEdge(b, a);
    };
    const auto packRgba = [](const GSVertex &vertex)
    {
        return static_cast<uint32_t>(vertex.r) |
               (static_cast<uint32_t>(vertex.g) << 8u) |
               (static_cast<uint32_t>(vertex.b) << 16u) |
               (static_cast<uint32_t>(vertex.a) << 24u);
    };
    const auto packDoubleBits = [](double value)
    {
        const uint64_t bits = std::bit_cast<uint64_t>(value);
        return std::array<uint32_t, 2>{
            static_cast<uint32_t>(bits),
            static_cast<uint32_t>(bits >> 32u)};
    };

    struct SortedVertex
    {
        size_t index;
        float x;
        float y;
    };
    std::array<SortedVertex, 3> sorted{{
        {0u, static_cast<float>(fixedVertices[0].x) / 16.0f,
         static_cast<float>(fixedVertices[0].y) / 16.0f},
        {1u, static_cast<float>(fixedVertices[1].x) / 16.0f,
         static_cast<float>(fixedVertices[1].y) / 16.0f},
        {2u, static_cast<float>(fixedVertices[2].x) / 16.0f,
         static_cast<float>(fixedVertices[2].y) / 16.0f},
    }};
    if (sorted[1].y < sorted[0].y)
        std::swap(sorted[0], sorted[1]);
    if (sorted[2].y < sorted[1].y)
        std::swap(sorted[1], sorted[2]);
    if (sorted[1].y < sorted[0].y)
        std::swap(sorted[0], sorted[1]);
    const float gradientDx01 = sorted[1].x - sorted[0].x;
    const float gradientDy01 = sorted[1].y - sorted[0].y;
    const float gradientDx02 = sorted[2].x - sorted[0].x;
    const float gradientDy02 = sorted[2].y - sorted[0].y;
    const float gradientCross = std::fma(
        gradientDy01,
        gradientDx02,
        -(gradientDx01 * gradientDy02));
    if (gradientCross == 0.0f || !std::isfinite(gradientCross))
        return {false, GsFallbackReason::UnsupportedPrimitiveState};
    const float gradientX01OverCross =
        gradientDx01 / gradientCross;
    const float gradientY01OverCross =
        gradientDy01 / gradientCross;
    const float gradientX02OverCross =
        gradientDx02 / gradientCross;
    const float gradientY02OverCross =
        gradientDy02 / gradientCross;

    GsVulkanFeedbackNearestDepthCt32Triangle prepared{};
    prepared.framebufferBaseBlock = context.frame.fbp << 5u;
    prepared.framebufferWidth =
        std::max<uint32_t>(context.frame.fbw, 1u);
    prepared.boundsX0 = static_cast<uint32_t>(bounds.x0);
    prepared.boundsY0 = static_cast<uint32_t>(bounds.y0);
    prepared.boundsX1 = static_cast<uint32_t>(bounds.x1);
    prepared.boundsY1 = static_cast<uint32_t>(bounds.y1);
    prepared.depthBaseBlock = context.zbuf.zbp << 5u;
    prepared.depthPsm = context.zbuf.psm;
    prepared.positiveArea = positiveArea ? 1u : 0u;
    prepared.topLeftEdgeMask =
        (orientedTopLeft(fixedVertices[1], fixedVertices[2]) ? 1u : 0u) |
        (orientedTopLeft(fixedVertices[2], fixedVertices[0]) ? 2u : 0u) |
        (orientedTopLeft(fixedVertices[0], fixedVertices[1]) ? 4u : 0u);
    const float textureScaleU =
        static_cast<float>(65536u << context.tex0.tw);
    const float textureScaleV =
        static_cast<float>(65536u << context.tex0.th);
    const auto finiteQ = [](float q)
    {
        return std::fabs(q) > 1.0e-8f ? q : 1.0f;
    };
    for (size_t index = 0u; index < 3u; ++index)
    {
        prepared.vertexX12_4[index] = fixedVertices[index].x;
        prepared.vertexY12_4[index] = fixedVertices[index].y;
        prepared.vertexZ[index] = vertices[index].zInteger;
        const float q = finiteQ(vertices[index].q);
        prepared.sBits[index] = std::bit_cast<uint32_t>(
            (vertices[index].s / q) * textureScaleU);
        prepared.tBits[index] = std::bit_cast<uint32_t>(
            (vertices[index].t / q) * textureScaleV);
    }
    prepared.rgba = packRgba(vertices[2]);

    const auto gradient =
        [&](double top, double middle, double bottom)
    {
        const double delta01 = middle - top;
        const double delta02 = bottom - top;
        return std::pair<double, double>{
            std::fma(
                delta02,
                static_cast<double>(gradientY01OverCross),
                -(delta01 *
                  static_cast<double>(gradientY02OverCross))),
            std::fma(
                delta01,
                static_cast<double>(gradientX02OverCross),
                -(delta02 *
                  static_cast<double>(gradientX01OverCross)))};
    };
    {
        const auto [dx, dy] = gradient(
            static_cast<double>(vertices[sorted[0].index].zInteger),
            static_cast<double>(vertices[sorted[1].index].zInteger),
            static_cast<double>(vertices[sorted[2].index].zInteger));
        prepared.depthDxBits = packDoubleBits(dx);
        prepared.depthDyBits = packDoubleBits(dy);
    }
    for (size_t component = 0u; component < 2u; ++component)
    {
        const auto value = [&](size_t sortedIndex)
        {
            const size_t index = sorted[sortedIndex].index;
            return std::bit_cast<float>(
                component == 0u
                    ? prepared.sBits[index]
                    : prepared.tBits[index]);
        };
        const float top = value(0u);
        const float delta01 = value(1u) - top;
        const float delta02 = value(2u) - top;
        prepared.textureDxBits[component] =
            std::bit_cast<uint32_t>(std::fma(
                delta02,
                gradientY01OverCross,
                -(delta01 * gradientY02OverCross)));
        prepared.textureDyBits[component] =
            std::bit_cast<uint32_t>(std::fma(
                delta01,
                gradientX02OverCross,
                -(delta02 * gradientX01OverCross)));
    }
    prepared.textureBaseBlock = context.tex0.tbp0;
    prepared.textureWidth = context.tex0.tbw;
    prepared.textureWidthLog2 = context.tex0.tw;
    prepared.textureHeightLog2 = context.tex0.th;
    prepared.textureWrapU =
        static_cast<uint32_t>(context.clamp & 0x3u);
    prepared.textureWrapV =
        static_cast<uint32_t>((context.clamp >> 2u) & 0x3u);

    if (feedbackNearestDepthCt32TriangleValidationError(prepared))
        return {false, GsFallbackReason::UnknownMemoryLayout};
    triangle = prepared;
    return decision;
}

#if (defined(__x86_64__) || defined(__i386__)) && \
    defined(__GNUC__) && !defined(__clang__) && \
    !defined(__SANITIZE_THREAD__)
// This setup uses explicit std::fma operations to match the shader's exact
// planes. Select hardware FMA where available, but keep every other product
// and sum uncontracted so feature dispatch cannot change the GS record.
__attribute__((target_clones("fma", "default"),
               optimize("fp-contract=off")))
#endif
GsBackendDecision
prepareGsVulkanResidentT8GouraudDepthCt32Triangle(
    const GsDrawCommand &command,
    uint32_t paletteIndex,
    GsVulkanResidentT8GouraudDepthCt32Triangle &triangle,
    const GsDrawResources *classifiedResources) noexcept
{
    const GsBackendDecision decision = classifiedResources
        ? GsBackendDecision{true, GsFallbackReason::Supported}
        : classifyGsT8GouraudDepthCt32Triangle(command);
    if (!decision.supported)
        return decision;
    const GSPrimReg &primitive = command.primitive();
    const bool textured = primitive.tme;
    const GSContext &context = command.context();
    const GsDrawBounds &bounds = command.bounds();
    const auto &vertices = command.vertices();
    const std::array<FixedTriangleVertex, 3> fixedVertices{{
        {command.fixedX()[0], command.fixedY()[0]},
        {command.fixedX()[1], command.fixedY()[1]},
        {command.fixedX()[2], command.fixedY()[2]},
    }};
    const int64_t signedArea = triangleEdge(
        fixedVertices[0], fixedVertices[1], fixedVertices[2]);
    if (signedArea == 0)
        return {false, GsFallbackReason::UnsupportedPrimitiveState};
    const bool positiveArea = signedArea > 0;
    const auto orientedTopLeft =
        [positiveArea](
            const FixedTriangleVertex &a,
            const FixedTriangleVertex &b)
    {
        return positiveArea
            ? isTopLeftEdge(a, b)
            : isTopLeftEdge(b, a);
    };
    const auto packRgba = [](const GSVertex &vertex)
    {
        return static_cast<uint32_t>(vertex.r) |
               (static_cast<uint32_t>(vertex.g) << 8u) |
               (static_cast<uint32_t>(vertex.b) << 16u) |
               (static_cast<uint32_t>(vertex.a) << 24u);
    };
    const auto packDoubleBits = [](double value)
    {
        const uint64_t bits = std::bit_cast<uint64_t>(value);
        return std::array<uint32_t, 2>{
            static_cast<uint32_t>(bits),
            static_cast<uint32_t>(bits >> 32u)};
    };

    struct SortedVertex
    {
        size_t index;
        float x;
        float y;
    };
    std::array<SortedVertex, 3> sorted{{
        {0u, static_cast<float>(fixedVertices[0].x) / 16.0f,
         static_cast<float>(fixedVertices[0].y) / 16.0f},
        {1u, static_cast<float>(fixedVertices[1].x) / 16.0f,
         static_cast<float>(fixedVertices[1].y) / 16.0f},
        {2u, static_cast<float>(fixedVertices[2].x) / 16.0f,
         static_cast<float>(fixedVertices[2].y) / 16.0f},
    }};
    if (sorted[1].y < sorted[0].y)
        std::swap(sorted[0], sorted[1]);
    if (sorted[2].y < sorted[1].y)
        std::swap(sorted[1], sorted[2]);
    if (sorted[1].y < sorted[0].y)
        std::swap(sorted[0], sorted[1]);
    const float gradientDx01 = sorted[1].x - sorted[0].x;
    const float gradientDy01 = sorted[1].y - sorted[0].y;
    const float gradientDx02 = sorted[2].x - sorted[0].x;
    const float gradientDy02 = sorted[2].y - sorted[0].y;
    const float gradientCross = std::fma(
        gradientDy01,
        gradientDx02,
        -(gradientDx01 * gradientDy02));
    if (gradientCross == 0.0f || !std::isfinite(gradientCross))
        return {false, GsFallbackReason::UnsupportedPrimitiveState};
    const float gradientX01OverCross =
        gradientDx01 / gradientCross;
    const float gradientY01OverCross =
        gradientDy01 / gradientCross;
    const float gradientX02OverCross =
        gradientDx02 / gradientCross;
    const float gradientY02OverCross =
        gradientDy02 / gradientCross;

    GsVulkanResidentT8GouraudDepthCt32Triangle prepared{};
    const float textureScaleU = textured
        ? static_cast<float>(65536u << context.tex0.tw)
        : 1.0f;
    const float textureScaleV = textured
        ? static_cast<float>(65536u << context.tex0.th)
        : 1.0f;
    const uint32_t maximumMipLevel = textured
        ? static_cast<uint32_t>((context.tex1 >> 2u) & 0x7u)
        : 0u;
    const uint8_t minificationFilter =
        static_cast<uint8_t>((context.tex1 >> 6u) & 0x7u);
    const bool mipmapsEnabled =
        maximumMipLevel != 0u &&
        minificationFilter >= 2u && minificationFilter <= 5u;
    const bool constantQFixed =
        textured &&
        (primitive.fst ||
         (!mipmapsEnabled &&
          vertices[0].q == vertices[1].q &&
          vertices[1].q == vertices[2].q));
    const auto finiteQ = [](float q)
    {
        return std::fabs(q) > 1.0e-8f ? q : 1.0f;
    };
    prepared.framebufferBaseBlock = context.frame.fbp << 5u;
    prepared.framebufferWidth =
        std::max<uint32_t>(context.frame.fbw, 1u);
    prepared.boundsX0 = static_cast<uint32_t>(bounds.x0);
    prepared.boundsY0 = static_cast<uint32_t>(bounds.y0);
    prepared.boundsX1 = static_cast<uint32_t>(bounds.x1);
    prepared.boundsY1 = static_cast<uint32_t>(bounds.y1);
    prepared.depthBaseBlock = context.zbuf.zbp << 5u;
    prepared.depthPsm = context.zbuf.psm;
    prepared.positiveArea = positiveArea ? 1u : 0u;
    prepared.topLeftEdgeMask =
        (orientedTopLeft(fixedVertices[1], fixedVertices[2]) ? 1u : 0u) |
        (orientedTopLeft(fixedVertices[2], fixedVertices[0]) ? 2u : 0u) |
        (orientedTopLeft(fixedVertices[0], fixedVertices[1]) ? 4u : 0u);
    for (size_t index = 0u; index < 3u; ++index)
    {
        prepared.vertexX12_4[index] = fixedVertices[index].x;
        prepared.vertexY12_4[index] = fixedVertices[index].y;
        prepared.vertexZ[index] = vertices[index].zInteger;
        prepared.rgba[index] = packRgba(
            primitive.iip ? vertices[index] : vertices[2]);
        // The scanline setup consumes already texture-scaled S/T. Preserve
        // those exact host products rather than repeating them on the device.
        if (!textured)
        {
            prepared.sBits[index] = std::bit_cast<uint32_t>(0.0f);
            prepared.tBits[index] = std::bit_cast<uint32_t>(0.0f);
            prepared.qBits[index] = std::bit_cast<uint32_t>(1.0f);
        }
        else if (primitive.fst)
        {
            prepared.sBits[index] = std::bit_cast<uint32_t>(
                static_cast<float>(vertices[index].u) * 4096.0f -
                32768.0f);
            prepared.tBits[index] = std::bit_cast<uint32_t>(
                static_cast<float>(vertices[index].v) * 4096.0f -
                32768.0f);
            prepared.qBits[index] = std::bit_cast<uint32_t>(1.0f);
        }
        else if (constantQFixed)
        {
            const float q = finiteQ(vertices[index].q);
            prepared.sBits[index] = std::bit_cast<uint32_t>(
                (vertices[index].s / q) * textureScaleU - 32768.0f);
            prepared.tBits[index] = std::bit_cast<uint32_t>(
                (vertices[index].t / q) * textureScaleV - 32768.0f);
            prepared.qBits[index] = std::bit_cast<uint32_t>(1.0f);
        }
        else
        {
            prepared.sBits[index] = std::bit_cast<uint32_t>(
                vertices[index].s * textureScaleU);
            prepared.tBits[index] = std::bit_cast<uint32_t>(
                vertices[index].t * textureScaleV);
            prepared.qBits[index] =
                std::bit_cast<uint32_t>(vertices[index].q);
        }
        prepared.vertexFog |=
            static_cast<uint32_t>(vertices[index].fog) << (index * 8u);
    }
    prepared.fogColor = command.globalState().fogColor;

    const auto gradient =
        [&](double top, double middle, double bottom)
    {
        const double delta01 = middle - top;
        const double delta02 = bottom - top;
        return std::pair<double, double>{
            std::fma(
                delta02,
                static_cast<double>(gradientY01OverCross),
                -(delta01 *
                  static_cast<double>(gradientY02OverCross))),
            std::fma(
                delta01,
                static_cast<double>(gradientX02OverCross),
                -(delta02 *
                  static_cast<double>(gradientX01OverCross)))};
    };
    for (size_t channel = 0u; channel < 4u; ++channel)
    {
        const uint32_t shift = static_cast<uint32_t>(channel * 8u);
        const uint32_t first =
            (prepared.rgba[0] >> shift) & 0xFFu;
        const uint32_t second =
            (prepared.rgba[1] >> shift) & 0xFFu;
        const uint32_t third =
            (prepared.rgba[2] >> shift) & 0xFFu;
        // A constant attribute has an exact zero plane gradient regardless
        // of vertex sorting or winding. This is especially common for source
        // alpha and avoids two host fma operations per constant channel.
        if (first == second && second == third)
            continue;
        const auto value = [&](size_t sortedIndex)
        {
            return static_cast<float>(
                       (prepared.rgba[sorted[sortedIndex].index] >>
                        (channel * 8u)) & 0xFFu) *
                   128.0f;
        };
        const float top = value(0u);
        const float delta01 = value(1u) - top;
        const float delta02 = value(2u) - top;
        prepared.colorDxBits[channel] = std::bit_cast<uint32_t>(std::fma(
            delta02,
            gradientY01OverCross,
            -(delta01 * gradientY02OverCross)));
        prepared.colorDyBits[channel] = std::bit_cast<uint32_t>(std::fma(
            delta01,
            gradientX02OverCross,
            -(delta02 * gradientX01OverCross)));
    }
    {
        const bool constantFog =
            vertices[0].fog == vertices[1].fog &&
            vertices[1].fog == vertices[2].fog;
        if (!constantFog)
        {
            const auto value = [&](size_t sortedIndex)
            {
                return static_cast<float>(
                           vertices[sorted[sortedIndex].index].fog) *
                       128.0f;
            };
            const float top = value(0u);
            const float delta01 = value(1u) - top;
            const float delta02 = value(2u) - top;
            prepared.fogDxBits = std::bit_cast<uint32_t>(std::fma(
                delta02,
                gradientY01OverCross,
                -(delta01 * gradientY02OverCross)));
            prepared.fogDyBits = std::bit_cast<uint32_t>(std::fma(
                delta01,
                gradientX02OverCross,
                -(delta02 * gradientX01OverCross)));
        }
    }
    {
        const auto [dx, dy] = gradient(
            static_cast<double>(
                vertices[sorted[0].index].zInteger),
            static_cast<double>(
                vertices[sorted[1].index].zInteger),
            static_cast<double>(
                vertices[sorted[2].index].zInteger));
        prepared.depthDxBits = packDoubleBits(dx);
        prepared.depthDyBits = packDoubleBits(dy);
    }
    if (textured)
    {
        for (size_t component = 0u; component < 3u; ++component)
        {
            const auto value = [&](size_t sortedIndex)
            {
                const GSVertex &vertex =
                    vertices[sorted[sortedIndex].index];
                if (component == 0u)
                    return constantQFixed
                        ? std::bit_cast<float>(prepared.sBits[
                              sorted[sortedIndex].index])
                        : vertex.s * textureScaleU;
                if (component == 1u)
                    return constantQFixed
                        ? std::bit_cast<float>(prepared.tBits[
                              sorted[sortedIndex].index])
                        : vertex.t * textureScaleV;
                return constantQFixed ? 1.0f : vertex.q;
            };
            const float top = value(0u);
            const float delta01 = value(1u) - top;
            const float delta02 = value(2u) - top;
            prepared.textureDxBits[component] =
                std::bit_cast<uint32_t>(std::fma(
                    delta02,
                    gradientY01OverCross,
                    -(delta01 * gradientY02OverCross)));
            prepared.textureDyBits[component] =
                std::bit_cast<uint32_t>(std::fma(
                    delta01,
                    gradientX02OverCross,
                    -(delta02 * gradientX01OverCross)));
        }
    }

    prepared.textureBaseBlocks[0] = textured ? context.tex0.tbp0 : 0u;
    prepared.textureWidths[0] = textured ? context.tex0.tbw : 1u;
    prepared.maximumMipLevel = maximumMipLevel;
    for (uint32_t level = 1u;
         level <= prepared.maximumMipLevel;
         ++level)
    {
        const uint32_t shift = (level - 1u) * 20u;
        prepared.textureBaseBlocks[level] =
            static_cast<uint32_t>((context.miptbp1 >> shift) & 0x3FFFu);
        prepared.textureWidths[level] =
            static_cast<uint32_t>(
                (context.miptbp1 >> (shift + 14u)) & 0x3Fu);
    }
    if (textured)
    {
        prepared.textureWidthLog2 = context.tex0.tw;
        prepared.textureHeightLog2 = context.tex0.th;
        prepared.textureWrapU =
            static_cast<uint32_t>(context.clamp & 0x3u);
        prepared.textureWrapV =
            static_cast<uint32_t>((context.clamp >> 2u) & 0x3u);
        const int32_t rawK =
            static_cast<int32_t>((context.tex1 >> 32u) & 0xFFFu);
        prepared.lodK =
            (rawK & 0x800) != 0 ? rawK - 0x1000 : rawK;
        prepared.lodL =
            static_cast<uint32_t>((context.tex1 >> 19u) & 0x3u);
    }
    prepared.alphaReference =
        static_cast<uint32_t>((context.test >> 4u) & 0xFFu);
    prepared.paletteIndex = paletteIndex;
    prepared.rasterFlags =
        (command.primitive().fge
             ? GS_VULKAN_T8_GOURAUD_FLAG_FOG
             : 0u) |
        ((((context.test >> 17u) & 0x3u) == 2u)
             ? GS_VULKAN_T8_GOURAUD_FLAG_DEPTH_GEQUAL
             : 0u) |
        (!context.zbuf.zmask
             ? GS_VULKAN_T8_GOURAUD_FLAG_DEPTH_WRITE
             : 0u) |
        ((((context.test >> 12u) & 0x3u) == 3u)
             ? GS_VULKAN_T8_GOURAUD_FLAG_ALPHA_FAIL_RGB_ONLY
             : 0u) |
        (constantQFixed
             ? GS_VULKAN_T8_GOURAUD_FLAG_CONSTANT_Q_FIXED
             : 0u) |
        (!textured
             ? GS_VULKAN_T8_GOURAUD_FLAG_TEXTURE_DISABLED
             : 0u) |
        (((context.alpha & 0xFFu) == 0x48u)
             ? GS_VULKAN_T8_GOURAUD_FLAG_ADDITIVE_SOURCE_ALPHA
             : 0u);

    // The classifier already proved exact non-aliasing resource masks for
    // this immutable command. The service boundary repeats the complete raw-
    // record validation before device execution; avoid rebuilding all page
    // ranges a third time during record preparation.
    if (!classifiedResources &&
        t8GouraudDepthCt32TriangleValidationError(
            prepared, false, true))
        return {false, GsFallbackReason::UnknownMemoryLayout};
    triangle = prepared;
    return decision;
}

GsBackendDecision
prepareGsVulkanT8GouraudDepthCt32Triangle(
    const GsDrawCommand &command,
    std::span<const uint32_t> decodedPalette,
    GsVulkanT8GouraudDepthCt32Triangle &triangle,
    const GsDrawResources *classifiedResources) noexcept
{
    const bool textured = command.primitive().tme;
    if ((textured && decodedPalette.size() != 256u) ||
        (!textured && !decodedPalette.empty()))
        return {false, GsFallbackReason::BackendUnavailable};

    GsVulkanResidentT8GouraudDepthCt32Triangle compact{};
    const GsBackendDecision decision =
        prepareGsVulkanResidentT8GouraudDepthCt32Triangle(
            command, 0u, compact, classifiedResources);
    if (!decision.supported)
        return decision;

    GsVulkanT8GouraudDepthCt32Triangle prepared{};
    static_assert(sizeof(compact) == offsetof(
        GsVulkanT8GouraudDepthCt32Triangle, palette));
    std::memcpy(&prepared, &compact, sizeof(compact));
    prepared.paletteIndex = 0u;
    if (textured)
    {
        std::copy(
            decodedPalette.begin(), decodedPalette.end(),
            prepared.palette.begin());
    }
    if (t8GouraudDepthCt32TriangleValidationError(prepared, false))
        return {false, GsFallbackReason::UnknownMemoryLayout};
    triangle = prepared;
    return decision;
}

#if PS2X_HAS_GS_VULKAN
namespace
{
    class DynamicLibrary final
    {
    public:
        DynamicLibrary() = default;
        ~DynamicLibrary()
        {
            close();
        }

        DynamicLibrary(const DynamicLibrary &) = delete;
        DynamicLibrary &operator=(const DynamicLibrary &) = delete;

        bool open(const std::string &path, std::string &error)
        {
            close();
#if defined(_WIN32)
            m_handle = LoadLibraryA(path.c_str());
            if (!m_handle)
            {
                error = "LoadLibrary failed with code " +
                        std::to_string(GetLastError());
                return false;
            }
#else
            dlerror();
            m_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!m_handle)
            {
                const char *message = dlerror();
                error = message ? message : "dlopen failed";
                return false;
            }
#endif
            m_path = path;
            error.clear();
            return true;
        }

        [[nodiscard]] void *symbol(const char *name) const noexcept
        {
#if defined(_WIN32)
            return m_handle
                ? reinterpret_cast<void *>(GetProcAddress(m_handle, name))
                : nullptr;
#else
            return m_handle ? dlsym(m_handle, name) : nullptr;
#endif
        }

        [[nodiscard]] const std::string &path() const noexcept
        {
            return m_path;
        }

    private:
        void close() noexcept
        {
            if (!m_handle)
                return;
#if defined(_WIN32)
            FreeLibrary(m_handle);
#else
            dlclose(m_handle);
#endif
            m_handle = nullptr;
            m_path.clear();
        }

#if defined(_WIN32)
        HMODULE m_handle = nullptr;
#else
        void *m_handle = nullptr;
#endif
        std::string m_path;
    };

    struct ValidationCounters
    {
        std::atomic<uint32_t> warnings{0u};
        std::atomic<uint32_t> errors{0u};
    };

    VKAPI_ATTR VkBool32 VKAPI_CALL validationCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT *,
        void *userData)
    {
        auto *counters = static_cast<ValidationCounters *>(userData);
        if (!counters)
            return VK_FALSE;
        if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0u)
            counters->errors.fetch_add(1u, std::memory_order_relaxed);
        else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0u)
            counters->warnings.fetch_add(1u, std::memory_order_relaxed);
        return VK_FALSE;
    }

    VkDebugUtilsMessengerCreateInfoEXT makeDebugCreateInfo(
        ValidationCounters &counters) noexcept
    {
        VkDebugUtilsMessengerCreateInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        info.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        info.messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        info.pfnUserCallback = validationCallback;
        info.pUserData = &counters;
        return info;
    }

    std::string resultDescription(VkResult result)
    {
        switch (result)
        {
        case VK_SUCCESS:
            return "VK_SUCCESS";
        case VK_NOT_READY:
            return "VK_NOT_READY";
        case VK_TIMEOUT:
            return "VK_TIMEOUT";
        case VK_ERROR_OUT_OF_HOST_MEMORY:
            return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:
            return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:
            return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST:
            return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_LAYER_NOT_PRESENT:
            return "VK_ERROR_LAYER_NOT_PRESENT";
        case VK_ERROR_EXTENSION_NOT_PRESENT:
            return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT:
            return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER:
            return "VK_ERROR_INCOMPATIBLE_DRIVER";
        default:
            return "VkResult(" + std::to_string(static_cast<int32_t>(result)) + ")";
        }
    }

    template <typename T, typename Enumerator>
    VkResult enumerateValues(Enumerator &&enumerator,
                             std::vector<T> &values)
    {
        for (uint32_t attempt = 0u; attempt < 4u; ++attempt)
        {
            uint32_t count = 0u;
            VkResult result = enumerator(&count, nullptr);
            if (result != VK_SUCCESS && result != VK_INCOMPLETE)
                return result;
            values.resize(count);
            if (count == 0u)
                return VK_SUCCESS;
            result = enumerator(&count, values.data());
            values.resize(count);
            if (result == VK_SUCCESS)
                return result;
            if (result != VK_INCOMPLETE)
                return result;
        }
        return VK_INCOMPLETE;
    }

    template <typename T>
    T loadInstanceProc(PFN_vkGetInstanceProcAddr getInstanceProcAddr,
                       VkInstance instance,
                       const char *name) noexcept
    {
        return reinterpret_cast<T>(getInstanceProcAddr(instance, name));
    }

    bool openVulkanLoader(const GsVulkanProbeConfig &config,
                          DynamicLibrary &library,
                          std::string &error)
    {
        if (!config.loaderPath.empty())
        {
            if (library.open(config.loaderPath, error))
                return true;
            error = "failed to load explicit Vulkan loader '" +
                    config.loaderPath + "': " + error;
            return false;
        }

#if defined(_WIN32)
        const char *candidates[] = {"vulkan-1.dll"};
#elif defined(__APPLE__)
        const char *candidates[] = {
            "libvulkan.1.dylib",
            "libvulkan.dylib",
            "libMoltenVK.dylib",
        };
#else
        const char *candidates[] = {
            "libvulkan.so.1",
            "libvulkan.so",
        };
#endif
        std::string lastError;
        for (const char *candidate : candidates)
        {
            if (library.open(candidate, lastError))
                return true;
        }
        error = "no Vulkan loader could be opened";
        if (!lastError.empty())
            error += ": " + lastError;
        return false;
    }

    GsVulkanDeviceKind deviceKind(VkPhysicalDeviceType type) noexcept
    {
        switch (type)
        {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
            return GsVulkanDeviceKind::IntegratedGpu;
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
            return GsVulkanDeviceKind::DiscreteGpu;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
            return GsVulkanDeviceKind::VirtualGpu;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
            return GsVulkanDeviceKind::Cpu;
        default:
            return GsVulkanDeviceKind::Other;
        }
    }

    int deviceScore(const GsVulkanDeviceReport &device) noexcept
    {
        int score = 0;
        switch (device.kind)
        {
        case GsVulkanDeviceKind::DiscreteGpu:
            score = 400;
            break;
        case GsVulkanDeviceKind::IntegratedGpu:
            score = 300;
            break;
        case GsVulkanDeviceKind::VirtualGpu:
            score = 200;
            break;
        case GsVulkanDeviceKind::Other:
            score = 100;
            break;
        case GsVulkanDeviceKind::Cpu:
            break;
        }
        if (device.dedicatedComputeQueue)
            score += 10;
        return score;
    }

    bool deviceMatchesPreference(const GsVulkanDeviceReport &device,
                                 const GsVulkanProbeConfig &config) noexcept
    {
        return (config.preferredVendorId == 0u ||
                device.vendorId == config.preferredVendorId) &&
               (config.preferredDeviceId == 0u ||
                device.deviceId == config.preferredDeviceId);
    }

    int32_t selectDevice(const std::vector<GsVulkanDeviceReport> &devices,
                         const GsVulkanProbeConfig &config)
    {
        int32_t selected = -1;
        int selectedScore = -1;
        for (size_t index = 0u; index < devices.size(); ++index)
        {
            const GsVulkanDeviceReport &candidate = devices[index];
            if (!candidate.suitable ||
                !deviceMatchesPreference(candidate, config))
            {
                continue;
            }

            const int score = deviceScore(candidate);
            bool replace = score > selectedScore;
            if (!replace && score == selectedScore && selected >= 0)
            {
                const GsVulkanDeviceReport &current =
                    devices[static_cast<size_t>(selected)];
                replace = std::tie(candidate.vendorId,
                                   candidate.deviceId,
                                   candidate.name) <
                          std::tie(current.vendorId,
                                   current.deviceId,
                                   current.name);
            }
            if (replace)
            {
                selected = static_cast<int32_t>(index);
                selectedScore = score;
            }
        }
        return selected;
    }

    struct InstanceOwner
    {
        VkInstance instance = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
        PFN_vkDestroyDebugUtilsMessengerEXT destroyMessenger = nullptr;
        PFN_vkDestroyInstance destroyInstance = nullptr;

        void reset() noexcept
        {
            if (messenger != VK_NULL_HANDLE && destroyMessenger)
                destroyMessenger(instance, messenger, nullptr);
            messenger = VK_NULL_HANDLE;
            if (instance != VK_NULL_HANDLE && destroyInstance)
                destroyInstance(instance, nullptr);
            instance = VK_NULL_HANDLE;
        }

        ~InstanceOwner()
        {
            reset();
        }
    };
}
#endif

GsVulkanCapabilityReport probeGsVulkanCapabilities(
    const GsVulkanProbeConfig &config)
{
    GsVulkanCapabilityReport report{};
    report.validationRequested = config.enableValidation;

#if !PS2X_HAS_GS_VULKAN
    report.status = GsVulkanProbeStatus::CompiledOut;
    report.detail = "Vulkan support was disabled or its headers were unavailable at build time";
    return report;
#else
    report.compiled = true;

    DynamicLibrary library;
    std::string loaderError;
    if (!openVulkanLoader(config, library, loaderError))
    {
        report.status = GsVulkanProbeStatus::LoaderUnavailable;
        report.detail = std::move(loaderError);
        return report;
    }
    report.loaderAvailable = true;

    const auto getInstanceProcAddr =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(
            library.symbol("vkGetInstanceProcAddr"));
    if (!getInstanceProcAddr)
    {
        report.status = GsVulkanProbeStatus::LoaderInvalid;
        report.detail = "Vulkan loader does not export vkGetInstanceProcAddr";
        return report;
    }

    const auto enumerateInstanceVersion =
        loadInstanceProc<PFN_vkEnumerateInstanceVersion>(
            getInstanceProcAddr, VK_NULL_HANDLE,
            "vkEnumerateInstanceVersion");
    report.loaderApiVersion = VK_API_VERSION_1_0;
    if (enumerateInstanceVersion)
    {
        const VkResult versionResult =
            enumerateInstanceVersion(&report.loaderApiVersion);
        if (versionResult != VK_SUCCESS)
        {
            report.status = GsVulkanProbeStatus::LoaderInvalid;
            report.detail = "vkEnumerateInstanceVersion failed: " +
                            resultDescription(versionResult);
            return report;
        }
    }

    const auto enumerateLayers =
        loadInstanceProc<PFN_vkEnumerateInstanceLayerProperties>(
            getInstanceProcAddr, VK_NULL_HANDLE,
            "vkEnumerateInstanceLayerProperties");
    const auto enumerateExtensions =
        loadInstanceProc<PFN_vkEnumerateInstanceExtensionProperties>(
            getInstanceProcAddr, VK_NULL_HANDLE,
            "vkEnumerateInstanceExtensionProperties");
    const auto createInstance =
        loadInstanceProc<PFN_vkCreateInstance>(
            getInstanceProcAddr, VK_NULL_HANDLE, "vkCreateInstance");
    if (!enumerateLayers || !enumerateExtensions || !createInstance)
    {
        report.status = GsVulkanProbeStatus::LoaderInvalid;
        report.detail = "Vulkan loader is missing required global entry points";
        return report;
    }

    std::vector<VkLayerProperties> layers;
    VkResult result = enumerateValues<VkLayerProperties>(
        [&](uint32_t *count, VkLayerProperties *values)
        {
            return enumerateLayers(count, values);
        },
        layers);
    if (result != VK_SUCCESS)
    {
        report.status = GsVulkanProbeStatus::LoaderInvalid;
        report.detail = "layer enumeration failed: " +
                        resultDescription(result);
        return report;
    }
    report.validationLayerAvailable = std::any_of(
        layers.begin(), layers.end(), [](const VkLayerProperties &layer)
        {
            return std::strcmp(layer.layerName,
                               "VK_LAYER_KHRONOS_validation") == 0;
        });

    std::vector<VkExtensionProperties> extensions;
    result = enumerateValues<VkExtensionProperties>(
        [&](uint32_t *count, VkExtensionProperties *values)
        {
            return enumerateExtensions(nullptr, count, values);
        },
        extensions);
    if (result != VK_SUCCESS)
    {
        report.status = GsVulkanProbeStatus::LoaderInvalid;
        report.detail = "instance-extension enumeration failed: " +
                        resultDescription(result);
        return report;
    }
    report.debugUtilsAvailable = std::any_of(
        extensions.begin(), extensions.end(),
        [](const VkExtensionProperties &extension)
        {
            return std::strcmp(extension.extensionName,
                               VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0;
        });
#if defined(VK_KHR_portability_enumeration)
    const bool portabilityEnumerationAvailable = std::any_of(
        extensions.begin(), extensions.end(),
        [](const VkExtensionProperties &extension)
        {
            return std::strcmp(
                       extension.extensionName,
                       VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
        });
#endif

    if (config.enableValidation &&
        (!report.validationLayerAvailable || !report.debugUtilsAvailable))
    {
        report.status = GsVulkanProbeStatus::ValidationUnavailable;
        report.detail = "validation requested but the Khronos layer or debug-utils extension is unavailable";
        return report;
    }

    ValidationCounters validationCounters;
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo =
        makeDebugCreateInfo(validationCounters);
    const char *validationLayer = "VK_LAYER_KHRONOS_validation";
    std::vector<const char *> enabledExtensions;
    if (config.enableValidation)
        enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#if defined(VK_KHR_portability_enumeration)
    if (portabilityEnumerationAvailable)
    {
        enabledExtensions.push_back(
            VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
#endif

    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "PS2Recomp GS capability probe";
    applicationInfo.applicationVersion = 1u;
    applicationInfo.pEngineName = "PS2Recomp";
    applicationInfo.engineVersion = 1u;
    applicationInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &applicationInfo;
    if (config.enableValidation)
    {
        instanceInfo.pNext = &debugCreateInfo;
        instanceInfo.enabledLayerCount = 1u;
        instanceInfo.ppEnabledLayerNames = &validationLayer;
    }
    instanceInfo.enabledExtensionCount =
        static_cast<uint32_t>(enabledExtensions.size());
    instanceInfo.ppEnabledExtensionNames = enabledExtensions.empty()
        ? nullptr
        : enabledExtensions.data();
#if defined(VK_KHR_portability_enumeration)
    if (portabilityEnumerationAvailable)
    {
        instanceInfo.flags |=
            VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
#endif

    InstanceOwner owner;
    result = createInstance(&instanceInfo, nullptr, &owner.instance);
    if (result != VK_SUCCESS)
    {
        report.status = GsVulkanProbeStatus::InstanceCreationFailed;
        report.detail = "vkCreateInstance failed: " +
                        resultDescription(result);
        report.validationWarnings =
            validationCounters.warnings.load(std::memory_order_relaxed);
        report.validationErrors =
            validationCounters.errors.load(std::memory_order_relaxed);
        return report;
    }

    owner.destroyInstance = loadInstanceProc<PFN_vkDestroyInstance>(
        getInstanceProcAddr, owner.instance, "vkDestroyInstance");
    if (!owner.destroyInstance)
    {
        report.status = GsVulkanProbeStatus::LoaderInvalid;
        report.detail = "Vulkan loader is missing vkDestroyInstance";
        return report;
    }

    auto finish = [&](GsVulkanProbeStatus status,
                      std::string detail) mutable
    {
        report.status = status;
        report.detail = std::move(detail);
        owner.reset();
        report.validationWarnings =
            validationCounters.warnings.load(std::memory_order_relaxed);
        report.validationErrors =
            validationCounters.errors.load(std::memory_order_relaxed);
        if (report.status == GsVulkanProbeStatus::Ready &&
            report.validationErrors != 0u)
        {
            report.status = GsVulkanProbeStatus::ValidationError;
            report.detail = "Vulkan validation reported errors during capability discovery";
            report.selectedDeviceIndex = -1;
        }
        return std::move(report);
    };

    if (config.enableValidation)
    {
        const auto createDebugMessenger =
            loadInstanceProc<PFN_vkCreateDebugUtilsMessengerEXT>(
                getInstanceProcAddr, owner.instance,
                "vkCreateDebugUtilsMessengerEXT");
        owner.destroyMessenger =
            loadInstanceProc<PFN_vkDestroyDebugUtilsMessengerEXT>(
                getInstanceProcAddr, owner.instance,
                "vkDestroyDebugUtilsMessengerEXT");
        if (!createDebugMessenger || !owner.destroyMessenger)
        {
            return finish(
                GsVulkanProbeStatus::LoaderInvalid,
                "debug-utils extension entry points are unavailable");
        }
        result = createDebugMessenger(
            owner.instance, &debugCreateInfo, nullptr, &owner.messenger);
        if (result != VK_SUCCESS)
        {
            return finish(
                GsVulkanProbeStatus::InstanceCreationFailed,
                "debug messenger creation failed: " +
                    resultDescription(result));
        }
        report.validationEnabled = true;
    }

    const auto enumeratePhysicalDevices =
        loadInstanceProc<PFN_vkEnumeratePhysicalDevices>(
            getInstanceProcAddr, owner.instance,
            "vkEnumeratePhysicalDevices");
    const auto getPhysicalDeviceProperties =
        loadInstanceProc<PFN_vkGetPhysicalDeviceProperties>(
            getInstanceProcAddr, owner.instance,
            "vkGetPhysicalDeviceProperties");
    const auto getPhysicalDeviceFeatures =
        loadInstanceProc<PFN_vkGetPhysicalDeviceFeatures>(
            getInstanceProcAddr, owner.instance,
            "vkGetPhysicalDeviceFeatures");
    const auto getQueueFamilyProperties =
        loadInstanceProc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            getInstanceProcAddr, owner.instance,
            "vkGetPhysicalDeviceQueueFamilyProperties");
    const auto getMemoryProperties =
        loadInstanceProc<PFN_vkGetPhysicalDeviceMemoryProperties>(
            getInstanceProcAddr, owner.instance,
            "vkGetPhysicalDeviceMemoryProperties");
    if (!enumeratePhysicalDevices || !getPhysicalDeviceProperties ||
        !getPhysicalDeviceFeatures || !getQueueFamilyProperties ||
        !getMemoryProperties)
    {
        return finish(
            GsVulkanProbeStatus::LoaderInvalid,
            "Vulkan instance is missing required physical-device entry points");
    }

    std::vector<VkPhysicalDevice> physicalDevices;
    result = enumerateValues<VkPhysicalDevice>(
        [&](uint32_t *count, VkPhysicalDevice *values)
        {
            return enumeratePhysicalDevices(owner.instance, count, values);
        },
        physicalDevices);
    if (result != VK_SUCCESS)
    {
        return finish(
            GsVulkanProbeStatus::DeviceEnumerationFailed,
            "physical-device enumeration failed: " +
                resultDescription(result));
    }
    if (physicalDevices.empty())
    {
        return finish(
            GsVulkanProbeStatus::NoPhysicalDevices,
            "Vulkan loader exposed no physical devices");
    }

    report.devices.reserve(physicalDevices.size());
    for (VkPhysicalDevice physicalDevice : physicalDevices)
    {
        VkPhysicalDeviceProperties properties{};
        VkPhysicalDeviceFeatures features{};
        VkPhysicalDeviceMemoryProperties memoryProperties{};
        getPhysicalDeviceProperties(physicalDevice, &properties);
        getPhysicalDeviceFeatures(physicalDevice, &features);
        getMemoryProperties(physicalDevice, &memoryProperties);

        uint32_t queueCount = 0u;
        getQueueFamilyProperties(physicalDevice, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queueCount);
        if (queueCount != 0u)
        {
            getQueueFamilyProperties(
                physicalDevice, &queueCount, queues.data());
            queues.resize(queueCount);
        }

        GsVulkanDeviceReport device{};
        device.name = properties.deviceName;
        device.vendorId = properties.vendorID;
        device.deviceId = properties.deviceID;
        device.apiVersion = properties.apiVersion;
        device.driverVersion = properties.driverVersion;
        device.kind = deviceKind(properties.deviceType);
        device.maxStorageBufferRange =
            properties.limits.maxStorageBufferRange;
        device.maxComputeWorkGroupCountX =
            properties.limits.maxComputeWorkGroupCount[0];
        device.maxComputeWorkGroupInvocations =
            properties.limits.maxComputeWorkGroupInvocations;
        device.maxComputeWorkGroupSizeX =
            properties.limits.maxComputeWorkGroupSize[0];
        device.shaderInt16 = features.shaderInt16 == VK_TRUE;
        device.shaderInt64 = features.shaderInt64 == VK_TRUE;
        device.shaderFloat64 = features.shaderFloat64 == VK_TRUE;

        for (uint32_t index = 0u; index < queues.size(); ++index)
        {
            const VkQueueFlags flags = queues[index].queueFlags;
            if (queues[index].queueCount == 0u ||
                (flags & VK_QUEUE_COMPUTE_BIT) == 0u)
            {
                continue;
            }
            const bool dedicated = (flags & VK_QUEUE_GRAPHICS_BIT) == 0u;
            if (!device.computeQueue || dedicated)
            {
                device.computeQueue = true;
                device.dedicatedComputeQueue = dedicated;
                device.queueFamilyIndex = index;
            }
            if (dedicated)
                break;
        }

        for (uint32_t index = 0u;
             index < memoryProperties.memoryTypeCount;
             ++index)
        {
            const VkMemoryType &type = memoryProperties.memoryTypes[index];
            if (type.heapIndex >= memoryProperties.memoryHeapCount ||
                memoryProperties.memoryHeaps[type.heapIndex].size <
                    GS_VULKAN_VRAM_SIZE)
            {
                continue;
            }
            device.deviceLocalMemory =
                device.deviceLocalMemory ||
                (type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0u;
            device.hostVisibleMemory =
                device.hostVisibleMemory ||
                (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0u;
        }

        const bool storageRange =
            device.maxStorageBufferRange >= GS_VULKAN_VRAM_SIZE;
        const bool dispatchLimits =
            device.maxComputeWorkGroupCountX >=
                GS_VULKAN_NOOP_GROUP_COUNT &&
            device.maxComputeWorkGroupInvocations >=
                GS_VULKAN_NOOP_LOCAL_SIZE &&
            device.maxComputeWorkGroupSizeX >=
                GS_VULKAN_NOOP_LOCAL_SIZE;
        device.exactVramStorage =
            device.computeQueue && storageRange && dispatchLimits &&
            device.deviceLocalMemory && device.hostVisibleMemory;
        device.suitable =
            device.exactVramStorage &&
            device.kind != GsVulkanDeviceKind::Cpu;
        device.exactCt32Triangle =
            device.suitable && device.shaderInt64;
        device.exactGouraudDepthCt32Triangle =
            device.exactCt32Triangle;
        device.exactT8GouraudDepthCt32Triangle =
            device.suitable && device.shaderInt64 &&
            device.shaderFloat64;
        device.exactDepthCt32Sprite = device.suitable;
        device.exactNearestCt32Sprite = device.suitable;
        device.exactLinearCt32Sprite = device.suitable;
        device.exactFeedbackLinearDepthCt32Sprite = device.suitable;
        device.exactFeedbackNearestDepthCt32Triangle =
            device.suitable && device.shaderInt64 &&
            device.shaderFloat64;

        if (device.kind == GsVulkanDeviceKind::Cpu)
            device.rejectionReason = "CPU Vulkan implementations are not hardware-GS targets";
        else if (!device.computeQueue)
            device.rejectionReason = "no compute-capable queue family";
        else if (!storageRange)
            device.rejectionReason = "maxStorageBufferRange is smaller than 4 MiB";
        else if (!dispatchLimits)
            device.rejectionReason = "compute dispatch limits cannot cover 4 MiB with the fixed kernel";
        else if (!device.deviceLocalMemory)
            device.rejectionReason = "no device-local memory heap can hold 4 MiB";
        else if (!device.hostVisibleMemory)
            device.rejectionReason = "no host-visible staging heap can hold 4 MiB";

        report.devices.push_back(std::move(device));
    }

    report.selectedDeviceIndex = selectDevice(report.devices, config);
    if (report.selectedDeviceIndex < 0)
    {
        const bool preferenceSpecified =
            config.preferredVendorId != 0u ||
            config.preferredDeviceId != 0u;
        return finish(
            GsVulkanProbeStatus::NoSuitableDevice,
            preferenceSpecified
                ? "no suitable Vulkan device matches the required vendor/device override"
                : "no hardware Vulkan device satisfies the raw 4 MiB storage requirements");
    }

    const GsVulkanDeviceReport *selected = report.selectedDevice();
    return finish(
        GsVulkanProbeStatus::Ready,
        selected
            ? "selected " + selected->name
            : "selected a suitable Vulkan device");
#endif
}

#if PS2X_HAS_GS_VULKAN
namespace
{
#include "shaders/ps2_gs_ct32_sprite_spv.inc"
#include "shaders/ps2_gs_depth_ct32_sprite_spv.inc"
#include "shaders/ps2_gs_feedback_linear_depth_ct32_sprite_spv.inc"
#include "shaders/ps2_gs_feedback_nearest_depth_ct32_triangle_spv.inc"
#include "shaders/ps2_gs_gouraud_depth_ct32_triangle_spv.inc"
#include "shaders/ps2_gs_t8_gouraud_depth_ct32_triangle_spv.inc"
#include "shaders/ps2_gs_ct32_triangle_spv.inc"
#include "shaders/ps2_gs_linear_ct32_sprite_spv.inc"
#include "shaders/ps2_gs_memory_cases_spv.inc"
#include "shaders/ps2_gs_nearest_ct32_sprite_spv.inc"
#include "shaders/ps2_gs_vram_noop_spv.inc"

    static_assert(sizeof(kGsCt32SpriteShaderSpv) == 7536u);
    static_assert(kGsCt32SpriteShaderSpv[0] == 0x07230203u);
    static_assert(sizeof(kGsDepthCt32SpriteShaderSpv) == 17196u);
    static_assert(kGsDepthCt32SpriteShaderSpv[0] == 0x07230203u);
    static_assert(
        sizeof(kGsFeedbackLinearDepthCt32SpriteShaderSpv) == 46420u);
    static_assert(
        kGsFeedbackLinearDepthCt32SpriteShaderSpv[0] == 0x07230203u);
    static_assert(
        sizeof(kGsFeedbackNearestDepthCt32TriangleShaderSpv) == 67552u);
    static_assert(
        kGsFeedbackNearestDepthCt32TriangleShaderSpv[0] == 0x07230203u);
    static_assert(
        sizeof(kGsGouraudDepthCt32TriangleShaderSpv) == 61112u);
    static_assert(
        kGsGouraudDepthCt32TriangleShaderSpv[0] == 0x07230203u);
    static_assert(
        sizeof(kGsT8GouraudDepthCt32TriangleShaderSpv) == 218876u);
    static_assert(
        kGsT8GouraudDepthCt32TriangleShaderSpv[0] == 0x07230203u);
    static_assert(sizeof(kGsCt32TriangleShaderSpv) == 10200u);
    static_assert(kGsCt32TriangleShaderSpv[0] == 0x07230203u);
    static_assert(sizeof(kGsLinearCt32SpriteShaderSpv) == 38900u);
    static_assert(kGsLinearCt32SpriteShaderSpv[0] == 0x07230203u);
    static_assert(sizeof(kGsMemoryCasesShaderSpv) == 10988u);
    static_assert(kGsMemoryCasesShaderSpv[0] == 0x07230203u);
    static_assert(sizeof(kGsNearestCt32SpriteShaderSpv) == 14128u);
    static_assert(kGsNearestCt32SpriteShaderSpv[0] == 0x07230203u);
    static_assert(sizeof(kGsVramNoopShaderSpv) == 1112u);
    static_assert(kGsVramNoopShaderSpv[0] == 0x07230203u);

    template <typename T>
    T loadDeviceProc(PFN_vkGetDeviceProcAddr getDeviceProcAddr,
                     VkDevice device,
                     const char *name) noexcept
    {
        return reinterpret_cast<T>(getDeviceProcAddr(device, name));
    }

    struct DeviceFunctions
    {
        PFN_vkDestroyDevice destroyDevice = nullptr;
        PFN_vkDeviceWaitIdle deviceWaitIdle = nullptr;
        PFN_vkGetDeviceQueue getDeviceQueue = nullptr;
        PFN_vkCreateBuffer createBuffer = nullptr;
        PFN_vkDestroyBuffer destroyBuffer = nullptr;
        PFN_vkGetBufferMemoryRequirements getBufferMemoryRequirements = nullptr;
        PFN_vkAllocateMemory allocateMemory = nullptr;
        PFN_vkFreeMemory freeMemory = nullptr;
        PFN_vkBindBufferMemory bindBufferMemory = nullptr;
        PFN_vkMapMemory mapMemory = nullptr;
        PFN_vkUnmapMemory unmapMemory = nullptr;
        PFN_vkFlushMappedMemoryRanges flushMappedMemoryRanges = nullptr;
        PFN_vkInvalidateMappedMemoryRanges invalidateMappedMemoryRanges = nullptr;
        PFN_vkCreateDescriptorSetLayout createDescriptorSetLayout = nullptr;
        PFN_vkDestroyDescriptorSetLayout destroyDescriptorSetLayout = nullptr;
        PFN_vkCreatePipelineLayout createPipelineLayout = nullptr;
        PFN_vkDestroyPipelineLayout destroyPipelineLayout = nullptr;
        PFN_vkCreateShaderModule createShaderModule = nullptr;
        PFN_vkDestroyShaderModule destroyShaderModule = nullptr;
        PFN_vkCreateComputePipelines createComputePipelines = nullptr;
        PFN_vkDestroyPipeline destroyPipeline = nullptr;
        PFN_vkCreateDescriptorPool createDescriptorPool = nullptr;
        PFN_vkDestroyDescriptorPool destroyDescriptorPool = nullptr;
        PFN_vkAllocateDescriptorSets allocateDescriptorSets = nullptr;
        PFN_vkUpdateDescriptorSets updateDescriptorSets = nullptr;
        PFN_vkCreateCommandPool createCommandPool = nullptr;
        PFN_vkDestroyCommandPool destroyCommandPool = nullptr;
        PFN_vkAllocateCommandBuffers allocateCommandBuffers = nullptr;
        PFN_vkResetCommandBuffer resetCommandBuffer = nullptr;
        PFN_vkBeginCommandBuffer beginCommandBuffer = nullptr;
        PFN_vkEndCommandBuffer endCommandBuffer = nullptr;
        PFN_vkCreateFence createFence = nullptr;
        PFN_vkDestroyFence destroyFence = nullptr;
        PFN_vkResetFences resetFences = nullptr;
        PFN_vkWaitForFences waitForFences = nullptr;
        PFN_vkQueueSubmit queueSubmit = nullptr;
        PFN_vkCmdPipelineBarrier cmdPipelineBarrier = nullptr;
        PFN_vkCmdCopyBuffer cmdCopyBuffer = nullptr;
        PFN_vkCmdBindPipeline cmdBindPipeline = nullptr;
        PFN_vkCmdBindDescriptorSets cmdBindDescriptorSets = nullptr;
        PFN_vkCmdPushConstants cmdPushConstants = nullptr;
        PFN_vkCmdDispatch cmdDispatch = nullptr;

        bool load(PFN_vkGetDeviceProcAddr getDeviceProcAddr,
                  VkDevice device,
                  std::string &error)
        {
#define PS2X_LOAD_DEVICE_PROC(member, type, name)                         \
            member = loadDeviceProc<type>(                               \
                getDeviceProcAddr, device, #name);                       \
            if (!member)                                                 \
            {                                                            \
                error = "Vulkan device is missing " #name;              \
                return false;                                            \
            }
            PS2X_LOAD_DEVICE_PROC(destroyDevice, PFN_vkDestroyDevice,
                                  vkDestroyDevice)
            PS2X_LOAD_DEVICE_PROC(deviceWaitIdle, PFN_vkDeviceWaitIdle,
                                  vkDeviceWaitIdle)
            PS2X_LOAD_DEVICE_PROC(getDeviceQueue, PFN_vkGetDeviceQueue,
                                  vkGetDeviceQueue)
            PS2X_LOAD_DEVICE_PROC(createBuffer, PFN_vkCreateBuffer,
                                  vkCreateBuffer)
            PS2X_LOAD_DEVICE_PROC(destroyBuffer, PFN_vkDestroyBuffer,
                                  vkDestroyBuffer)
            PS2X_LOAD_DEVICE_PROC(
                getBufferMemoryRequirements,
                PFN_vkGetBufferMemoryRequirements,
                vkGetBufferMemoryRequirements)
            PS2X_LOAD_DEVICE_PROC(allocateMemory, PFN_vkAllocateMemory,
                                  vkAllocateMemory)
            PS2X_LOAD_DEVICE_PROC(freeMemory, PFN_vkFreeMemory,
                                  vkFreeMemory)
            PS2X_LOAD_DEVICE_PROC(bindBufferMemory, PFN_vkBindBufferMemory,
                                  vkBindBufferMemory)
            PS2X_LOAD_DEVICE_PROC(mapMemory, PFN_vkMapMemory, vkMapMemory)
            PS2X_LOAD_DEVICE_PROC(unmapMemory, PFN_vkUnmapMemory,
                                  vkUnmapMemory)
            PS2X_LOAD_DEVICE_PROC(
                flushMappedMemoryRanges,
                PFN_vkFlushMappedMemoryRanges,
                vkFlushMappedMemoryRanges)
            PS2X_LOAD_DEVICE_PROC(
                invalidateMappedMemoryRanges,
                PFN_vkInvalidateMappedMemoryRanges,
                vkInvalidateMappedMemoryRanges)
            PS2X_LOAD_DEVICE_PROC(
                createDescriptorSetLayout,
                PFN_vkCreateDescriptorSetLayout,
                vkCreateDescriptorSetLayout)
            PS2X_LOAD_DEVICE_PROC(
                destroyDescriptorSetLayout,
                PFN_vkDestroyDescriptorSetLayout,
                vkDestroyDescriptorSetLayout)
            PS2X_LOAD_DEVICE_PROC(createPipelineLayout,
                                  PFN_vkCreatePipelineLayout,
                                  vkCreatePipelineLayout)
            PS2X_LOAD_DEVICE_PROC(destroyPipelineLayout,
                                  PFN_vkDestroyPipelineLayout,
                                  vkDestroyPipelineLayout)
            PS2X_LOAD_DEVICE_PROC(createShaderModule,
                                  PFN_vkCreateShaderModule,
                                  vkCreateShaderModule)
            PS2X_LOAD_DEVICE_PROC(destroyShaderModule,
                                  PFN_vkDestroyShaderModule,
                                  vkDestroyShaderModule)
            PS2X_LOAD_DEVICE_PROC(createComputePipelines,
                                  PFN_vkCreateComputePipelines,
                                  vkCreateComputePipelines)
            PS2X_LOAD_DEVICE_PROC(destroyPipeline, PFN_vkDestroyPipeline,
                                  vkDestroyPipeline)
            PS2X_LOAD_DEVICE_PROC(createDescriptorPool,
                                  PFN_vkCreateDescriptorPool,
                                  vkCreateDescriptorPool)
            PS2X_LOAD_DEVICE_PROC(destroyDescriptorPool,
                                  PFN_vkDestroyDescriptorPool,
                                  vkDestroyDescriptorPool)
            PS2X_LOAD_DEVICE_PROC(allocateDescriptorSets,
                                  PFN_vkAllocateDescriptorSets,
                                  vkAllocateDescriptorSets)
            PS2X_LOAD_DEVICE_PROC(updateDescriptorSets,
                                  PFN_vkUpdateDescriptorSets,
                                  vkUpdateDescriptorSets)
            PS2X_LOAD_DEVICE_PROC(createCommandPool,
                                  PFN_vkCreateCommandPool,
                                  vkCreateCommandPool)
            PS2X_LOAD_DEVICE_PROC(destroyCommandPool,
                                  PFN_vkDestroyCommandPool,
                                  vkDestroyCommandPool)
            PS2X_LOAD_DEVICE_PROC(allocateCommandBuffers,
                                  PFN_vkAllocateCommandBuffers,
                                  vkAllocateCommandBuffers)
            PS2X_LOAD_DEVICE_PROC(resetCommandBuffer,
                                  PFN_vkResetCommandBuffer,
                                  vkResetCommandBuffer)
            PS2X_LOAD_DEVICE_PROC(beginCommandBuffer,
                                  PFN_vkBeginCommandBuffer,
                                  vkBeginCommandBuffer)
            PS2X_LOAD_DEVICE_PROC(endCommandBuffer,
                                  PFN_vkEndCommandBuffer,
                                  vkEndCommandBuffer)
            PS2X_LOAD_DEVICE_PROC(createFence, PFN_vkCreateFence,
                                  vkCreateFence)
            PS2X_LOAD_DEVICE_PROC(destroyFence, PFN_vkDestroyFence,
                                  vkDestroyFence)
            PS2X_LOAD_DEVICE_PROC(resetFences, PFN_vkResetFences,
                                  vkResetFences)
            PS2X_LOAD_DEVICE_PROC(waitForFences, PFN_vkWaitForFences,
                                  vkWaitForFences)
            PS2X_LOAD_DEVICE_PROC(queueSubmit, PFN_vkQueueSubmit,
                                  vkQueueSubmit)
            PS2X_LOAD_DEVICE_PROC(cmdPipelineBarrier,
                                  PFN_vkCmdPipelineBarrier,
                                  vkCmdPipelineBarrier)
            PS2X_LOAD_DEVICE_PROC(cmdCopyBuffer, PFN_vkCmdCopyBuffer,
                                  vkCmdCopyBuffer)
            PS2X_LOAD_DEVICE_PROC(cmdBindPipeline,
                                  PFN_vkCmdBindPipeline,
                                  vkCmdBindPipeline)
            PS2X_LOAD_DEVICE_PROC(cmdBindDescriptorSets,
                                  PFN_vkCmdBindDescriptorSets,
                                  vkCmdBindDescriptorSets)
            PS2X_LOAD_DEVICE_PROC(cmdPushConstants,
                                  PFN_vkCmdPushConstants,
                                  vkCmdPushConstants)
            PS2X_LOAD_DEVICE_PROC(cmdDispatch, PFN_vkCmdDispatch,
                                  vkCmdDispatch)
#undef PS2X_LOAD_DEVICE_PROC
            return true;
        }
    };

    struct BufferAllocation
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize allocationSize = 0u;
        bool coherent = false;
    };

    struct PageCopyPlan
    {
        std::array<VkBufferCopy, GS_VRAM_PAGE_COUNT> regions{};
        uint32_t regionCount = 0u;
        size_t pageCount = 0u;
        VkDeviceSize byteCount = 0u;
    };

    PageCopyPlan buildPageCopyPlan(
        const GsVramPageMask &pages,
        bool upload) noexcept
    {
        PageCopyPlan plan{};
        VkDeviceSize packedOffset = 0u;
        size_t page = 0u;
        while (page < GS_VRAM_PAGE_COUNT)
        {
            if (!pages.test(page))
            {
                ++page;
                continue;
            }

            const size_t firstPage = page;
            do
            {
                ++page;
            } while (page < GS_VRAM_PAGE_COUNT && pages.test(page));

            const VkDeviceSize vramOffset =
                firstPage * GS_VRAM_PAGE_SIZE;
            const VkDeviceSize byteCount =
                (page - firstPage) * GS_VRAM_PAGE_SIZE;
            VkBufferCopy &region = plan.regions[plan.regionCount++];
            region.srcOffset = upload ? packedOffset : vramOffset;
            region.dstOffset = upload ? vramOffset : packedOffset;
            region.size = byteCount;
            packedOffset += byteCount;
        }
        plan.pageCount = pages.count();
        plan.byteCount = packedOffset;
        return plan;
    }

    class VulkanExecutionContext final
    {
    public:
        VulkanExecutionContext() = default;
        ~VulkanExecutionContext()
        {
            shutdown();
        }

        VulkanExecutionContext(const VulkanExecutionContext &) = delete;
        VulkanExecutionContext &operator=(
            const VulkanExecutionContext &) = delete;

        bool initialize(const GsVulkanServiceConfig &config,
                        GsVulkanCapabilityReport &report,
                        std::string &error);
        bool roundTrip(std::span<const uint8_t> input,
                       std::vector<uint8_t> &output,
                       GsVulkanCapabilityReport &report,
                       GsVulkanServiceStatistics &statistics,
                       std::string &error);
        bool executeMemoryCases(
            std::span<const uint8_t> input,
            std::span<const GsVulkanMemoryCase> cases,
            std::vector<uint8_t> &output,
            std::vector<GsVulkanMemoryResult> &results,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeCt32Sprite(
            std::span<const uint8_t> input,
            const GsVulkanCt32Sprite &sprite,
            std::vector<uint8_t> &output,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeDepthCt32Sprite(
            std::span<const uint8_t> input,
            const GsVulkanDepthCt32Sprite &sprite,
            std::vector<uint8_t> &output,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeNearestCt32Sprite(
            std::span<const uint8_t> input,
            const GsVulkanNearestCt32Sprite &sprite,
            std::vector<uint8_t> &output,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeLinearCt32Sprite(
            std::span<const uint8_t> input,
            const GsVulkanLinearCt32Sprite &sprite,
            std::vector<uint8_t> &output,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeFeedbackLinearDepthCt32Sprite(
            std::span<const uint8_t> input,
            std::span<const uint8_t> feedbackSnapshot,
            const GsVulkanFeedbackLinearDepthCt32Sprite &sprite,
            std::vector<uint8_t> &output,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeCt32Triangle(
            std::span<const uint8_t> input,
            const GsVulkanCt32Triangle &triangle,
            std::vector<uint8_t> &output,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeGouraudDepthCt32Triangle(
            std::span<const uint8_t> input,
            const GsVulkanGouraudDepthCt32Triangle &triangle,
            std::vector<uint8_t> &output,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeT8GouraudDepthCt32Triangle(
            std::span<const uint8_t> input,
            const GsVulkanT8GouraudDepthCt32Triangle &triangle,
            std::vector<uint8_t> &output,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool uploadVramPages(
            std::span<const uint8_t> source,
            const GsVramPageMask &pages,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool downloadVramPages(
            std::span<uint8_t> destination,
            const GsVramPageMask &pages,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool downloadFeedbackSnapshot(
            std::span<uint8_t> destination,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeResidentCt32Sprites(
            std::span<const GsVulkanCt32Sprite> sprites,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeResidentDepthCt32Sprites(
            std::span<const GsVulkanDepthCt32Sprite> sprites,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeResidentFeedbackLinearDepthCt32Sprites(
            std::span<const uint8_t> feedbackSnapshot,
            std::span<const GsVulkanFeedbackLinearDepthCt32Sprite> sprites,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeResidentFeedbackNearestDepthCt32Triangles(
            std::span<const uint8_t> feedbackSnapshot,
            GsVulkanFeedbackSnapshotMode snapshotMode,
            std::span<
                const GsVulkanFeedbackNearestDepthCt32Triangle> triangles,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeResidentNearestCt32Sprites(
            std::span<const GsVulkanNearestCt32Sprite> sprites,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeResidentLinearCt32Sprites(
            std::span<const GsVulkanLinearCt32Sprite> sprites,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeResidentCt32Triangles(
            std::span<const GsVulkanCt32Triangle> triangles,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeResidentGouraudDepthCt32Triangles(
            std::span<const GsVulkanGouraudDepthCt32Triangle> triangles,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeResidentT8GouraudDepthCt32Triangles(
            std::span<const GsVulkanResidentT8GouraudDepthCt32Triangle> triangles,
            std::span<const GsVulkanT8Palette> palettes,
            std::span<const uint8_t> pageUploadSource,
            const GsVramPageMask &pageUploadPages,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        void refreshDiagnostics(GsVulkanCapabilityReport &report,
                                GsVulkanServiceStatistics &statistics) const;
        void shutdown() noexcept;
        [[nodiscard]] bool healthy() const noexcept
        {
            return m_healthy;
        }

    private:
        bool createInstance(const GsVulkanProbeConfig &config,
                            GsVulkanCapabilityReport &report,
                            std::string &error);
        bool createDevice(const GsVulkanDeviceReport &selected,
                          GsVulkanCapabilityReport &report,
                          std::string &error);
        bool createResources(GsVulkanCapabilityReport &report,
                             std::string &error);
        bool createBuffer(VkDeviceSize size,
                          VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags required,
                          VkMemoryPropertyFlags preferred,
                          BufferAllocation &allocation,
                          GsVulkanCapabilityReport &report,
                          std::string &error);
        bool executeKernel(
            std::span<const uint8_t> input,
            std::span<const GsVulkanMemoryCase> cases,
            std::span<const uint8_t> feedbackSnapshot,
            VkPipeline pipeline,
            uint32_t groupCountX,
            uint32_t groupCountY,
            const void *pushConstants,
            uint32_t pushConstantSize,
            std::vector<uint8_t> &output,
            std::vector<GsVulkanMemoryResult> *results,
            std::string_view operationName,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool executeKernelWithAuxiliary(
            std::span<const uint8_t> input,
            std::span<const GsVulkanMemoryCase> cases,
            std::span<const uint8_t> auxiliaryInput,
            std::span<const uint8_t> auxiliaryResultInput,
            std::span<const uint8_t> feedbackSnapshot,
            VkPipeline pipeline,
            uint32_t groupCountX,
            uint32_t groupCountY,
            const void *pushConstants,
            uint32_t pushConstantSize,
            std::vector<uint8_t> &output,
            std::vector<GsVulkanMemoryResult> *results,
            std::string_view operationName,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool flushMappedAllocation(
            const BufferAllocation &allocation,
            std::string_view label,
            GsVulkanCapabilityReport &report,
            std::string &error);
        bool invalidateMappedAllocation(
            const BufferAllocation &allocation,
            std::string_view label,
            GsVulkanCapabilityReport &report,
            std::string &error);
        bool beginCommands(
            GsVulkanCapabilityReport &report,
            std::string &error);
        bool submitCommands(
            std::string_view operationName,
            uint64_t shaderDispatchCount,
            uint64_t pipelineBarrierCount,
            uint64_t pipelineBindCount,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        bool finishOperation(
            std::string_view operationName,
            uint32_t validationErrorsBefore,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error);
        int32_t findMemoryType(uint32_t typeBits,
                               VkMemoryPropertyFlags required,
                               VkMemoryPropertyFlags preferred) const noexcept;
        bool checkResult(VkResult result,
                         std::string_view operation,
                         GsVulkanCapabilityReport &report,
                         std::string &error,
                         GsVulkanProbeStatus failureStatus);

        DynamicLibrary m_library;
        ValidationCounters m_validation;
        InstanceOwner m_instance;
        PFN_vkGetInstanceProcAddr m_getInstanceProcAddr = nullptr;
        PFN_vkGetDeviceProcAddr m_getDeviceProcAddr = nullptr;
        VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
        VkPhysicalDeviceMemoryProperties m_memoryProperties{};
        VkDevice m_device = VK_NULL_HANDLE;
        DeviceFunctions m_functions{};
        VkQueue m_queue = VK_NULL_HANDLE;
        uint32_t m_queueFamilyIndex = 0u;
        BufferAllocation m_vram;
        BufferAllocation m_staging;
        BufferAllocation m_feedbackSnapshot;
        BufferAllocation m_feedbackStaging;
        BufferAllocation m_memoryCases;
        BufferAllocation m_memoryResults;
        void *m_stagingMap = nullptr;
        void *m_feedbackStagingMap = nullptr;
        void *m_memoryCasesMap = nullptr;
        void *m_memoryResultsMap = nullptr;
        VkDescriptorSetLayout m_descriptorSetLayout = VK_NULL_HANDLE;
        VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
        VkShaderModule m_noopShaderModule = VK_NULL_HANDLE;
        VkPipeline m_noopPipeline = VK_NULL_HANDLE;
        VkShaderModule m_memoryShaderModule = VK_NULL_HANDLE;
        VkPipeline m_memoryPipeline = VK_NULL_HANDLE;
        VkShaderModule m_spriteShaderModule = VK_NULL_HANDLE;
        VkPipeline m_spritePipeline = VK_NULL_HANDLE;
        VkShaderModule m_depthCt32SpriteShaderModule = VK_NULL_HANDLE;
        VkPipeline m_depthCt32SpritePipeline = VK_NULL_HANDLE;
        VkShaderModule m_nearestCt32SpriteShaderModule = VK_NULL_HANDLE;
        VkPipeline m_nearestCt32SpritePipeline = VK_NULL_HANDLE;
        VkShaderModule m_linearCt32SpriteShaderModule = VK_NULL_HANDLE;
        VkPipeline m_linearCt32SpritePipeline = VK_NULL_HANDLE;
        VkShaderModule m_feedbackLinearDepthCt32SpriteShaderModule =
            VK_NULL_HANDLE;
        VkPipeline m_feedbackLinearDepthCt32SpritePipeline =
            VK_NULL_HANDLE;
        VkShaderModule m_feedbackNearestDepthCt32TriangleShaderModule =
            VK_NULL_HANDLE;
        VkPipeline m_feedbackNearestDepthCt32TrianglePipeline =
            VK_NULL_HANDLE;
        VkShaderModule m_triangleShaderModule = VK_NULL_HANDLE;
        VkPipeline m_trianglePipeline = VK_NULL_HANDLE;
        VkShaderModule m_gouraudDepthCt32TriangleShaderModule =
            VK_NULL_HANDLE;
        VkPipeline m_gouraudDepthCt32TrianglePipeline =
            VK_NULL_HANDLE;
        VkShaderModule m_t8GouraudDepthCt32TriangleShaderModule =
            VK_NULL_HANDLE;
        VkPipeline m_t8GouraudDepthCt32TrianglePipeline =
            VK_NULL_HANDLE;
        VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
        VkCommandPool m_commandPool = VK_NULL_HANDLE;
        VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
        VkFence m_fence = VK_NULL_HANDLE;
        uint64_t m_fenceTimeoutNanoseconds = 0u;
        uint64_t m_pipelineCacheMisses = 0u;
        bool m_exactCt32Triangle = false;
        bool m_exactGouraudDepthCt32Triangle = false;
        bool m_exactT8GouraudDepthCt32Triangle = false;
        bool m_exactFeedbackNearestDepthCt32Triangle = false;
        bool m_feedbackSnapshotResidentValid = false;
        bool m_healthy = false;
    };

    bool VulkanExecutionContext::checkResult(
        VkResult result,
        std::string_view operation,
        GsVulkanCapabilityReport &report,
        std::string &error,
        GsVulkanProbeStatus failureStatus)
    {
        if (result == VK_SUCCESS)
            return true;
        error = std::string(operation) + " failed: " +
                resultDescription(result);
        report.status = result == VK_ERROR_DEVICE_LOST
            ? GsVulkanProbeStatus::DeviceLost
            : failureStatus;
        report.detail = error;
        if (result == VK_ERROR_DEVICE_LOST)
            m_healthy = false;
        return false;
    }

    bool VulkanExecutionContext::createInstance(
        const GsVulkanProbeConfig &config,
        GsVulkanCapabilityReport &report,
        std::string &error)
    {
        if (!openVulkanLoader(config, m_library, error))
        {
            report.status = GsVulkanProbeStatus::LoaderUnavailable;
            report.detail = error;
            return false;
        }

        m_getInstanceProcAddr =
            reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                m_library.symbol("vkGetInstanceProcAddr"));
        if (!m_getInstanceProcAddr)
        {
            error = "Vulkan loader does not export vkGetInstanceProcAddr";
            report.status = GsVulkanProbeStatus::LoaderInvalid;
            report.detail = error;
            return false;
        }

        const auto enumerateExtensions =
            loadInstanceProc<PFN_vkEnumerateInstanceExtensionProperties>(
                m_getInstanceProcAddr, VK_NULL_HANDLE,
                "vkEnumerateInstanceExtensionProperties");
        const auto createInstanceProc =
            loadInstanceProc<PFN_vkCreateInstance>(
                m_getInstanceProcAddr, VK_NULL_HANDLE,
                "vkCreateInstance");
        if (!enumerateExtensions || !createInstanceProc)
        {
            error = "Vulkan loader is missing instance-creation entry points";
            report.status = GsVulkanProbeStatus::LoaderInvalid;
            report.detail = error;
            return false;
        }

        std::vector<VkExtensionProperties> extensions;
        const VkResult extensionResult =
            enumerateValues<VkExtensionProperties>(
                [&](uint32_t *count, VkExtensionProperties *values)
                {
                    return enumerateExtensions(nullptr, count, values);
                },
                extensions);
        if (!checkResult(extensionResult,
                         "instance-extension enumeration",
                         report, error,
                         GsVulkanProbeStatus::LoaderInvalid))
        {
            return false;
        }

        const bool debugUtilsAvailable = std::any_of(
            extensions.begin(), extensions.end(),
            [](const VkExtensionProperties &extension)
            {
                return std::strcmp(
                           extension.extensionName,
                           VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0;
            });
        if (config.enableValidation && !debugUtilsAvailable)
        {
            error = "validation requested but debug-utils is unavailable during service creation";
            report.status = GsVulkanProbeStatus::ValidationUnavailable;
            report.detail = error;
            return false;
        }

        std::vector<const char *> enabledExtensions;
        if (config.enableValidation)
            enabledExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#if defined(VK_KHR_portability_enumeration)
        const bool portabilityEnumerationAvailable = std::any_of(
            extensions.begin(), extensions.end(),
            [](const VkExtensionProperties &extension)
            {
                return std::strcmp(
                           extension.extensionName,
                           VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0;
            });
        if (portabilityEnumerationAvailable)
        {
            enabledExtensions.push_back(
                VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        }
#endif

        VkApplicationInfo applicationInfo{};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "PS2Recomp GS Vulkan service";
        applicationInfo.applicationVersion = 1u;
        applicationInfo.pEngineName = "PS2Recomp";
        applicationInfo.engineVersion = 1u;
        applicationInfo.apiVersion = VK_API_VERSION_1_0;

        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo =
            makeDebugCreateInfo(m_validation);
        const char *validationLayer = "VK_LAYER_KHRONOS_validation";
        VkInstanceCreateInfo instanceInfo{};
        instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceInfo.pApplicationInfo = &applicationInfo;
        instanceInfo.enabledExtensionCount =
            static_cast<uint32_t>(enabledExtensions.size());
        instanceInfo.ppEnabledExtensionNames = enabledExtensions.empty()
            ? nullptr
            : enabledExtensions.data();
        if (config.enableValidation)
        {
            instanceInfo.pNext = &debugCreateInfo;
            instanceInfo.enabledLayerCount = 1u;
            instanceInfo.ppEnabledLayerNames = &validationLayer;
        }
#if defined(VK_KHR_portability_enumeration)
        if (portabilityEnumerationAvailable)
        {
            instanceInfo.flags |=
                VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        }
#endif

        const VkResult instanceResult = createInstanceProc(
            &instanceInfo, nullptr, &m_instance.instance);
        if (!checkResult(instanceResult, "vkCreateInstance",
                         report, error,
                         GsVulkanProbeStatus::InstanceCreationFailed))
        {
            return false;
        }
        m_instance.destroyInstance =
            loadInstanceProc<PFN_vkDestroyInstance>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkDestroyInstance");
        if (!m_instance.destroyInstance)
        {
            error = "Vulkan instance is missing vkDestroyInstance";
            report.status = GsVulkanProbeStatus::LoaderInvalid;
            report.detail = error;
            return false;
        }

        if (config.enableValidation)
        {
            const auto createDebugMessenger =
                loadInstanceProc<PFN_vkCreateDebugUtilsMessengerEXT>(
                    m_getInstanceProcAddr, m_instance.instance,
                    "vkCreateDebugUtilsMessengerEXT");
            m_instance.destroyMessenger =
                loadInstanceProc<PFN_vkDestroyDebugUtilsMessengerEXT>(
                    m_getInstanceProcAddr, m_instance.instance,
                    "vkDestroyDebugUtilsMessengerEXT");
            if (!createDebugMessenger || !m_instance.destroyMessenger)
            {
                error = "Vulkan instance is missing debug-utils entry points";
                report.status = GsVulkanProbeStatus::LoaderInvalid;
                report.detail = error;
                return false;
            }
            const VkResult debugResult = createDebugMessenger(
                m_instance.instance, &debugCreateInfo, nullptr,
                &m_instance.messenger);
            if (!checkResult(debugResult, "debug messenger creation",
                             report, error,
                             GsVulkanProbeStatus::InstanceCreationFailed))
            {
                return false;
            }
            report.validationEnabled = true;
        }
        return true;
    }

    bool VulkanExecutionContext::createDevice(
        const GsVulkanDeviceReport &selected,
        GsVulkanCapabilityReport &report,
        std::string &error)
    {
        const auto enumeratePhysicalDevices =
            loadInstanceProc<PFN_vkEnumeratePhysicalDevices>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkEnumeratePhysicalDevices");
        const auto getPhysicalDeviceProperties =
            loadInstanceProc<PFN_vkGetPhysicalDeviceProperties>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkGetPhysicalDeviceProperties");
        const auto getQueueFamilyProperties =
            loadInstanceProc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkGetPhysicalDeviceQueueFamilyProperties");
        const auto getMemoryProperties =
            loadInstanceProc<PFN_vkGetPhysicalDeviceMemoryProperties>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkGetPhysicalDeviceMemoryProperties");
        const auto enumerateDeviceExtensions =
            loadInstanceProc<PFN_vkEnumerateDeviceExtensionProperties>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkEnumerateDeviceExtensionProperties");
        const auto createDeviceProc =
            loadInstanceProc<PFN_vkCreateDevice>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkCreateDevice");
        m_getDeviceProcAddr =
            loadInstanceProc<PFN_vkGetDeviceProcAddr>(
                m_getInstanceProcAddr, m_instance.instance,
                "vkGetDeviceProcAddr");
        if (!enumeratePhysicalDevices || !getPhysicalDeviceProperties ||
            !getQueueFamilyProperties || !getMemoryProperties ||
            !enumerateDeviceExtensions || !createDeviceProc ||
            !m_getDeviceProcAddr)
        {
            error = "Vulkan instance is missing logical-device entry points";
            report.status = GsVulkanProbeStatus::LoaderInvalid;
            report.detail = error;
            return false;
        }

        std::vector<VkPhysicalDevice> physicalDevices;
        const VkResult enumerateResult =
            enumerateValues<VkPhysicalDevice>(
                [&](uint32_t *count, VkPhysicalDevice *values)
                {
                    return enumeratePhysicalDevices(
                        m_instance.instance, count, values);
                },
                physicalDevices);
        if (!checkResult(enumerateResult,
                         "physical-device enumeration",
                         report, error,
                         GsVulkanProbeStatus::DeviceEnumerationFailed))
        {
            return false;
        }

        for (VkPhysicalDevice candidate : physicalDevices)
        {
            VkPhysicalDeviceProperties properties{};
            getPhysicalDeviceProperties(candidate, &properties);
            if (properties.vendorID == selected.vendorId &&
                properties.deviceID == selected.deviceId &&
                selected.name == properties.deviceName)
            {
                m_physicalDevice = candidate;
                break;
            }
        }
        if (m_physicalDevice == VK_NULL_HANDLE)
        {
            error = "the selected Vulkan device disappeared before service creation";
            report.status = GsVulkanProbeStatus::NoSuitableDevice;
            report.detail = error;
            return false;
        }

        uint32_t queueCount = 0u;
        getQueueFamilyProperties(m_physicalDevice, &queueCount, nullptr);
        std::vector<VkQueueFamilyProperties> queues(queueCount);
        if (queueCount != 0u)
        {
            getQueueFamilyProperties(
                m_physicalDevice, &queueCount, queues.data());
            queues.resize(queueCount);
        }
        m_queueFamilyIndex = selected.queueFamilyIndex;
        if (m_queueFamilyIndex >= queues.size() ||
            queues[m_queueFamilyIndex].queueCount == 0u ||
            (queues[m_queueFamilyIndex].queueFlags &
             VK_QUEUE_COMPUTE_BIT) == 0u)
        {
            error = "the selected Vulkan compute queue is no longer available";
            report.status = GsVulkanProbeStatus::NoSuitableDevice;
            report.detail = error;
            return false;
        }

        std::vector<VkExtensionProperties> deviceExtensions;
        const VkResult extensionResult =
            enumerateValues<VkExtensionProperties>(
                [&](uint32_t *count, VkExtensionProperties *values)
                {
                    return enumerateDeviceExtensions(
                        m_physicalDevice, nullptr, count, values);
                },
                deviceExtensions);
        if (!checkResult(extensionResult,
                         "device-extension enumeration",
                         report, error,
                         GsVulkanProbeStatus::DeviceCreationFailed))
        {
            return false;
        }

        std::vector<const char *> enabledDeviceExtensions;
#if defined(VK_KHR_portability_subset)
        const bool portabilitySubsetAvailable = std::any_of(
            deviceExtensions.begin(), deviceExtensions.end(),
            [](const VkExtensionProperties &extension)
            {
                return std::strcmp(
                           extension.extensionName,
                           VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) == 0;
            });
        if (portabilitySubsetAvailable)
        {
            enabledDeviceExtensions.push_back(
                VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
        }
#endif

        const float priority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = m_queueFamilyIndex;
        queueInfo.queueCount = 1u;
        queueInfo.pQueuePriorities = &priority;

        VkPhysicalDeviceFeatures enabledFeatures{};
        enabledFeatures.shaderInt64 =
            (selected.exactCt32Triangle ||
             selected.exactGouraudDepthCt32Triangle ||
             selected.exactT8GouraudDepthCt32Triangle ||
             selected.exactFeedbackNearestDepthCt32Triangle)
                ? VK_TRUE
                : VK_FALSE;
        enabledFeatures.shaderFloat64 =
            (selected.exactT8GouraudDepthCt32Triangle ||
             selected.exactFeedbackNearestDepthCt32Triangle)
                ? VK_TRUE
                : VK_FALSE;
        m_exactCt32Triangle = selected.exactCt32Triangle;
        m_exactGouraudDepthCt32Triangle =
            selected.exactGouraudDepthCt32Triangle;
        m_exactT8GouraudDepthCt32Triangle =
            selected.exactT8GouraudDepthCt32Triangle;
        m_exactFeedbackNearestDepthCt32Triangle =
            selected.exactFeedbackNearestDepthCt32Triangle;
        VkDeviceCreateInfo deviceInfo{};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.queueCreateInfoCount = 1u;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        deviceInfo.enabledExtensionCount =
            static_cast<uint32_t>(enabledDeviceExtensions.size());
        deviceInfo.ppEnabledExtensionNames =
            enabledDeviceExtensions.empty()
                ? nullptr
                : enabledDeviceExtensions.data();
        deviceInfo.pEnabledFeatures = &enabledFeatures;

        const VkResult deviceResult = createDeviceProc(
            m_physicalDevice, &deviceInfo, nullptr, &m_device);
        if (!checkResult(deviceResult, "vkCreateDevice",
                         report, error,
                         GsVulkanProbeStatus::DeviceCreationFailed))
        {
            return false;
        }
        if (!m_functions.load(
                m_getDeviceProcAddr, m_device, error))
        {
            report.status = GsVulkanProbeStatus::LoaderInvalid;
            report.detail = error;
            return false;
        }
        m_functions.getDeviceQueue(
            m_device, m_queueFamilyIndex, 0u, &m_queue);
        if (m_queue == VK_NULL_HANDLE)
        {
            error = "vkGetDeviceQueue returned a null compute queue";
            report.status = GsVulkanProbeStatus::DeviceCreationFailed;
            report.detail = error;
            return false;
        }
        getMemoryProperties(m_physicalDevice, &m_memoryProperties);
        return true;
    }

    int32_t VulkanExecutionContext::findMemoryType(
        uint32_t typeBits,
        VkMemoryPropertyFlags required,
        VkMemoryPropertyFlags preferred) const noexcept
    {
        int32_t bestIndex = -1;
        uint32_t bestScore = 0u;
        for (uint32_t index = 0u;
             index < m_memoryProperties.memoryTypeCount;
             ++index)
        {
            if ((typeBits & (1u << index)) == 0u)
                continue;
            const VkMemoryPropertyFlags flags =
                m_memoryProperties.memoryTypes[index].propertyFlags;
            if ((flags & required) != required)
                continue;

            uint32_t score = 0u;
            VkMemoryPropertyFlags remaining = flags & preferred;
            while (remaining != 0u)
            {
                score += remaining & 1u;
                remaining >>= 1u;
            }
            if (bestIndex < 0 || score > bestScore)
            {
                bestIndex = static_cast<int32_t>(index);
                bestScore = score;
            }
        }
        return bestIndex;
    }

    bool VulkanExecutionContext::createBuffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags required,
        VkMemoryPropertyFlags preferred,
        BufferAllocation &allocation,
        GsVulkanCapabilityReport &report,
        std::string &error)
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult result = m_functions.createBuffer(
            m_device, &bufferInfo, nullptr, &allocation.buffer);
        if (!checkResult(result, "vkCreateBuffer", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
            return false;

        VkMemoryRequirements requirements{};
        m_functions.getBufferMemoryRequirements(
            m_device, allocation.buffer, &requirements);
        const int32_t memoryType = findMemoryType(
            requirements.memoryTypeBits, required, preferred);
        if (memoryType < 0)
        {
            error = "no compatible Vulkan memory type satisfies the buffer requirements";
            return false;
        }

        VkMemoryAllocateInfo allocationInfo{};
        allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocationInfo.allocationSize = requirements.size;
        allocationInfo.memoryTypeIndex =
            static_cast<uint32_t>(memoryType);
        result = m_functions.allocateMemory(
            m_device, &allocationInfo, nullptr, &allocation.memory);
        if (!checkResult(result, "vkAllocateMemory", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
            return false;
        allocation.allocationSize = requirements.size;
        allocation.coherent =
            (m_memoryProperties
                 .memoryTypes[static_cast<uint32_t>(memoryType)]
                 .propertyFlags &
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u;

        result = m_functions.bindBufferMemory(
            m_device, allocation.buffer, allocation.memory, 0u);
        if (!checkResult(result, "vkBindBufferMemory", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
            return false;
        return true;
    }

    bool VulkanExecutionContext::createResources(
        GsVulkanCapabilityReport &report,
        std::string &error)
    {
        constexpr VkDeviceSize memoryCaseBytes =
            sizeof(GsVulkanMemoryCase) *
            static_cast<VkDeviceSize>(GS_VULKAN_MAX_MEMORY_CASES);
        constexpr VkDeviceSize memoryResultBytes =
            GS_VULKAN_AUXILIARY_STORAGE_SIZE;
        if (!createBuffer(
                GS_VULKAN_VRAM_SIZE,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                m_vram, report, error) ||
            !createBuffer(
                GS_VULKAN_VRAM_SIZE,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                m_staging, report, error) ||
            !createBuffer(
                GS_VULKAN_VRAM_SIZE,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                m_feedbackSnapshot, report, error) ||
            !createBuffer(
                GS_VULKAN_VRAM_SIZE,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                m_feedbackStaging, report, error) ||
            !createBuffer(
                memoryCaseBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                m_memoryCases, report, error) ||
            !createBuffer(
                memoryResultBytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                m_memoryResults, report, error))
        {
            if (report.status != GsVulkanProbeStatus::DeviceLost)
                report.status =
                    GsVulkanProbeStatus::ResourceCreationFailed;
            report.detail = error;
            return false;
        }

        VkResult result = m_functions.mapMemory(
            m_device, m_staging.memory, 0u, VK_WHOLE_SIZE, 0u,
            &m_stagingMap);
        if (!checkResult(result, "vkMapMemory", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }
        result = m_functions.mapMemory(
            m_device, m_feedbackStaging.memory,
            0u, VK_WHOLE_SIZE, 0u,
            &m_feedbackStagingMap);
        if (!checkResult(
                result, "vkMapMemory(feedback snapshot)",
                report, error,
                GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }
        result = m_functions.mapMemory(
            m_device, m_memoryCases.memory, 0u, VK_WHOLE_SIZE, 0u,
            &m_memoryCasesMap);
        if (!checkResult(result, "vkMapMemory(memory cases)",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }
        result = m_functions.mapMemory(
            m_device, m_memoryResults.memory, 0u, VK_WHOLE_SIZE, 0u,
            &m_memoryResultsMap);
        if (!checkResult(result, "vkMapMemory(memory results)",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }

        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
        for (uint32_t index = 0u; index < bindings.size(); ++index)
        {
            bindings[index].binding = index;
            bindings[index].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[index].descriptorCount = 1u;
            bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo descriptorLayoutInfo{};
        descriptorLayoutInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorLayoutInfo.bindingCount =
            static_cast<uint32_t>(bindings.size());
        descriptorLayoutInfo.pBindings = bindings.data();
        result = m_functions.createDescriptorSetLayout(
            m_device, &descriptorLayoutInfo, nullptr,
            &m_descriptorSetLayout);
        if (!checkResult(result, "vkCreateDescriptorSetLayout",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.offset = 0u;
        pushConstantRange.size =
            sizeof(GsVulkanFeedbackLinearDepthCt32Sprite);
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1u;
        pipelineLayoutInfo.pSetLayouts = &m_descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1u;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        result = m_functions.createPipelineLayout(
            m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout);
        if (!checkResult(result, "vkCreatePipelineLayout",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }

        const auto createComputePipeline =
            [&](const uint32_t *code, size_t codeSize,
                VkShaderModule &shaderModule, VkPipeline &pipeline,
                std::string_view label)
        {
            // The service eagerly populates its fixed, bounded pipeline set.
            // Each creation attempt is the only possible cache miss; submitted
            // operations reuse one of these handles and account cache hits.
            ++m_pipelineCacheMisses;
            VkShaderModuleCreateInfo shaderInfo{};
            shaderInfo.sType =
                VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            shaderInfo.codeSize = codeSize;
            shaderInfo.pCode = code;
            VkResult createResult = m_functions.createShaderModule(
                m_device, &shaderInfo, nullptr, &shaderModule);
            if (!checkResult(
                    createResult,
                    std::string("vkCreateShaderModule(") +
                        std::string(label) + ')',
                    report, error,
                    GsVulkanProbeStatus::ResourceCreationFailed))
            {
                return false;
            }

            VkPipelineShaderStageCreateInfo stageInfo{};
            stageInfo.sType =
                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stageInfo.module = shaderModule;
            stageInfo.pName = "main";
            VkComputePipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType =
                VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.stage = stageInfo;
            pipelineInfo.layout = m_pipelineLayout;
            createResult = m_functions.createComputePipelines(
                m_device, VK_NULL_HANDLE, 1u, &pipelineInfo, nullptr,
                &pipeline);
            return checkResult(
                createResult,
                std::string("vkCreateComputePipelines(") +
                    std::string(label) + ')',
                report, error,
                GsVulkanProbeStatus::ResourceCreationFailed);
        };
        if (!createComputePipeline(
                kGsVramNoopShaderSpv,
                sizeof(kGsVramNoopShaderSpv),
                m_noopShaderModule, m_noopPipeline, "VRAM no-op") ||
            !createComputePipeline(
                kGsMemoryCasesShaderSpv,
                sizeof(kGsMemoryCasesShaderSpv),
                m_memoryShaderModule, m_memoryPipeline,
                "memory cases") ||
            !createComputePipeline(
                kGsCt32SpriteShaderSpv,
                sizeof(kGsCt32SpriteShaderSpv),
                m_spriteShaderModule, m_spritePipeline,
                "CT32 sprite") ||
            !createComputePipeline(
                kGsDepthCt32SpriteShaderSpv,
                sizeof(kGsDepthCt32SpriteShaderSpv),
                m_depthCt32SpriteShaderModule,
                m_depthCt32SpritePipeline,
                "depth CT32 sprite") ||
            !createComputePipeline(
                kGsNearestCt32SpriteShaderSpv,
                sizeof(kGsNearestCt32SpriteShaderSpv),
                m_nearestCt32SpriteShaderModule,
                m_nearestCt32SpritePipeline,
                "nearest CT32 texture sprite") ||
            !createComputePipeline(
                kGsLinearCt32SpriteShaderSpv,
                sizeof(kGsLinearCt32SpriteShaderSpv),
                m_linearCt32SpriteShaderModule,
                m_linearCt32SpritePipeline,
                "linear CT32 texture sprite") ||
            !createComputePipeline(
                kGsFeedbackLinearDepthCt32SpriteShaderSpv,
                sizeof(kGsFeedbackLinearDepthCt32SpriteShaderSpv),
                m_feedbackLinearDepthCt32SpriteShaderModule,
                m_feedbackLinearDepthCt32SpritePipeline,
                "feedback linear depth CT32 texture sprite"))
        {
            return false;
        }
        if (m_exactCt32Triangle &&
            !createComputePipeline(
                kGsCt32TriangleShaderSpv,
                sizeof(kGsCt32TriangleShaderSpv),
                m_triangleShaderModule, m_trianglePipeline,
                "CT32 triangle"))
        {
            return false;
        }
        if (m_exactGouraudDepthCt32Triangle &&
            !createComputePipeline(
                kGsGouraudDepthCt32TriangleShaderSpv,
                sizeof(kGsGouraudDepthCt32TriangleShaderSpv),
                m_gouraudDepthCt32TriangleShaderModule,
                m_gouraudDepthCt32TrianglePipeline,
                "Gouraud depth CT32 triangle"))
        {
            return false;
        }
        if (m_exactT8GouraudDepthCt32Triangle &&
            !createComputePipeline(
                kGsT8GouraudDepthCt32TriangleShaderSpv,
                sizeof(kGsT8GouraudDepthCt32TriangleShaderSpv),
                m_t8GouraudDepthCt32TriangleShaderModule,
                m_t8GouraudDepthCt32TrianglePipeline,
                "T8 Gouraud depth CT32 triangle"))
        {
            return false;
        }
        if (m_exactFeedbackNearestDepthCt32Triangle &&
            !createComputePipeline(
                kGsFeedbackNearestDepthCt32TriangleShaderSpv,
                sizeof(kGsFeedbackNearestDepthCt32TriangleShaderSpv),
                m_feedbackNearestDepthCt32TriangleShaderModule,
                m_feedbackNearestDepthCt32TrianglePipeline,
                "feedback nearest depth CT32 triangle"))
        {
            return false;
        }

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = 4u;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1u;
        poolInfo.poolSizeCount = 1u;
        poolInfo.pPoolSizes = &poolSize;
        result = m_functions.createDescriptorPool(
            m_device, &poolInfo, nullptr, &m_descriptorPool);
        if (!checkResult(result, "vkCreateDescriptorPool",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }

        VkDescriptorSetAllocateInfo descriptorSetInfo{};
        descriptorSetInfo.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetInfo.descriptorPool = m_descriptorPool;
        descriptorSetInfo.descriptorSetCount = 1u;
        descriptorSetInfo.pSetLayouts = &m_descriptorSetLayout;
        result = m_functions.allocateDescriptorSets(
            m_device, &descriptorSetInfo, &m_descriptorSet);
        if (!checkResult(result, "vkAllocateDescriptorSets",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }

        std::array<VkDescriptorBufferInfo, 4> descriptorBuffers{};
        descriptorBuffers[0].buffer = m_vram.buffer;
        descriptorBuffers[0].range = GS_VULKAN_VRAM_SIZE;
        descriptorBuffers[1].buffer = m_memoryCases.buffer;
        descriptorBuffers[1].range = memoryCaseBytes;
        descriptorBuffers[2].buffer = m_memoryResults.buffer;
        descriptorBuffers[2].range = memoryResultBytes;
        descriptorBuffers[3].buffer = m_feedbackSnapshot.buffer;
        descriptorBuffers[3].range = GS_VULKAN_VRAM_SIZE;
        std::array<VkWriteDescriptorSet, 4> descriptorWrites{};
        for (uint32_t index = 0u;
             index < descriptorWrites.size(); ++index)
        {
            descriptorWrites[index].sType =
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[index].dstSet = m_descriptorSet;
            descriptorWrites[index].dstBinding = index;
            descriptorWrites[index].descriptorCount = 1u;
            descriptorWrites[index].descriptorType =
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[index].pBufferInfo =
                &descriptorBuffers[index];
        }
        m_functions.updateDescriptorSets(
            m_device,
            static_cast<uint32_t>(descriptorWrites.size()),
            descriptorWrites.data(), 0u, nullptr);

        VkCommandPoolCreateInfo commandPoolInfo{};
        commandPoolInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        commandPoolInfo.flags =
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        commandPoolInfo.queueFamilyIndex = m_queueFamilyIndex;
        result = m_functions.createCommandPool(
            m_device, &commandPoolInfo, nullptr, &m_commandPool);
        if (!checkResult(result, "vkCreateCommandPool",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }

        VkCommandBufferAllocateInfo commandBufferInfo{};
        commandBufferInfo.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        commandBufferInfo.commandPool = m_commandPool;
        commandBufferInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        commandBufferInfo.commandBufferCount = 1u;
        result = m_functions.allocateCommandBuffers(
            m_device, &commandBufferInfo, &m_commandBuffer);
        if (!checkResult(result, "vkAllocateCommandBuffers",
                         report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        result = m_functions.createFence(
            m_device, &fenceInfo, nullptr, &m_fence);
        if (!checkResult(result, "vkCreateFence", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return false;
        }
        return true;
    }

    bool VulkanExecutionContext::initialize(
        const GsVulkanServiceConfig &config,
        GsVulkanCapabilityReport &report,
        std::string &error)
    {
        const GsVulkanDeviceReport *selectedDevice =
            report.selectedDevice();
        if (!report.ready() || !selectedDevice)
        {
            error = report.detail.empty()
                ? "Vulkan capability discovery did not select a device"
                : report.detail;
            return false;
        }
        if (config.fenceTimeoutNanoseconds == 0u)
        {
            error = "the Vulkan fence timeout must be non-zero";
            report.status = GsVulkanProbeStatus::ResourceCreationFailed;
            report.detail = error;
            return false;
        }
        const GsVulkanDeviceReport selected = *selectedDevice;
        m_fenceTimeoutNanoseconds = config.fenceTimeoutNanoseconds;
        if (!createInstance(config.probe, report, error) ||
            !createDevice(selected, report, error) ||
            !createResources(report, error))
        {
            return false;
        }

        GsVulkanServiceStatistics diagnostics{};
        refreshDiagnostics(report, diagnostics);
        if (report.validationErrors != 0u)
        {
            error = "Vulkan validation reported errors during service creation";
            report.status = GsVulkanProbeStatus::ValidationError;
            report.detail = error;
            return false;
        }
        report.status = GsVulkanProbeStatus::Ready;
        report.detail = "Vulkan GS service ready on " + selected.name;
        m_healthy = true;
        return true;
    }

    void VulkanExecutionContext::refreshDiagnostics(
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics) const
    {
        report.validationWarnings =
            m_validation.warnings.load(std::memory_order_relaxed);
        report.validationErrors =
            m_validation.errors.load(std::memory_order_relaxed);
        statistics.validationWarnings = report.validationWarnings;
        statistics.validationErrors = report.validationErrors;
        statistics.pipelineCacheMisses = m_pipelineCacheMisses;
        statistics.deviceLost =
            report.status == GsVulkanProbeStatus::DeviceLost;
    }

    bool VulkanExecutionContext::roundTrip(
        std::span<const uint8_t> input,
        std::vector<uint8_t> &output,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (input.size() != GS_VULKAN_VRAM_SIZE)
        {
            error = "Vulkan VRAM round trip requires exactly 4 MiB";
            return false;
        }

        const GsVulkanElementParameters parameters{
            static_cast<uint32_t>(
                GS_VULKAN_VRAM_SIZE / sizeof(uint32_t)),
            0u,
        };
        return executeKernel(
            input, {}, {}, m_noopPipeline,
            GS_VULKAN_NOOP_GROUP_COUNT, 1u,
            &parameters, sizeof(parameters),
            output, nullptr, "VRAM round trip",
            report, statistics, error);
    }

    bool VulkanExecutionContext::executeMemoryCases(
        std::span<const uint8_t> input,
        std::span<const GsVulkanMemoryCase> cases,
        std::vector<uint8_t> &output,
        std::vector<GsVulkanMemoryResult> &results,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (input.size() != GS_VULKAN_VRAM_SIZE)
        {
            error = "Vulkan memory cases require exactly 4 MiB of VRAM";
            return false;
        }
        if (cases.empty() ||
            cases.size() > GS_VULKAN_MAX_MEMORY_CASES)
        {
            error = "Vulkan memory cases require between 1 and " +
                    std::to_string(GS_VULKAN_MAX_MEMORY_CASES) +
                    " records";
            return false;
        }

        const uint32_t caseCount =
            static_cast<uint32_t>(cases.size());
        const uint32_t groupCount =
            (caseCount + GS_VULKAN_NOOP_LOCAL_SIZE - 1u) /
            GS_VULKAN_NOOP_LOCAL_SIZE;
        const GsVulkanElementParameters parameters{caseCount, 0u};
        return executeKernel(
            input, cases, {}, m_memoryPipeline, groupCount, 1u,
            &parameters, sizeof(parameters),
            output, &results, "memory-case batch",
            report, statistics, error);
    }

    bool VulkanExecutionContext::executeCt32Sprite(
        std::span<const uint8_t> input,
        const GsVulkanCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (input.size() != GS_VULKAN_VRAM_SIZE)
        {
            error = "Vulkan CT32 sprite requires exactly 4 MiB of VRAM";
            return false;
        }
        if (const char *validationError =
                ct32SpriteValidationError(sprite))
        {
            error = validationError;
            return false;
        }

        constexpr uint32_t localSize = 8u;
        const uint32_t width = sprite.x1 - sprite.x0;
        const uint32_t height = sprite.y1 - sprite.y0;
        const uint32_t groupCountX =
            (width + localSize - 1u) / localSize;
        const uint32_t groupCountY =
            (height + localSize - 1u) / localSize;
        return executeKernel(
            input, {}, {}, m_spritePipeline,
            groupCountX, groupCountY,
            &sprite, sizeof(sprite), output, nullptr,
            "CT32 sprite", report, statistics, error);
    }

    bool VulkanExecutionContext::executeDepthCt32Sprite(
        std::span<const uint8_t> input,
        const GsVulkanDepthCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (m_depthCt32SpritePipeline == VK_NULL_HANDLE)
        {
            error =
                "Vulkan device does not support exact depth CT32 sprites";
            return false;
        }
        if (input.size() != GS_VULKAN_VRAM_SIZE)
        {
            error =
                "Vulkan depth CT32 sprite requires exactly 4 MiB of VRAM";
            return false;
        }
        if (const char *validationError =
                depthCt32SpriteValidationError(sprite))
        {
            error = validationError;
            return false;
        }

        constexpr uint32_t localSize = 8u;
        const uint32_t width = sprite.boundsX1 - sprite.boundsX0;
        const uint32_t height = sprite.boundsY1 - sprite.boundsY0;
        const uint32_t groupCountX =
            (width + localSize - 1u) / localSize;
        const uint32_t groupCountY =
            (height + localSize - 1u) / localSize;
        return executeKernel(
            input, {}, {}, m_depthCt32SpritePipeline,
            groupCountX, groupCountY,
            &sprite, sizeof(sprite), output, nullptr,
            "depth CT32 sprite", report, statistics, error);
    }

    bool VulkanExecutionContext::executeNearestCt32Sprite(
        std::span<const uint8_t> input,
        const GsVulkanNearestCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (m_nearestCt32SpritePipeline == VK_NULL_HANDLE)
        {
            error =
                "Vulkan device does not support exact nearest CT32 sprites";
            return false;
        }
        if (input.size() != GS_VULKAN_VRAM_SIZE)
        {
            error =
                "Vulkan nearest CT32 sprite requires exactly 4 MiB of VRAM";
            return false;
        }
        if (const char *validationError =
                nearestCt32SpriteValidationError(sprite))
        {
            error = validationError;
            return false;
        }

        constexpr uint32_t localSize = 8u;
        const uint32_t width = sprite.boundsX1 - sprite.boundsX0;
        const uint32_t height = sprite.boundsY1 - sprite.boundsY0;
        const uint32_t groupCountX =
            (width + localSize - 1u) / localSize;
        const uint32_t groupCountY =
            (height + localSize - 1u) / localSize;
        return executeKernel(
            input, {}, {}, m_nearestCt32SpritePipeline,
            groupCountX, groupCountY,
            &sprite, sizeof(sprite), output, nullptr,
            "nearest CT32 texture sprite",
            report, statistics, error);
    }

    bool VulkanExecutionContext::executeLinearCt32Sprite(
        std::span<const uint8_t> input,
        const GsVulkanLinearCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (m_linearCt32SpritePipeline == VK_NULL_HANDLE)
        {
            error =
                "Vulkan device does not support exact linear CT32 sprites";
            return false;
        }
        if (input.size() != GS_VULKAN_VRAM_SIZE)
        {
            error =
                "Vulkan linear CT32 sprite requires exactly 4 MiB of VRAM";
            return false;
        }
        if (const char *validationError =
                linearCt32SpriteValidationError(sprite))
        {
            error = validationError;
            return false;
        }

        constexpr uint32_t localSize = 8u;
        const uint32_t width = sprite.boundsX1 - sprite.boundsX0;
        const uint32_t height = sprite.boundsY1 - sprite.boundsY0;
        const uint32_t groupCountX =
            (width + localSize - 1u) / localSize;
        const uint32_t groupCountY =
            (height + localSize - 1u) / localSize;
        return executeKernel(
            input, {}, {}, m_linearCt32SpritePipeline,
            groupCountX, groupCountY,
            &sprite, sizeof(sprite), output, nullptr,
            "linear CT32 texture sprite",
            report, statistics, error);
    }

    bool VulkanExecutionContext::executeFeedbackLinearDepthCt32Sprite(
        std::span<const uint8_t> input,
        std::span<const uint8_t> feedbackSnapshot,
        const GsVulkanFeedbackLinearDepthCt32Sprite &sprite,
        std::vector<uint8_t> &output,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (m_feedbackLinearDepthCt32SpritePipeline == VK_NULL_HANDLE)
        {
            error =
                "Vulkan device does not support exact feedback linear depth CT32 sprites";
            return false;
        }
        if (input.size() != GS_VULKAN_VRAM_SIZE)
        {
            error =
                "Vulkan feedback linear depth CT32 sprite requires exactly 4 MiB of canonical VRAM";
            return false;
        }
        if (feedbackSnapshot.size() != GS_VULKAN_VRAM_SIZE)
        {
            error =
                "Vulkan feedback linear depth CT32 sprite requires exactly 4 MiB of snapshot VRAM";
            return false;
        }
        if (const char *validationError =
                feedbackLinearDepthCt32SpriteValidationError(sprite))
        {
            error = validationError;
            return false;
        }

        constexpr uint32_t localSize = 8u;
        const uint32_t width = sprite.boundsX1 - sprite.boundsX0;
        const uint32_t height = sprite.boundsY1 - sprite.boundsY0;
        const uint32_t groupCountX =
            (width + localSize - 1u) / localSize;
        const uint32_t groupCountY =
            (height + localSize - 1u) / localSize;
        return executeKernel(
            input, {}, feedbackSnapshot,
            m_feedbackLinearDepthCt32SpritePipeline,
            groupCountX, groupCountY,
            &sprite, sizeof(sprite), output, nullptr,
            "feedback linear depth CT32 texture sprite",
            report, statistics, error);
    }

    bool VulkanExecutionContext::executeCt32Triangle(
        std::span<const uint8_t> input,
        const GsVulkanCt32Triangle &triangle,
        std::vector<uint8_t> &output,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (!m_exactCt32Triangle ||
            m_trianglePipeline == VK_NULL_HANDLE)
        {
            error = "Vulkan device does not support exact CT32 triangles";
            return false;
        }
        if (input.size() != GS_VULKAN_VRAM_SIZE)
        {
            error = "Vulkan CT32 triangle requires exactly 4 MiB of VRAM";
            return false;
        }
        if (const char *validationError =
                ct32TriangleValidationError(triangle))
        {
            error = validationError;
            return false;
        }

        constexpr uint32_t localSize = 8u;
        const uint32_t width =
            triangle.boundsX1 - triangle.boundsX0;
        const uint32_t height =
            triangle.boundsY1 - triangle.boundsY0;
        const uint32_t groupCountX =
            (width + localSize - 1u) / localSize;
        const uint32_t groupCountY =
            (height + localSize - 1u) / localSize;
        return executeKernel(
            input, {}, {}, m_trianglePipeline,
            groupCountX, groupCountY,
            &triangle, sizeof(triangle), output, nullptr,
            "CT32 triangle", report, statistics, error);
    }

    bool VulkanExecutionContext::executeGouraudDepthCt32Triangle(
        std::span<const uint8_t> input,
        const GsVulkanGouraudDepthCt32Triangle &triangle,
        std::vector<uint8_t> &output,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (!m_exactGouraudDepthCt32Triangle ||
            m_gouraudDepthCt32TrianglePipeline == VK_NULL_HANDLE)
        {
            error =
                "Vulkan device does not support exact Gouraud depth CT32 triangles";
            return false;
        }
        if (input.size() != GS_VULKAN_VRAM_SIZE)
        {
            error =
                "Vulkan Gouraud depth CT32 triangle requires exactly 4 MiB of VRAM";
            return false;
        }
        if (const char *validationError =
                gouraudDepthCt32TriangleValidationError(triangle))
        {
            error = validationError;
            return false;
        }

        constexpr uint32_t localSize = 8u;
        const uint32_t width =
            triangle.boundsX1 - triangle.boundsX0;
        const uint32_t height =
            triangle.boundsY1 - triangle.boundsY0;
        const uint32_t groupCountX =
            (width + localSize - 1u) / localSize;
        const uint32_t groupCountY =
            (height + localSize - 1u) / localSize;
        const std::span<const uint8_t> recordBytes(
            reinterpret_cast<const uint8_t *>(&triangle),
            sizeof(triangle));
        const GsVulkanGouraudTriangleParameters parameters{};
        return executeKernelWithAuxiliary(
            input, {}, recordBytes, {}, {},
            m_gouraudDepthCt32TrianglePipeline,
            groupCountX, groupCountY,
            &parameters, sizeof(parameters), output, nullptr,
            "Gouraud depth CT32 triangle",
            report, statistics, error);
    }

    bool VulkanExecutionContext::executeT8GouraudDepthCt32Triangle(
        std::span<const uint8_t> input,
        const GsVulkanT8GouraudDepthCt32Triangle &triangle,
        std::vector<uint8_t> &output,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (!m_exactT8GouraudDepthCt32Triangle ||
            m_t8GouraudDepthCt32TrianglePipeline == VK_NULL_HANDLE)
        {
            error =
                "Vulkan device does not support exact T8 Gouraud depth CT32 triangles";
            return false;
        }
        if (input.size() != GS_VULKAN_VRAM_SIZE)
        {
            error =
                "Vulkan T8 Gouraud depth CT32 triangle requires exactly 4 MiB of VRAM";
            return false;
        }
        if (const char *validationError =
                t8GouraudDepthCt32TriangleValidationError(triangle))
        {
            error = validationError;
            return false;
        }

        constexpr uint32_t localSize = 8u;
        const uint32_t width =
            triangle.boundsX1 - triangle.boundsX0;
        const uint32_t height =
            triangle.boundsY1 - triangle.boundsY0;
        const uint32_t groupCountX =
            (width + localSize - 1u) / localSize;
        const uint32_t groupCountY =
            (height + localSize - 1u) / localSize;
        GsVulkanResidentT8GouraudDepthCt32Triangle compact{};
        std::memcpy(&compact, &triangle, sizeof(compact));
        compact.paletteIndex = 0u;
        const GsVulkanT8TriangleParameters parameters{};
        const std::span<const uint8_t> recordBytes(
            reinterpret_cast<const uint8_t *>(&compact),
            sizeof(compact));
        const std::span<const uint8_t> paletteBytes(
            reinterpret_cast<const uint8_t *>(triangle.palette.data()),
            sizeof(triangle.palette));
        return executeKernelWithAuxiliary(
            input, {}, recordBytes, paletteBytes, {},
            m_t8GouraudDepthCt32TrianglePipeline,
            groupCountX, groupCountY,
            &parameters, sizeof(parameters), output, nullptr,
            "T8 Gouraud depth CT32 triangle",
            report, statistics, error);
    }

    bool VulkanExecutionContext::flushMappedAllocation(
        const BufferAllocation &allocation,
        std::string_view label,
        GsVulkanCapabilityReport &report,
        std::string &error)
    {
        if (allocation.coherent)
            return true;
        VkMappedMemoryRange mappedRange{};
        mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        mappedRange.memory = allocation.memory;
        mappedRange.offset = 0u;
        mappedRange.size = VK_WHOLE_SIZE;
        const VkResult result = m_functions.flushMappedMemoryRanges(
            m_device, 1u, &mappedRange);
        const std::string operation =
            "vkFlushMappedMemoryRanges(" + std::string(label) + ')';
        if (checkResult(
                result, operation, report, error,
                GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return true;
        }
        m_healthy = false;
        return false;
    }

    bool VulkanExecutionContext::invalidateMappedAllocation(
        const BufferAllocation &allocation,
        std::string_view label,
        GsVulkanCapabilityReport &report,
        std::string &error)
    {
        if (allocation.coherent)
            return true;
        VkMappedMemoryRange mappedRange{};
        mappedRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        mappedRange.memory = allocation.memory;
        mappedRange.offset = 0u;
        mappedRange.size = VK_WHOLE_SIZE;
        const VkResult result = m_functions.invalidateMappedMemoryRanges(
            m_device, 1u, &mappedRange);
        const std::string operation =
            "vkInvalidateMappedMemoryRanges(" +
            std::string(label) + ')';
        if (checkResult(
                result, operation, report, error,
                GsVulkanProbeStatus::ResourceCreationFailed))
        {
            return true;
        }
        m_healthy = false;
        return false;
    }

    bool VulkanExecutionContext::beginCommands(
        GsVulkanCapabilityReport &report,
        std::string &error)
    {
        VkResult result = m_functions.resetFences(
            m_device, 1u, &m_fence);
        if (!checkResult(result, "vkResetFences", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            m_healthy = false;
            return false;
        }
        result = m_functions.resetCommandBuffer(m_commandBuffer, 0u);
        if (!checkResult(result, "vkResetCommandBuffer", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            m_healthy = false;
            return false;
        }

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = m_functions.beginCommandBuffer(
            m_commandBuffer, &beginInfo);
        if (!checkResult(result, "vkBeginCommandBuffer", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            m_healthy = false;
            return false;
        }
        return true;
    }

    bool VulkanExecutionContext::submitCommands(
        std::string_view operationName,
        uint64_t shaderDispatchCount,
        uint64_t pipelineBarrierCount,
        uint64_t pipelineBindCount,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        VkResult result = m_functions.endCommandBuffer(m_commandBuffer);
        if (!checkResult(result, "vkEndCommandBuffer", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            m_healthy = false;
            return false;
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1u;
        submitInfo.pCommandBuffers = &m_commandBuffer;
        result = m_functions.queueSubmit(
            m_queue, 1u, &submitInfo, m_fence);
        if (!checkResult(result, "vkQueueSubmit", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            m_healthy = false;
            return false;
        }
        ++statistics.queueSubmissions;
        statistics.shaderDispatches += shaderDispatchCount;
        statistics.pipelineBarriers += pipelineBarrierCount;
        statistics.pipelineBinds += pipelineBindCount;
        statistics.pipelineCacheHits += pipelineBindCount;

        const auto waitStart = std::chrono::steady_clock::now();
        ++statistics.fenceWaits;
        result = m_functions.waitForFences(
            m_device, 1u, &m_fence, VK_TRUE,
            m_fenceTimeoutNanoseconds);
        const auto waitEnd = std::chrono::steady_clock::now();
        statistics.fenceWaitNanoseconds +=
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    waitEnd - waitStart)
                    .count());
        if (result == VK_TIMEOUT)
        {
            ++statistics.fenceTimeouts;
            error = "Vulkan GS " + std::string(operationName) +
                    " fence timed out";
            report.status = GsVulkanProbeStatus::ExecutionTimeout;
            report.detail = error;
            m_healthy = false;
            return false;
        }
        if (!checkResult(result, "vkWaitForFences", report, error,
                         GsVulkanProbeStatus::ResourceCreationFailed))
        {
            m_healthy = false;
            return false;
        }
        return true;
    }

    bool VulkanExecutionContext::finishOperation(
        std::string_view operationName,
        uint32_t validationErrorsBefore,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        refreshDiagnostics(report, statistics);
        if (report.validationErrors == validationErrorsBefore)
            return true;
        error = "Vulkan validation reported an error during the " +
                std::string(operationName);
        report.status = GsVulkanProbeStatus::ValidationError;
        report.detail = error;
        m_healthy = false;
        return false;
    }

    bool VulkanExecutionContext::uploadVramPages(
        std::span<const uint8_t> source,
        const GsVramPageMask &pages,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        const PageCopyPlan plan = buildPageCopyPlan(pages, true);
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (plan.pageCount == 0u ||
            source.size() != GS_VULKAN_VRAM_SIZE)
        {
            error = "invalid Vulkan GS page upload request";
            return false;
        }

        const uint32_t validationErrorsBefore =
            m_validation.errors.load(std::memory_order_relaxed);
        for (uint32_t index = 0u; index < plan.regionCount; ++index)
        {
            const VkBufferCopy &region = plan.regions[index];
            std::memcpy(
                static_cast<uint8_t *>(m_stagingMap) + region.srcOffset,
                source.data() + region.dstOffset,
                static_cast<size_t>(region.size));
        }
        if (!flushMappedAllocation(
                m_staging, "VRAM page upload", report, error) ||
            !beginCommands(report, error))
        {
            return false;
        }

        VkBufferMemoryBarrier stagingBarrier{};
        stagingBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        stagingBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        stagingBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        stagingBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        stagingBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        stagingBarrier.buffer = m_staging.buffer;
        stagingBarrier.offset = 0u;
        stagingBarrier.size = plan.byteCount;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0u, 0u, nullptr, 1u, &stagingBarrier, 0u, nullptr);

        VkBufferMemoryBarrier vramPrepareBarrier{};
        vramPrepareBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        vramPrepareBarrier.srcAccessMask =
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vramPrepareBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vramPrepareBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vramPrepareBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vramPrepareBarrier.buffer = m_vram.buffer;
        vramPrepareBarrier.offset = 0u;
        vramPrepareBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0u, 0u, nullptr, 1u, &vramPrepareBarrier, 0u, nullptr);

        m_functions.cmdCopyBuffer(
            m_commandBuffer, m_staging.buffer, m_vram.buffer,
            plan.regionCount, plan.regions.data());

        VkBufferMemoryBarrier vramCompleteBarrier{};
        vramCompleteBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        vramCompleteBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vramCompleteBarrier.dstAccessMask =
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vramCompleteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vramCompleteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vramCompleteBarrier.buffer = m_vram.buffer;
        vramCompleteBarrier.offset = 0u;
        vramCompleteBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0u, 0u, nullptr, 1u, &vramCompleteBarrier, 0u, nullptr);

        if (!submitCommands(
                "VRAM page upload", 0u, 3u, 0u,
                report, statistics, error) ||
            !finishOperation(
                "VRAM page upload", validationErrorsBefore,
                report, statistics, error))
        {
            return false;
        }

        statistics.bytesUploaded += plan.byteCount;
        statistics.pagesUploaded += plan.pageCount;
        statistics.pageUploadRegions += plan.regionCount;
        error.clear();
        return true;
    }

    bool VulkanExecutionContext::downloadVramPages(
        std::span<uint8_t> destination,
        const GsVramPageMask &pages,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        const PageCopyPlan plan = buildPageCopyPlan(pages, false);
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (plan.pageCount == 0u ||
            destination.size() != GS_VULKAN_VRAM_SIZE)
        {
            error = "invalid Vulkan GS page download request";
            return false;
        }

        const uint32_t validationErrorsBefore =
            m_validation.errors.load(std::memory_order_relaxed);
        if (!beginCommands(report, error))
            return false;

        VkBufferMemoryBarrier vramBarrier{};
        vramBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        vramBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        vramBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vramBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vramBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        vramBarrier.buffer = m_vram.buffer;
        vramBarrier.offset = 0u;
        vramBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0u, 0u, nullptr, 1u, &vramBarrier, 0u, nullptr);

        VkBufferMemoryBarrier stagingPrepareBarrier{};
        stagingPrepareBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        stagingPrepareBarrier.srcAccessMask =
            VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT;
        stagingPrepareBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        stagingPrepareBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        stagingPrepareBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        stagingPrepareBarrier.buffer = m_staging.buffer;
        stagingPrepareBarrier.offset = 0u;
        stagingPrepareBarrier.size = plan.byteCount;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0u, 0u, nullptr, 1u, &stagingPrepareBarrier,
            0u, nullptr);

        m_functions.cmdCopyBuffer(
            m_commandBuffer, m_vram.buffer, m_staging.buffer,
            plan.regionCount, plan.regions.data());

        VkBufferMemoryBarrier stagingCompleteBarrier{};
        stagingCompleteBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        stagingCompleteBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        stagingCompleteBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        stagingCompleteBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        stagingCompleteBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        stagingCompleteBarrier.buffer = m_staging.buffer;
        stagingCompleteBarrier.offset = 0u;
        stagingCompleteBarrier.size = plan.byteCount;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0u, 0u, nullptr, 1u, &stagingCompleteBarrier,
            0u, nullptr);

        if (!submitCommands(
                "VRAM page download", 0u, 3u, 0u,
                report, statistics, error) ||
            !invalidateMappedAllocation(
                m_staging, "VRAM page download", report, error) ||
            !finishOperation(
                "VRAM page download", validationErrorsBefore,
                report, statistics, error))
        {
            return false;
        }

        for (uint32_t index = 0u; index < plan.regionCount; ++index)
        {
            const VkBufferCopy &region = plan.regions[index];
            std::memcpy(
                destination.data() + region.srcOffset,
                static_cast<const uint8_t *>(m_stagingMap) +
                    region.dstOffset,
                static_cast<size_t>(region.size));
        }
        statistics.bytesDownloaded += plan.byteCount;
        statistics.pagesDownloaded += plan.pageCount;
        statistics.pageDownloadRegions += plan.regionCount;
        error.clear();
        return true;
    }

    bool VulkanExecutionContext::downloadFeedbackSnapshot(
        std::span<uint8_t> destination,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (!m_feedbackSnapshotResidentValid ||
            destination.size() != GS_VULKAN_VRAM_SIZE)
        {
            error =
                "Vulkan GS feedback snapshot download requires one resident 4 MiB snapshot";
            return false;
        }

        const uint32_t validationErrorsBefore =
            m_validation.errors.load(std::memory_order_relaxed);
        if (!beginCommands(report, error))
            return false;

        VkBufferMemoryBarrier snapshotBarrier{};
        snapshotBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        snapshotBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        snapshotBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        snapshotBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        snapshotBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        snapshotBarrier.buffer = m_feedbackSnapshot.buffer;
        snapshotBarrier.offset = 0u;
        snapshotBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0u, 0u, nullptr, 1u, &snapshotBarrier, 0u, nullptr);

        VkBufferMemoryBarrier stagingPrepareBarrier{};
        stagingPrepareBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        stagingPrepareBarrier.srcAccessMask =
            VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT;
        stagingPrepareBarrier.dstAccessMask =
            VK_ACCESS_TRANSFER_WRITE_BIT;
        stagingPrepareBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        stagingPrepareBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        stagingPrepareBarrier.buffer = m_staging.buffer;
        stagingPrepareBarrier.offset = 0u;
        stagingPrepareBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0u, 0u, nullptr, 1u, &stagingPrepareBarrier,
            0u, nullptr);

        VkBufferCopy copyRegion{};
        copyRegion.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdCopyBuffer(
            m_commandBuffer,
            m_feedbackSnapshot.buffer,
            m_staging.buffer,
            1u, &copyRegion);

        VkBufferMemoryBarrier stagingCompleteBarrier{};
        stagingCompleteBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        stagingCompleteBarrier.srcAccessMask =
            VK_ACCESS_TRANSFER_WRITE_BIT;
        stagingCompleteBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        stagingCompleteBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        stagingCompleteBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        stagingCompleteBarrier.buffer = m_staging.buffer;
        stagingCompleteBarrier.offset = 0u;
        stagingCompleteBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0u, 0u, nullptr, 1u, &stagingCompleteBarrier,
            0u, nullptr);

        if (!submitCommands(
                "feedback snapshot download", 0u, 3u, 0u,
                report, statistics, error) ||
            !invalidateMappedAllocation(
                m_staging, "feedback snapshot download", report, error) ||
            !finishOperation(
                "feedback snapshot download", validationErrorsBefore,
                report, statistics, error))
        {
            return false;
        }

        std::memcpy(
            destination.data(), m_stagingMap, GS_VULKAN_VRAM_SIZE);
        statistics.bytesDownloaded += GS_VULKAN_VRAM_SIZE;
        error.clear();
        return true;
    }

    bool VulkanExecutionContext::executeResidentCt32Sprites(
        std::span<const GsVulkanCt32Sprite> sprites,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (!validateResidentCt32SpriteBatch(sprites, error))
            return false;

        constexpr uint32_t localSize = 8u;
        const uint32_t validationErrorsBefore =
            m_validation.errors.load(std::memory_order_relaxed);
        if (!beginCommands(report, error))
            return false;

        VkBufferMemoryBarrier prepareBarrier{};
        prepareBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        prepareBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        prepareBarrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        prepareBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.buffer = m_vram.buffer;
        prepareBarrier.offset = 0u;
        prepareBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, 1u, &prepareBarrier, 0u, nullptr);

        m_functions.cmdBindPipeline(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_spritePipeline);
        m_functions.cmdBindDescriptorSets(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayout, 0u, 1u, &m_descriptorSet,
            0u, nullptr);
        for (const GsVulkanCt32Sprite &sprite : sprites)
        {
            const uint32_t groupCountX =
                (sprite.x1 - sprite.x0 + localSize - 1u) / localSize;
            const uint32_t groupCountY =
                (sprite.y1 - sprite.y0 + localSize - 1u) / localSize;
            m_functions.cmdPushConstants(
                m_commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                sizeof(sprite), &sprite);
            m_functions.cmdDispatch(
                m_commandBuffer, groupCountX, groupCountY, 1u);
        }

        VkBufferMemoryBarrier completeBarrier{};
        completeBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        completeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        completeBarrier.dstAccessMask =
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        completeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.buffer = m_vram.buffer;
        completeBarrier.offset = 0u;
        completeBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0u, 0u, nullptr, 1u, &completeBarrier, 0u, nullptr);

        if (!submitCommands(
                "resident CT32 sprite batch", sprites.size(), 2u, 1u,
                report, statistics, error) ||
            !finishOperation(
                "resident CT32 sprite batch", validationErrorsBefore,
                report, statistics, error))
        {
            return false;
        }
        error.clear();
        return true;
    }

    bool VulkanExecutionContext::executeResidentDepthCt32Sprites(
        std::span<const GsVulkanDepthCt32Sprite> sprites,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (m_depthCt32SpritePipeline == VK_NULL_HANDLE)
        {
            error =
                "Vulkan device does not support exact depth CT32 sprites";
            return false;
        }
        if (!validateResidentDepthCt32SpriteBatch(sprites, error))
            return false;

        constexpr uint32_t localSize = 8u;
        const uint32_t validationErrorsBefore =
            m_validation.errors.load(std::memory_order_relaxed);
        if (!beginCommands(report, error))
            return false;

        VkBufferMemoryBarrier prepareBarrier{};
        prepareBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        prepareBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        prepareBarrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        prepareBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.buffer = m_vram.buffer;
        prepareBarrier.offset = 0u;
        prepareBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, 1u, &prepareBarrier, 0u, nullptr);

        m_functions.cmdBindPipeline(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_depthCt32SpritePipeline);
        m_functions.cmdBindDescriptorSets(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayout, 0u, 1u, &m_descriptorSet,
            0u, nullptr);
        uint64_t dependencyBarrierCount = 0u;
        GsVramPageMask priorReadPages;
        GsVramPageMask priorWritePages;
        for (const GsVulkanDepthCt32Sprite &sprite : sprites)
        {
            const DepthCt32SpriteAccessPages access =
                depthCt32SpriteAccessPages(sprite);
            const bool hasDependency =
                priorWritePages.intersects(access.readPages) ||
                priorWritePages.intersects(access.writePages) ||
                priorReadPages.intersects(access.writePages);
            if (hasDependency)
            {
                VkBufferMemoryBarrier dependencyBarrier{};
                dependencyBarrier.sType =
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                dependencyBarrier.srcAccessMask =
                    VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT;
                dependencyBarrier.dstAccessMask =
                    VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT;
                dependencyBarrier.srcQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                dependencyBarrier.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                dependencyBarrier.buffer = m_vram.buffer;
                dependencyBarrier.offset = 0u;
                dependencyBarrier.size = GS_VULKAN_VRAM_SIZE;
                m_functions.cmdPipelineBarrier(
                    m_commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0u, 0u, nullptr, 1u, &dependencyBarrier,
                    0u, nullptr);
                ++dependencyBarrierCount;
                priorReadPages.clear();
                priorWritePages.clear();
            }

            const uint32_t groupCountX =
                (sprite.boundsX1 - sprite.boundsX0 +
                 localSize - 1u) / localSize;
            const uint32_t groupCountY =
                (sprite.boundsY1 - sprite.boundsY0 +
                 localSize - 1u) / localSize;
            m_functions.cmdPushConstants(
                m_commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                sizeof(sprite), &sprite);
            m_functions.cmdDispatch(
                m_commandBuffer, groupCountX, groupCountY, 1u);
            priorReadPages.unionWith(access.readPages);
            priorWritePages.unionWith(access.writePages);
        }

        VkBufferMemoryBarrier completeBarrier{};
        completeBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        completeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        completeBarrier.dstAccessMask =
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        completeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.buffer = m_vram.buffer;
        completeBarrier.offset = 0u;
        completeBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0u, 0u, nullptr, 1u, &completeBarrier, 0u, nullptr);

        if (!submitCommands(
                "resident depth CT32 sprite batch",
                sprites.size(), 2u + dependencyBarrierCount, 1u,
                report, statistics, error) ||
            !finishOperation(
                "resident depth CT32 sprite batch",
                validationErrorsBefore,
                report, statistics, error))
        {
            return false;
        }
        error.clear();
        return true;
    }

    bool VulkanExecutionContext::
    executeResidentFeedbackLinearDepthCt32Sprites(
        std::span<const uint8_t> feedbackSnapshot,
        std::span<const GsVulkanFeedbackLinearDepthCt32Sprite> sprites,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (m_feedbackLinearDepthCt32SpritePipeline == VK_NULL_HANDLE)
        {
            error =
                "Vulkan device does not support exact feedback linear depth CT32 sprites";
            return false;
        }
        if (feedbackSnapshot.size() != GS_VULKAN_VRAM_SIZE)
        {
            error =
                "Vulkan resident feedback linear depth CT32 sprites require exactly 4 MiB of snapshot VRAM";
            return false;
        }
        if (!validateResidentFeedbackLinearDepthCt32SpriteBatch(
                sprites, error))
        {
            return false;
        }

        constexpr uint32_t localSize = 8u;
        const uint32_t validationErrorsBefore =
            m_validation.errors.load(std::memory_order_relaxed);
        std::memcpy(
            m_feedbackStagingMap,
            feedbackSnapshot.data(),
            feedbackSnapshot.size());
        if (!flushMappedAllocation(
                m_feedbackStaging,
                "feedback snapshot", report, error) ||
            !beginCommands(report, error))
        {
            return false;
        }

        VkBufferMemoryBarrier feedbackStagingBarrier{};
        feedbackStagingBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        feedbackStagingBarrier.srcAccessMask =
            VK_ACCESS_HOST_WRITE_BIT;
        feedbackStagingBarrier.dstAccessMask =
            VK_ACCESS_TRANSFER_READ_BIT;
        feedbackStagingBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        feedbackStagingBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        feedbackStagingBarrier.buffer = m_feedbackStaging.buffer;
        feedbackStagingBarrier.offset = 0u;
        feedbackStagingBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0u, 0u, nullptr, 1u, &feedbackStagingBarrier,
            0u, nullptr);

        VkBufferCopy copyRegion{};
        copyRegion.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdCopyBuffer(
            m_commandBuffer,
            m_feedbackStaging.buffer,
            m_feedbackSnapshot.buffer,
            1u, &copyRegion);

        VkBufferMemoryBarrier feedbackUploadBarrier{};
        feedbackUploadBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        feedbackUploadBarrier.srcAccessMask =
            VK_ACCESS_TRANSFER_WRITE_BIT;
        feedbackUploadBarrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT;
        feedbackUploadBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        feedbackUploadBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        feedbackUploadBarrier.buffer = m_feedbackSnapshot.buffer;
        feedbackUploadBarrier.offset = 0u;
        feedbackUploadBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, 1u, &feedbackUploadBarrier,
            0u, nullptr);

        VkBufferMemoryBarrier prepareBarrier{};
        prepareBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        prepareBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        prepareBarrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        prepareBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.buffer = m_vram.buffer;
        prepareBarrier.offset = 0u;
        prepareBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, 1u, &prepareBarrier, 0u, nullptr);

        m_functions.cmdBindPipeline(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_feedbackLinearDepthCt32SpritePipeline);
        m_functions.cmdBindDescriptorSets(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayout, 0u, 1u, &m_descriptorSet,
            0u, nullptr);
        uint64_t dependencyBarrierCount = 0u;
        GsVramPageMask priorReadPages;
        GsVramPageMask priorWritePages;
        for (const GsVulkanFeedbackLinearDepthCt32Sprite &sprite :
             sprites)
        {
            const DepthCt32SpriteAccessPages access =
                feedbackLinearDepthCt32SpriteAccessPages(sprite);
            const bool hasDependency =
                priorWritePages.intersects(access.readPages) ||
                priorWritePages.intersects(access.writePages) ||
                priorReadPages.intersects(access.writePages);
            if (hasDependency)
            {
                VkBufferMemoryBarrier dependencyBarrier{};
                dependencyBarrier.sType =
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                dependencyBarrier.srcAccessMask =
                    VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT;
                dependencyBarrier.dstAccessMask =
                    VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT;
                dependencyBarrier.srcQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                dependencyBarrier.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                dependencyBarrier.buffer = m_vram.buffer;
                dependencyBarrier.offset = 0u;
                dependencyBarrier.size = GS_VULKAN_VRAM_SIZE;
                m_functions.cmdPipelineBarrier(
                    m_commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0u, 0u, nullptr, 1u, &dependencyBarrier,
                    0u, nullptr);
                ++dependencyBarrierCount;
                priorReadPages.clear();
                priorWritePages.clear();
            }

            const uint32_t groupCountX =
                (sprite.boundsX1 - sprite.boundsX0 +
                 localSize - 1u) / localSize;
            const uint32_t groupCountY =
                (sprite.boundsY1 - sprite.boundsY0 +
                 localSize - 1u) / localSize;
            m_functions.cmdPushConstants(
                m_commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                sizeof(sprite), &sprite);
            m_functions.cmdDispatch(
                m_commandBuffer, groupCountX, groupCountY, 1u);
            priorReadPages.unionWith(access.readPages);
            priorWritePages.unionWith(access.writePages);
        }

        VkBufferMemoryBarrier completeBarrier{};
        completeBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        completeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        completeBarrier.dstAccessMask =
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        completeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.buffer = m_vram.buffer;
        completeBarrier.offset = 0u;
        completeBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0u, 0u, nullptr, 1u, &completeBarrier, 0u, nullptr);

        if (!submitCommands(
                "resident feedback linear depth CT32 sprite batch",
                sprites.size(), 4u + dependencyBarrierCount, 1u,
                report, statistics, error) ||
            !finishOperation(
                "resident feedback linear depth CT32 sprite batch",
                validationErrorsBefore,
                report, statistics, error))
        {
            return false;
        }
        statistics.bytesUploaded += GS_VULKAN_VRAM_SIZE;
        error.clear();
        return true;
    }

    bool VulkanExecutionContext::executeResidentNearestCt32Sprites(
        std::span<const GsVulkanNearestCt32Sprite> sprites,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (m_nearestCt32SpritePipeline == VK_NULL_HANDLE)
        {
            error =
                "Vulkan device does not support exact nearest CT32 sprites";
            return false;
        }
        if (!validateResidentNearestCt32SpriteBatch(sprites, error))
            return false;

        constexpr uint32_t localSize = 8u;
        const uint32_t validationErrorsBefore =
            m_validation.errors.load(std::memory_order_relaxed);
        if (!beginCommands(report, error))
            return false;

        VkBufferMemoryBarrier prepareBarrier{};
        prepareBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        prepareBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        prepareBarrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        prepareBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.buffer = m_vram.buffer;
        prepareBarrier.offset = 0u;
        prepareBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, 1u, &prepareBarrier, 0u, nullptr);

        m_functions.cmdBindPipeline(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_nearestCt32SpritePipeline);
        m_functions.cmdBindDescriptorSets(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayout, 0u, 1u, &m_descriptorSet,
            0u, nullptr);
        for (const GsVulkanNearestCt32Sprite &sprite : sprites)
        {
            const uint32_t groupCountX =
                (sprite.boundsX1 - sprite.boundsX0 +
                 localSize - 1u) / localSize;
            const uint32_t groupCountY =
                (sprite.boundsY1 - sprite.boundsY0 +
                 localSize - 1u) / localSize;
            m_functions.cmdPushConstants(
                m_commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                sizeof(sprite), &sprite);
            m_functions.cmdDispatch(
                m_commandBuffer, groupCountX, groupCountY, 1u);
        }

        VkBufferMemoryBarrier completeBarrier{};
        completeBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        completeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        completeBarrier.dstAccessMask =
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        completeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.buffer = m_vram.buffer;
        completeBarrier.offset = 0u;
        completeBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0u, 0u, nullptr, 1u, &completeBarrier, 0u, nullptr);

        if (!submitCommands(
                "resident nearest CT32 sprite batch",
                sprites.size(), 2u, 1u,
                report, statistics, error) ||
            !finishOperation(
                "resident nearest CT32 sprite batch",
                validationErrorsBefore,
                report, statistics, error))
        {
            return false;
        }
        error.clear();
        return true;
    }

    bool VulkanExecutionContext::executeResidentLinearCt32Sprites(
        std::span<const GsVulkanLinearCt32Sprite> sprites,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (m_linearCt32SpritePipeline == VK_NULL_HANDLE)
        {
            error =
                "Vulkan device does not support exact linear CT32 sprites";
            return false;
        }
        if (!validateResidentLinearCt32SpriteBatch(sprites, error))
            return false;

        constexpr uint32_t localSize = 8u;
        const uint32_t validationErrorsBefore =
            m_validation.errors.load(std::memory_order_relaxed);
        if (!beginCommands(report, error))
            return false;

        VkBufferMemoryBarrier prepareBarrier{};
        prepareBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        prepareBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        prepareBarrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        prepareBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.buffer = m_vram.buffer;
        prepareBarrier.offset = 0u;
        prepareBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, 1u, &prepareBarrier, 0u, nullptr);

        m_functions.cmdBindPipeline(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_linearCt32SpritePipeline);
        m_functions.cmdBindDescriptorSets(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayout, 0u, 1u, &m_descriptorSet,
            0u, nullptr);
        uint64_t dependencyBarrierCount = 0u;
        GsVramPageMask priorReadPages;
        GsVramPageMask priorWritePages;
        for (const GsVulkanLinearCt32Sprite &sprite : sprites)
        {
            const GsVramPageMask readPages =
                linearCt32TexturePages(sprite);
            const GsVramPageMask writePages =
                gsVramPagesForSurfaceRect(
                    sprite.framebufferBaseBlock,
                    sprite.framebufferWidth,
                    static_cast<uint8_t>(GSMem::C32),
                    sprite.boundsX0,
                    sprite.boundsY0,
                    sprite.boundsX1 - sprite.boundsX0,
                    sprite.boundsY1 - sprite.boundsY0);
            const bool hasDependency =
                priorWritePages.intersects(readPages) ||
                priorWritePages.intersects(writePages) ||
                priorReadPages.intersects(writePages);
            if (hasDependency)
            {
                VkBufferMemoryBarrier dependencyBarrier{};
                dependencyBarrier.sType =
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                dependencyBarrier.srcAccessMask =
                    VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT;
                dependencyBarrier.dstAccessMask =
                    VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT;
                dependencyBarrier.srcQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                dependencyBarrier.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                dependencyBarrier.buffer = m_vram.buffer;
                dependencyBarrier.offset = 0u;
                dependencyBarrier.size = GS_VULKAN_VRAM_SIZE;
                m_functions.cmdPipelineBarrier(
                    m_commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0u, 0u, nullptr, 1u, &dependencyBarrier,
                    0u, nullptr);
                ++dependencyBarrierCount;
                priorReadPages.clear();
                priorWritePages.clear();
            }
            const uint32_t groupCountX =
                (sprite.boundsX1 - sprite.boundsX0 +
                 localSize - 1u) / localSize;
            const uint32_t groupCountY =
                (sprite.boundsY1 - sprite.boundsY0 +
                 localSize - 1u) / localSize;
            m_functions.cmdPushConstants(
                m_commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                sizeof(sprite), &sprite);
            m_functions.cmdDispatch(
                m_commandBuffer, groupCountX, groupCountY, 1u);
            priorReadPages.unionWith(readPages);
            priorWritePages.unionWith(writePages);
        }

        VkBufferMemoryBarrier completeBarrier{};
        completeBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        completeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        completeBarrier.dstAccessMask =
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        completeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.buffer = m_vram.buffer;
        completeBarrier.offset = 0u;
        completeBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0u, 0u, nullptr, 1u, &completeBarrier, 0u, nullptr);

        if (!submitCommands(
                "resident linear CT32 sprite batch",
                sprites.size(), 2u + dependencyBarrierCount, 1u,
                report, statistics, error) ||
            !finishOperation(
                "resident linear CT32 sprite batch",
                validationErrorsBefore,
                report, statistics, error))
        {
            return false;
        }
        error.clear();
        return true;
    }

    bool VulkanExecutionContext::executeResidentCt32Triangles(
        std::span<const GsVulkanCt32Triangle> triangles,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (!m_exactCt32Triangle ||
            m_trianglePipeline == VK_NULL_HANDLE)
        {
            error = "Vulkan device does not support exact CT32 triangles";
            return false;
        }
        if (!validateResidentCt32TriangleBatch(triangles, error))
            return false;

        constexpr uint32_t localSize = 8u;
        const uint32_t validationErrorsBefore =
            m_validation.errors.load(std::memory_order_relaxed);
        if (!beginCommands(report, error))
            return false;

        VkBufferMemoryBarrier prepareBarrier{};
        prepareBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        prepareBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        prepareBarrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        prepareBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.buffer = m_vram.buffer;
        prepareBarrier.offset = 0u;
        prepareBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, 1u, &prepareBarrier, 0u, nullptr);

        m_functions.cmdBindPipeline(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_trianglePipeline);
        m_functions.cmdBindDescriptorSets(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayout, 0u, 1u, &m_descriptorSet,
            0u, nullptr);
        for (const GsVulkanCt32Triangle &triangle : triangles)
        {
            const uint32_t groupCountX =
                (triangle.boundsX1 - triangle.boundsX0 +
                 localSize - 1u) / localSize;
            const uint32_t groupCountY =
                (triangle.boundsY1 - triangle.boundsY0 +
                 localSize - 1u) / localSize;
            m_functions.cmdPushConstants(
                m_commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                sizeof(triangle), &triangle);
            m_functions.cmdDispatch(
                m_commandBuffer, groupCountX, groupCountY, 1u);
        }

        VkBufferMemoryBarrier completeBarrier{};
        completeBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        completeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        completeBarrier.dstAccessMask =
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        completeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.buffer = m_vram.buffer;
        completeBarrier.offset = 0u;
        completeBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0u, 0u, nullptr, 1u, &completeBarrier, 0u, nullptr);

        if (!submitCommands(
                "resident CT32 triangle batch", triangles.size(), 2u, 1u,
                report, statistics, error) ||
            !finishOperation(
                "resident CT32 triangle batch", validationErrorsBefore,
                report, statistics, error))
        {
            return false;
        }
        error.clear();
        return true;
    }

    bool VulkanExecutionContext::
        executeResidentGouraudDepthCt32Triangles(
            std::span<const GsVulkanGouraudDepthCt32Triangle> triangles,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (!m_exactGouraudDepthCt32Triangle ||
            m_gouraudDepthCt32TrianglePipeline == VK_NULL_HANDLE)
        {
            error =
                "Vulkan device does not support exact Gouraud depth CT32 triangles";
            return false;
        }
        if (!validateResidentGouraudDepthCt32TriangleBatch(
                triangles, error))
        {
            return false;
        }

        GsVulkanTriangleTileWorkPlan tileWork;
        const GsVulkanDeviceReport *selectedDevice =
            report.selectedDevice();
        const bool tiled =
            buildTriangleTileWork(triangles, tileWork) &&
            selectedDevice &&
            tileWork.tileCount <=
                selectedDevice->maxComputeWorkGroupCountX;

        const VkDeviceSize recordBytes =
            sizeof(GsVulkanGouraudDepthCt32Triangle) *
            triangles.size();
        std::memcpy(
            m_memoryCasesMap, triangles.data(),
            static_cast<size_t>(recordBytes));
        if (!flushMappedAllocation(
                m_memoryCases, "Gouraud triangle records", report, error))
        {
            return false;
        }
        VkDeviceSize tileBytes = 0u;
        if (tiled)
        {
            tileBytes = sizeof(uint32_t) * tileWork.words.size();
            std::memcpy(
                m_memoryResultsMap, tileWork.words.data(),
                static_cast<size_t>(tileBytes));
            if (!flushMappedAllocation(
                    m_memoryResults, "Gouraud triangle tile state",
                    report, error))
            {
                return false;
            }
        }

        constexpr uint32_t localSize = 8u;
        const uint32_t validationErrorsBefore =
            m_validation.errors.load(std::memory_order_relaxed);
        if (!beginCommands(report, error))
            return false;

        std::array<VkBufferMemoryBarrier, 2> hostUploadBarriers{};
        VkBufferMemoryBarrier &recordUploadBarrier =
            hostUploadBarriers[0];
        recordUploadBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        recordUploadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        recordUploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        recordUploadBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        recordUploadBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        recordUploadBarrier.buffer = m_memoryCases.buffer;
        recordUploadBarrier.offset = 0u;
        recordUploadBarrier.size = recordBytes;
        uint32_t hostUploadBarrierCount = 1u;
        if (tiled)
        {
            VkBufferMemoryBarrier &tileUploadBarrier =
                hostUploadBarriers[1];
            tileUploadBarrier.sType =
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            tileUploadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            tileUploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            tileUploadBarrier.srcQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            tileUploadBarrier.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            tileUploadBarrier.buffer = m_memoryResults.buffer;
            tileUploadBarrier.offset = 0u;
            tileUploadBarrier.size = tileBytes;
            hostUploadBarrierCount = 2u;
        }
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, hostUploadBarrierCount,
            hostUploadBarriers.data(), 0u, nullptr);

        VkBufferMemoryBarrier prepareBarrier{};
        prepareBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        prepareBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        prepareBarrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        prepareBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.buffer = m_vram.buffer;
        prepareBarrier.offset = 0u;
        prepareBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, 1u, &prepareBarrier, 0u, nullptr);

        m_functions.cmdBindPipeline(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_gouraudDepthCt32TrianglePipeline);
        m_functions.cmdBindDescriptorSets(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayout, 0u, 1u, &m_descriptorSet,
            0u, nullptr);
        size_t dispatchCount = 0u;
        size_t pipelineBarrierCount = 3u;
        if (tiled)
        {
            const GsVulkanGouraudTriangleParameters parameters{
                0u, tileWork.indexOffsetWords};
            m_functions.cmdPushConstants(
                m_commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                sizeof(parameters), &parameters);
            m_functions.cmdDispatch(
                m_commandBuffer, tileWork.tileCount, 1u, 1u);
            dispatchCount = 1u;
        }
        else for (size_t index = 0u; index < triangles.size(); ++index)
        {
            if (index != 0u)
            {
                VkBufferMemoryBarrier dependencyBarrier{};
                dependencyBarrier.sType =
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                dependencyBarrier.srcAccessMask =
                    VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT;
                dependencyBarrier.dstAccessMask =
                    VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT;
                dependencyBarrier.srcQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                dependencyBarrier.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                dependencyBarrier.buffer = m_vram.buffer;
                dependencyBarrier.offset = 0u;
                dependencyBarrier.size = GS_VULKAN_VRAM_SIZE;
                m_functions.cmdPipelineBarrier(
                    m_commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0u, 0u, nullptr, 1u, &dependencyBarrier,
                    0u, nullptr);
            }

            const GsVulkanGouraudDepthCt32Triangle &triangle =
                triangles[index];
            const uint32_t groupCountX =
                (triangle.boundsX1 - triangle.boundsX0 +
                 localSize - 1u) / localSize;
            const uint32_t groupCountY =
                (triangle.boundsY1 - triangle.boundsY0 +
                 localSize - 1u) / localSize;
            const GsVulkanGouraudTriangleParameters parameters{
                static_cast<uint32_t>(index), 0u};
            m_functions.cmdPushConstants(
                m_commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                sizeof(parameters), &parameters);
            m_functions.cmdDispatch(
                m_commandBuffer, groupCountX, groupCountY, 1u);
            ++dispatchCount;
        }
        if (!tiled)
            pipelineBarrierCount = triangles.size() + 2u;

        VkBufferMemoryBarrier completeBarrier{};
        completeBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        completeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        completeBarrier.dstAccessMask =
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        completeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.buffer = m_vram.buffer;
        completeBarrier.offset = 0u;
        completeBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0u, 0u, nullptr, 1u, &completeBarrier, 0u, nullptr);

        if (!submitCommands(
                "resident Gouraud depth CT32 triangle batch",
                dispatchCount, pipelineBarrierCount, 1u,
                report, statistics, error) ||
            !finishOperation(
                "resident Gouraud depth CT32 triangle batch",
                validationErrorsBefore,
                report, statistics, error))
        {
            return false;
        }
        error.clear();
        return true;
    }

    bool VulkanExecutionContext::
        executeResidentFeedbackNearestDepthCt32Triangles(
            std::span<const uint8_t> feedbackSnapshot,
            GsVulkanFeedbackSnapshotMode snapshotMode,
            std::span<
                const GsVulkanFeedbackNearestDepthCt32Triangle> triangles,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (!m_exactFeedbackNearestDepthCt32Triangle ||
            m_feedbackNearestDepthCt32TrianglePipeline == VK_NULL_HANDLE)
        {
            error =
                "Vulkan device does not support exact feedback nearest depth CT32 triangles";
            return false;
        }
        const bool uploadsHostSnapshot =
            snapshotMode == GsVulkanFeedbackSnapshotMode::UploadHost;
        if ((uploadsHostSnapshot &&
             feedbackSnapshot.size() != GS_VULKAN_VRAM_SIZE) ||
            (!uploadsHostSnapshot && !feedbackSnapshot.empty()))
        {
            error =
                "Vulkan feedback nearest depth CT32 triangles received an invalid snapshot source";
            return false;
        }
        if (snapshotMode == GsVulkanFeedbackSnapshotMode::ReuseResident &&
            !m_feedbackSnapshotResidentValid)
        {
            error =
                "Vulkan feedback nearest depth CT32 triangles cannot reuse an uninitialized resident snapshot";
            return false;
        }
        if (!validateResidentFeedbackNearestDepthCt32TriangleBatch(
                triangles, error))
        {
            return false;
        }

        GsVulkanTriangleTileWorkPlan tileWork;
        const GsVulkanDeviceReport *selectedDevice =
            report.selectedDevice();
        const bool tiled =
            buildTriangleTileWork(triangles, tileWork) &&
            selectedDevice &&
            tileWork.tileCount <=
                selectedDevice->maxComputeWorkGroupCountX;

        const VkDeviceSize recordBytes =
            sizeof(GsVulkanFeedbackNearestDepthCt32Triangle) *
            triangles.size();
        std::memcpy(
            m_memoryCasesMap, triangles.data(),
            static_cast<size_t>(recordBytes));
        if (!flushMappedAllocation(
                m_memoryCases,
                "feedback nearest triangle records",
                report, error))
        {
            return false;
        }
        const VkDeviceSize tileBytes = tiled
            ? sizeof(uint32_t) * tileWork.words.size()
            : 0u;
        if (tiled)
        {
            std::memcpy(
                m_memoryResultsMap, tileWork.words.data(),
                static_cast<size_t>(tileBytes));
            if (!flushMappedAllocation(
                    m_memoryResults,
                    "feedback nearest triangle tile state",
                    report, error))
            {
                return false;
            }
        }
        if (uploadsHostSnapshot)
        {
            std::memcpy(
                m_feedbackStagingMap,
                feedbackSnapshot.data(),
                feedbackSnapshot.size());
            if (!flushMappedAllocation(
                    m_feedbackStaging,
                    "feedback nearest triangle snapshot",
                    report, error))
            {
                return false;
            }
        }

        const uint32_t validationErrorsBefore =
            m_validation.errors.load(std::memory_order_relaxed);
        if (!beginCommands(report, error))
            return false;

        std::array<VkBufferMemoryBarrier, 2> hostRecordBarriers{};
        VkBufferMemoryBarrier &recordBarrier = hostRecordBarriers[0];
        recordBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        recordBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        recordBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        recordBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        recordBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        recordBarrier.buffer = m_memoryCases.buffer;
        recordBarrier.offset = 0u;
        recordBarrier.size = recordBytes;
        uint32_t hostRecordBarrierCount = 1u;
        if (tiled)
        {
            VkBufferMemoryBarrier &tileBarrier = hostRecordBarriers[1];
            tileBarrier.sType =
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            tileBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            tileBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            tileBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            tileBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            tileBarrier.buffer = m_memoryResults.buffer;
            tileBarrier.offset = 0u;
            tileBarrier.size = tileBytes;
            hostRecordBarrierCount = 2u;
        }
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr,
            hostRecordBarrierCount, hostRecordBarriers.data(),
            0u, nullptr);

        uint64_t pipelineBarrierCount = 1u;
        if (uploadsHostSnapshot)
        {
            VkBufferMemoryBarrier stagingBarrier{};
            stagingBarrier.sType =
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            stagingBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            stagingBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            stagingBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            stagingBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            stagingBarrier.buffer = m_feedbackStaging.buffer;
            stagingBarrier.offset = 0u;
            stagingBarrier.size = GS_VULKAN_VRAM_SIZE;
            m_functions.cmdPipelineBarrier(
                m_commandBuffer,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0u, 0u, nullptr, 1u, &stagingBarrier,
                0u, nullptr);

            VkBufferCopy copyRegion{};
            copyRegion.size = GS_VULKAN_VRAM_SIZE;
            m_functions.cmdCopyBuffer(
                m_commandBuffer,
                m_feedbackStaging.buffer,
                m_feedbackSnapshot.buffer,
                1u, &copyRegion);
            ++pipelineBarrierCount;
        }
        else if (snapshotMode ==
                 GsVulkanFeedbackSnapshotMode::CaptureResident)
        {
            VkBufferMemoryBarrier captureSourceBarrier{};
            captureSourceBarrier.sType =
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            captureSourceBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            captureSourceBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            captureSourceBarrier.srcQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            captureSourceBarrier.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            captureSourceBarrier.buffer = m_vram.buffer;
            captureSourceBarrier.offset = 0u;
            captureSourceBarrier.size = GS_VULKAN_VRAM_SIZE;
            m_functions.cmdPipelineBarrier(
                m_commandBuffer,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0u, 0u, nullptr, 1u, &captureSourceBarrier,
                0u, nullptr);

            VkBufferCopy copyRegion{};
            copyRegion.size = GS_VULKAN_VRAM_SIZE;
            m_functions.cmdCopyBuffer(
                m_commandBuffer,
                m_vram.buffer,
                m_feedbackSnapshot.buffer,
                1u, &copyRegion);
            ++pipelineBarrierCount;
        }

        VkBufferMemoryBarrier snapshotReadBarrier{};
        snapshotReadBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        snapshotReadBarrier.srcAccessMask =
            snapshotMode == GsVulkanFeedbackSnapshotMode::ReuseResident
                ? VK_ACCESS_MEMORY_WRITE_BIT
                : VK_ACCESS_TRANSFER_WRITE_BIT;
        snapshotReadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        snapshotReadBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        snapshotReadBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        snapshotReadBarrier.buffer = m_feedbackSnapshot.buffer;
        snapshotReadBarrier.offset = 0u;
        snapshotReadBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            snapshotMode == GsVulkanFeedbackSnapshotMode::ReuseResident
                ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT
                : VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, 1u, &snapshotReadBarrier,
            0u, nullptr);
        ++pipelineBarrierCount;

        VkBufferMemoryBarrier prepareBarrier{};
        prepareBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        prepareBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        prepareBarrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        prepareBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.buffer = m_vram.buffer;
        prepareBarrier.offset = 0u;
        prepareBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, 1u, &prepareBarrier,
            0u, nullptr);
        ++pipelineBarrierCount;

        m_functions.cmdBindPipeline(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_feedbackNearestDepthCt32TrianglePipeline);
        m_functions.cmdBindDescriptorSets(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayout, 0u, 1u, &m_descriptorSet,
            0u, nullptr);

        constexpr uint32_t localSize = 8u;
        size_t dispatchCount = 0u;
        if (tiled)
        {
            const GsVulkanT8TriangleParameters parameters{
                0u, tileWork.indexOffsetWords, 0u};
            m_functions.cmdPushConstants(
                m_commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                sizeof(parameters), &parameters);
            m_functions.cmdDispatch(
                m_commandBuffer, tileWork.tileCount, 1u, 1u);
            dispatchCount = 1u;
        }
        else for (size_t index = 0u; index < triangles.size(); ++index)
        {
            if (index != 0u)
            {
                VkBufferMemoryBarrier dependencyBarrier{};
                dependencyBarrier.sType =
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                dependencyBarrier.srcAccessMask =
                    VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT;
                dependencyBarrier.dstAccessMask =
                    VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT;
                dependencyBarrier.srcQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                dependencyBarrier.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                dependencyBarrier.buffer = m_vram.buffer;
                dependencyBarrier.offset = 0u;
                dependencyBarrier.size = GS_VULKAN_VRAM_SIZE;
                m_functions.cmdPipelineBarrier(
                    m_commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0u, 0u, nullptr, 1u, &dependencyBarrier,
                    0u, nullptr);
                ++pipelineBarrierCount;
            }
            const auto &triangle = triangles[index];
            const uint32_t groupCountX =
                (triangle.boundsX1 - triangle.boundsX0 +
                 localSize - 1u) / localSize;
            const uint32_t groupCountY =
                (triangle.boundsY1 - triangle.boundsY0 +
                 localSize - 1u) / localSize;
            const GsVulkanT8TriangleParameters parameters{
                static_cast<uint32_t>(index), 0u, 0u};
            m_functions.cmdPushConstants(
                m_commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                sizeof(parameters), &parameters);
            m_functions.cmdDispatch(
                m_commandBuffer, groupCountX, groupCountY, 1u);
            ++dispatchCount;
        }

        VkBufferMemoryBarrier completeBarrier{};
        completeBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        completeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        completeBarrier.dstAccessMask =
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        completeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.buffer = m_vram.buffer;
        completeBarrier.offset = 0u;
        completeBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0u, 0u, nullptr, 1u, &completeBarrier,
            0u, nullptr);
        ++pipelineBarrierCount;

        if (!submitCommands(
                "resident feedback nearest depth CT32 triangle batch",
                dispatchCount, pipelineBarrierCount, 1u,
                report, statistics, error) ||
            !finishOperation(
                "resident feedback nearest depth CT32 triangle batch",
                validationErrorsBefore,
                report, statistics, error))
        {
            return false;
        }
        m_feedbackSnapshotResidentValid = true;
        if (snapshotMode ==
            GsVulkanFeedbackSnapshotMode::CaptureResident)
        {
            ++statistics.residentFeedbackSnapshotsCaptured;
        }
        else if (snapshotMode ==
                 GsVulkanFeedbackSnapshotMode::ReuseResident)
        {
            ++statistics.residentFeedbackSnapshotsReused;
        }
        error.clear();
        return true;
    }

    bool VulkanExecutionContext::
        executeResidentT8GouraudDepthCt32Triangles(
            std::span<
                const GsVulkanResidentT8GouraudDepthCt32Triangle> triangles,
            std::span<const GsVulkanT8Palette> palettes,
            std::span<const uint8_t> pageUploadSource,
            const GsVramPageMask &pageUploadPages,
            GsVulkanCapabilityReport &report,
            GsVulkanServiceStatistics &statistics,
            std::string &error)
    {
        if (!m_healthy)
        {
            error = "Vulkan GS service is not healthy";
            return false;
        }
        if (!m_exactT8GouraudDepthCt32Triangle ||
            m_t8GouraudDepthCt32TrianglePipeline == VK_NULL_HANDLE)
        {
            error =
                "Vulkan device does not support exact T8 Gouraud depth CT32 triangles";
            return false;
        }
        const PageCopyPlan pageUploadPlan =
            buildPageCopyPlan(pageUploadPages, true);
        const bool uploadsPages = pageUploadPlan.pageCount != 0u;
        if (uploadsPages != !pageUploadSource.empty() ||
            (uploadsPages &&
             pageUploadSource.size() != GS_VULKAN_VRAM_SIZE))
        {
            error =
                "Vulkan resident T8 page fusion received an invalid upload";
            return false;
        }
        GsVulkanTriangleTileWorkPlan tileWork;
        const GsVulkanDeviceReport *selectedDevice =
            report.selectedDevice();
        bool tiled =
            buildTriangleTileWork(triangles, tileWork) &&
            selectedDevice &&
            tileWork.tileCount <=
                selectedDevice->maxComputeWorkGroupCountX;

        constexpr size_t auxiliaryCapacityWords =
            GS_VULKAN_AUXILIARY_STORAGE_SIZE / sizeof(uint32_t);
        const size_t paletteWords = palettes.size() * 256u;
        if (paletteWords > auxiliaryCapacityWords)
        {
            error = "Vulkan resident T8 palette table exceeds auxiliary storage";
            return false;
        }
        if (tiled &&
            tileWork.words.size() + paletteWords >
                auxiliaryCapacityWords)
        {
            tiled = false;
        }
        std::vector<uint32_t> auxiliaryWords;
        if (tiled)
            auxiliaryWords = std::move(tileWork.words);
        const uint32_t paletteIndexWords =
            static_cast<uint32_t>(auxiliaryWords.size());
        auxiliaryWords.reserve(auxiliaryWords.size() + paletteWords);
        for (const GsVulkanT8Palette &palette : palettes)
        {
            auxiliaryWords.insert(
                auxiliaryWords.end(),
                palette.begin(), palette.end());
        }

        if (uploadsPages)
        {
            for (uint32_t index = 0u;
                 index < pageUploadPlan.regionCount; ++index)
            {
                const VkBufferCopy &region =
                    pageUploadPlan.regions[index];
                std::memcpy(
                    static_cast<uint8_t *>(m_stagingMap) +
                        region.srcOffset,
                    pageUploadSource.data() + region.dstOffset,
                    static_cast<size_t>(region.size));
            }
            if (!flushMappedAllocation(
                    m_staging, "fused T8 page upload", report, error))
            {
                return false;
            }
        }

        const VkDeviceSize recordBytes =
            sizeof(GsVulkanResidentT8GouraudDepthCt32Triangle) *
            triangles.size();
        std::memcpy(
            m_memoryCasesMap, triangles.data(),
            static_cast<size_t>(recordBytes));
        if (!flushMappedAllocation(
                m_memoryCases, "T8 triangle records", report, error))
        {
            return false;
        }
        const VkDeviceSize auxiliaryBytes =
            sizeof(uint32_t) * auxiliaryWords.size();
        std::memcpy(
            m_memoryResultsMap, auxiliaryWords.data(),
            static_cast<size_t>(auxiliaryBytes));
        if (!flushMappedAllocation(
                m_memoryResults, "T8 triangle tile and palette state",
                report, error))
        {
            return false;
        }

        constexpr uint32_t localSize = 8u;
        const uint32_t validationErrorsBefore =
            m_validation.errors.load(std::memory_order_relaxed);
        if (!beginCommands(report, error))
            return false;

        size_t fusedUploadBarrierCount = 0u;
        if (uploadsPages)
        {
            VkBufferMemoryBarrier stagingBarrier{};
            stagingBarrier.sType =
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            stagingBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            stagingBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            stagingBarrier.srcQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            stagingBarrier.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            stagingBarrier.buffer = m_staging.buffer;
            stagingBarrier.offset = 0u;
            stagingBarrier.size = pageUploadPlan.byteCount;
            m_functions.cmdPipelineBarrier(
                m_commandBuffer,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0u, 0u, nullptr, 1u, &stagingBarrier, 0u, nullptr);

            VkBufferMemoryBarrier vramUploadPrepareBarrier{};
            vramUploadPrepareBarrier.sType =
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            vramUploadPrepareBarrier.srcAccessMask =
                VK_ACCESS_MEMORY_READ_BIT |
                VK_ACCESS_MEMORY_WRITE_BIT;
            vramUploadPrepareBarrier.dstAccessMask =
                VK_ACCESS_TRANSFER_WRITE_BIT;
            vramUploadPrepareBarrier.srcQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            vramUploadPrepareBarrier.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            vramUploadPrepareBarrier.buffer = m_vram.buffer;
            vramUploadPrepareBarrier.offset = 0u;
            vramUploadPrepareBarrier.size = GS_VULKAN_VRAM_SIZE;
            m_functions.cmdPipelineBarrier(
                m_commandBuffer,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0u, 0u, nullptr, 1u, &vramUploadPrepareBarrier,
                0u, nullptr);

            m_functions.cmdCopyBuffer(
                m_commandBuffer, m_staging.buffer, m_vram.buffer,
                pageUploadPlan.regionCount,
                pageUploadPlan.regions.data());

            VkBufferMemoryBarrier vramUploadCompleteBarrier{};
            vramUploadCompleteBarrier.sType =
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            vramUploadCompleteBarrier.srcAccessMask =
                VK_ACCESS_TRANSFER_WRITE_BIT;
            vramUploadCompleteBarrier.dstAccessMask =
                VK_ACCESS_MEMORY_READ_BIT |
                VK_ACCESS_MEMORY_WRITE_BIT;
            vramUploadCompleteBarrier.srcQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            vramUploadCompleteBarrier.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            vramUploadCompleteBarrier.buffer = m_vram.buffer;
            vramUploadCompleteBarrier.offset = 0u;
            vramUploadCompleteBarrier.size = GS_VULKAN_VRAM_SIZE;
            m_functions.cmdPipelineBarrier(
                m_commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0u, 0u, nullptr, 1u, &vramUploadCompleteBarrier,
                0u, nullptr);
            fusedUploadBarrierCount = 3u;
        }

        std::array<VkBufferMemoryBarrier, 2> hostUploadBarriers{};
        VkBufferMemoryBarrier &recordUploadBarrier =
            hostUploadBarriers[0];
        recordUploadBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        recordUploadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        recordUploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        recordUploadBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        recordUploadBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        recordUploadBarrier.buffer = m_memoryCases.buffer;
        recordUploadBarrier.offset = 0u;
        recordUploadBarrier.size = recordBytes;
        uint32_t hostUploadBarrierCount = 1u;
        VkBufferMemoryBarrier &tileUploadBarrier =
            hostUploadBarriers[1];
        tileUploadBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        tileUploadBarrier.srcAccessMask =
            VK_ACCESS_HOST_WRITE_BIT;
        tileUploadBarrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT;
        tileUploadBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        tileUploadBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        tileUploadBarrier.buffer = m_memoryResults.buffer;
        tileUploadBarrier.offset = 0u;
        tileUploadBarrier.size = auxiliaryBytes;
        hostUploadBarrierCount = 2u;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, hostUploadBarrierCount,
            hostUploadBarriers.data(),
            0u, nullptr);

        VkBufferMemoryBarrier prepareBarrier{};
        prepareBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        prepareBarrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        prepareBarrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        prepareBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        prepareBarrier.buffer = m_vram.buffer;
        prepareBarrier.offset = 0u;
        prepareBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, 1u, &prepareBarrier, 0u, nullptr);

        m_functions.cmdBindPipeline(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_t8GouraudDepthCt32TrianglePipeline);
        m_functions.cmdBindDescriptorSets(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayout, 0u, 1u, &m_descriptorSet,
            0u, nullptr);
        size_t dispatchCount = 0u;
        size_t pipelineBarrierCount = 3u + fusedUploadBarrierCount;
        if (tiled)
        {
            const GsVulkanT8TriangleParameters parameters{
                0u, tileWork.indexOffsetWords, paletteIndexWords};
            m_functions.cmdPushConstants(
                m_commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                sizeof(parameters), &parameters);
            m_functions.cmdDispatch(
                m_commandBuffer, tileWork.tileCount, 1u, 1u);
            dispatchCount = 1u;
        }
        else for (size_t index = 0u; index < triangles.size(); ++index)
        {
            if (index != 0u)
            {
                VkBufferMemoryBarrier dependencyBarrier{};
                dependencyBarrier.sType =
                    VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                dependencyBarrier.srcAccessMask =
                    VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT;
                dependencyBarrier.dstAccessMask =
                    VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT;
                dependencyBarrier.srcQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                dependencyBarrier.dstQueueFamilyIndex =
                    VK_QUEUE_FAMILY_IGNORED;
                dependencyBarrier.buffer = m_vram.buffer;
                dependencyBarrier.offset = 0u;
                dependencyBarrier.size = GS_VULKAN_VRAM_SIZE;
                m_functions.cmdPipelineBarrier(
                    m_commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0u, 0u, nullptr, 1u, &dependencyBarrier,
                    0u, nullptr);
            }

            const GsVulkanResidentT8GouraudDepthCt32Triangle &triangle =
                triangles[index];
            const uint32_t groupCountX =
                (triangle.boundsX1 - triangle.boundsX0 +
                 localSize - 1u) / localSize;
            const uint32_t groupCountY =
                (triangle.boundsY1 - triangle.boundsY0 +
                 localSize - 1u) / localSize;
            const GsVulkanT8TriangleParameters parameters{
                static_cast<uint32_t>(index), 0u,
                paletteIndexWords};
            m_functions.cmdPushConstants(
                m_commandBuffer, m_pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0u,
                sizeof(parameters), &parameters);
            m_functions.cmdDispatch(
                m_commandBuffer, groupCountX, groupCountY, 1u);
            ++dispatchCount;
        }
        if (!tiled)
        {
            pipelineBarrierCount =
                triangles.size() + 2u + fusedUploadBarrierCount;
        }

        VkBufferMemoryBarrier completeBarrier{};
        completeBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        completeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        completeBarrier.dstAccessMask =
            VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        completeBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        completeBarrier.buffer = m_vram.buffer;
        completeBarrier.offset = 0u;
        completeBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0u, 0u, nullptr, 1u, &completeBarrier, 0u, nullptr);

        if (!submitCommands(
                "resident T8 Gouraud depth CT32 triangle batch",
                dispatchCount, pipelineBarrierCount, 1u,
                report, statistics, error) ||
            !finishOperation(
                "resident T8 Gouraud depth CT32 triangle batch",
                validationErrorsBefore,
                report, statistics, error))
        {
            return false;
        }
        if (uploadsPages)
        {
            statistics.bytesUploaded += pageUploadPlan.byteCount;
            statistics.pagesUploaded += pageUploadPlan.pageCount;
            statistics.pageUploadRegions += pageUploadPlan.regionCount;
        }
        error.clear();
        return true;
    }

    bool VulkanExecutionContext::executeKernel(
        std::span<const uint8_t> input,
        std::span<const GsVulkanMemoryCase> cases,
        std::span<const uint8_t> feedbackSnapshot,
        VkPipeline pipeline,
        uint32_t groupCountX,
        uint32_t groupCountY,
        const void *pushConstants,
        uint32_t pushConstantSize,
        std::vector<uint8_t> &output,
        std::vector<GsVulkanMemoryResult> *results,
        std::string_view operationName,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        return executeKernelWithAuxiliary(
            input, cases, {}, {}, feedbackSnapshot,
            pipeline, groupCountX, groupCountY,
            pushConstants, pushConstantSize,
            output, results, operationName,
            report, statistics, error);
    }

    bool VulkanExecutionContext::executeKernelWithAuxiliary(
        std::span<const uint8_t> input,
        std::span<const GsVulkanMemoryCase> cases,
        std::span<const uint8_t> auxiliaryInput,
        std::span<const uint8_t> auxiliaryResultInput,
        std::span<const uint8_t> feedbackSnapshot,
        VkPipeline pipeline,
        uint32_t groupCountX,
        uint32_t groupCountY,
        const void *pushConstants,
        uint32_t pushConstantSize,
        std::vector<uint8_t> &output,
        std::vector<GsVulkanMemoryResult> *results,
        std::string_view operationName,
        GsVulkanCapabilityReport &report,
        GsVulkanServiceStatistics &statistics,
        std::string &error)
    {
        const bool memoryOperation = results != nullptr;
        const bool auxiliaryOperation = !auxiliaryInput.empty();
        const bool auxiliaryResultOperation =
            !auxiliaryResultInput.empty();
        const bool feedbackOperation = !feedbackSnapshot.empty();
        constexpr size_t auxiliaryCapacity =
            sizeof(GsVulkanMemoryCase) *
            static_cast<size_t>(GS_VULKAN_MAX_MEMORY_CASES);
        if (pipeline == VK_NULL_HANDLE ||
            groupCountX == 0u || groupCountY == 0u ||
            !pushConstants || pushConstantSize == 0u ||
            pushConstantSize >
                sizeof(GsVulkanFeedbackLinearDepthCt32Sprite) ||
            (pushConstantSize & 3u) != 0u ||
            memoryOperation != !cases.empty() ||
            (memoryOperation && auxiliaryOperation) ||
            auxiliaryInput.size() > auxiliaryCapacity ||
            auxiliaryResultInput.size() >
                GS_VULKAN_AUXILIARY_STORAGE_SIZE ||
            (feedbackOperation &&
             feedbackSnapshot.size() != GS_VULKAN_VRAM_SIZE))
        {
            error = "invalid Vulkan GS kernel request";
            return false;
        }

        const uint32_t validationErrorsBefore =
            m_validation.errors.load(std::memory_order_relaxed);
        std::memcpy(m_stagingMap, input.data(), input.size());
        if (feedbackOperation)
        {
            std::memcpy(
                m_feedbackStagingMap,
                feedbackSnapshot.data(),
                feedbackSnapshot.size());
        }
        const VkDeviceSize caseBytes =
            sizeof(GsVulkanMemoryCase) * cases.size();
        const VkDeviceSize resultBytes =
            sizeof(GsVulkanMemoryResult) * cases.size();
        if (memoryOperation)
        {
            std::memcpy(
                m_memoryCasesMap, cases.data(),
                static_cast<size_t>(caseBytes));
        }
        else if (auxiliaryOperation)
        {
            std::memcpy(
                m_memoryCasesMap,
                auxiliaryInput.data(), auxiliaryInput.size());
        }
        if (auxiliaryResultOperation)
        {
            std::memcpy(
                m_memoryResultsMap,
                auxiliaryResultInput.data(),
                auxiliaryResultInput.size());
        }

        if (!flushMappedAllocation(
                m_staging, "VRAM", report, error) ||
            (feedbackOperation &&
             !flushMappedAllocation(
                 m_feedbackStaging,
                 "feedback snapshot", report, error)) ||
            ((memoryOperation || auxiliaryOperation) &&
             !flushMappedAllocation(
                 m_memoryCases,
                 memoryOperation
                     ? "memory cases"
                     : "auxiliary records",
                 report, error)) ||
            (auxiliaryResultOperation &&
             !flushMappedAllocation(
                 m_memoryResults, "auxiliary state", report, error)) ||
            !beginCommands(report, error))
        {
            return false;
        }

        VkBufferMemoryBarrier stagingUploadBarrier{};
        stagingUploadBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        stagingUploadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        stagingUploadBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        stagingUploadBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        stagingUploadBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        stagingUploadBarrier.buffer = m_staging.buffer;
        stagingUploadBarrier.offset = 0u;
        stagingUploadBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0u, 0u, nullptr, 1u, &stagingUploadBarrier,
            0u, nullptr);

        if (feedbackOperation)
        {
            VkBufferMemoryBarrier feedbackStagingUploadBarrier{};
            feedbackStagingUploadBarrier.sType =
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            feedbackStagingUploadBarrier.srcAccessMask =
                VK_ACCESS_HOST_WRITE_BIT;
            feedbackStagingUploadBarrier.dstAccessMask =
                VK_ACCESS_TRANSFER_READ_BIT;
            feedbackStagingUploadBarrier.srcQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            feedbackStagingUploadBarrier.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            feedbackStagingUploadBarrier.buffer =
                m_feedbackStaging.buffer;
            feedbackStagingUploadBarrier.offset = 0u;
            feedbackStagingUploadBarrier.size =
                GS_VULKAN_VRAM_SIZE;
            m_functions.cmdPipelineBarrier(
                m_commandBuffer,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0u, 0u, nullptr, 1u,
                &feedbackStagingUploadBarrier,
                0u, nullptr);
        }

        if (memoryOperation || auxiliaryOperation)
        {
            VkBufferMemoryBarrier caseUploadBarrier{};
            caseUploadBarrier.sType =
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            caseUploadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            caseUploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            caseUploadBarrier.srcQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            caseUploadBarrier.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            caseUploadBarrier.buffer = m_memoryCases.buffer;
            caseUploadBarrier.offset = 0u;
            caseUploadBarrier.size = memoryOperation
                ? caseBytes
                : static_cast<VkDeviceSize>(auxiliaryInput.size());
            m_functions.cmdPipelineBarrier(
                m_commandBuffer,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0u, 0u, nullptr, 1u, &caseUploadBarrier,
                0u, nullptr);
        }

        if (auxiliaryResultOperation)
        {
            VkBufferMemoryBarrier auxiliaryUploadBarrier{};
            auxiliaryUploadBarrier.sType =
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            auxiliaryUploadBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            auxiliaryUploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            auxiliaryUploadBarrier.srcQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            auxiliaryUploadBarrier.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            auxiliaryUploadBarrier.buffer = m_memoryResults.buffer;
            auxiliaryUploadBarrier.offset = 0u;
            auxiliaryUploadBarrier.size = auxiliaryResultInput.size();
            m_functions.cmdPipelineBarrier(
                m_commandBuffer,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0u, 0u, nullptr, 1u, &auxiliaryUploadBarrier,
                0u, nullptr);
        }

        VkBufferCopy copyRegion{};
        copyRegion.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdCopyBuffer(
            m_commandBuffer, m_staging.buffer, m_vram.buffer,
            1u, &copyRegion);
        if (feedbackOperation)
        {
            m_functions.cmdCopyBuffer(
                m_commandBuffer,
                m_feedbackStaging.buffer,
                m_feedbackSnapshot.buffer,
                1u, &copyRegion);
        }

        VkBufferMemoryBarrier uploadComputeBarrier{};
        uploadComputeBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        uploadComputeBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        uploadComputeBarrier.dstAccessMask =
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        uploadComputeBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        uploadComputeBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        uploadComputeBarrier.buffer = m_vram.buffer;
        uploadComputeBarrier.offset = 0u;
        uploadComputeBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0u, 0u, nullptr, 1u, &uploadComputeBarrier,
            0u, nullptr);

        if (feedbackOperation)
        {
            VkBufferMemoryBarrier feedbackUploadComputeBarrier{};
            feedbackUploadComputeBarrier.sType =
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            feedbackUploadComputeBarrier.srcAccessMask =
                VK_ACCESS_TRANSFER_WRITE_BIT;
            feedbackUploadComputeBarrier.dstAccessMask =
                VK_ACCESS_SHADER_READ_BIT;
            feedbackUploadComputeBarrier.srcQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            feedbackUploadComputeBarrier.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            feedbackUploadComputeBarrier.buffer =
                m_feedbackSnapshot.buffer;
            feedbackUploadComputeBarrier.offset = 0u;
            feedbackUploadComputeBarrier.size =
                GS_VULKAN_VRAM_SIZE;
            m_functions.cmdPipelineBarrier(
                m_commandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0u, 0u, nullptr, 1u,
                &feedbackUploadComputeBarrier,
                0u, nullptr);
        }

        m_functions.cmdBindPipeline(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline);
        m_functions.cmdBindDescriptorSets(
            m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelineLayout, 0u, 1u, &m_descriptorSet,
            0u, nullptr);
        m_functions.cmdPushConstants(
            m_commandBuffer, m_pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0u,
            pushConstantSize, pushConstants);
        m_functions.cmdDispatch(
            m_commandBuffer, groupCountX, groupCountY, 1u);

        VkBufferMemoryBarrier computeDownloadBarrier{};
        computeDownloadBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        computeDownloadBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        computeDownloadBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        computeDownloadBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        computeDownloadBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        computeDownloadBarrier.buffer = m_vram.buffer;
        computeDownloadBarrier.offset = 0u;
        computeDownloadBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0u, 0u, nullptr, 1u, &computeDownloadBarrier,
            0u, nullptr);

        if (memoryOperation)
        {
            VkBufferMemoryBarrier resultDownloadBarrier{};
            resultDownloadBarrier.sType =
                VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            resultDownloadBarrier.srcAccessMask =
                VK_ACCESS_SHADER_WRITE_BIT;
            resultDownloadBarrier.dstAccessMask =
                VK_ACCESS_HOST_READ_BIT;
            resultDownloadBarrier.srcQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            resultDownloadBarrier.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            resultDownloadBarrier.buffer = m_memoryResults.buffer;
            resultDownloadBarrier.offset = 0u;
            resultDownloadBarrier.size = resultBytes;
            m_functions.cmdPipelineBarrier(
                m_commandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0u, 0u, nullptr, 1u, &resultDownloadBarrier,
                0u, nullptr);
        }

        m_functions.cmdCopyBuffer(
            m_commandBuffer, m_vram.buffer, m_staging.buffer,
            1u, &copyRegion);

        VkBufferMemoryBarrier stagingDownloadBarrier{};
        stagingDownloadBarrier.sType =
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        stagingDownloadBarrier.srcAccessMask =
            VK_ACCESS_TRANSFER_WRITE_BIT;
        stagingDownloadBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        stagingDownloadBarrier.srcQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        stagingDownloadBarrier.dstQueueFamilyIndex =
            VK_QUEUE_FAMILY_IGNORED;
        stagingDownloadBarrier.buffer = m_staging.buffer;
        stagingDownloadBarrier.offset = 0u;
        stagingDownloadBarrier.size = GS_VULKAN_VRAM_SIZE;
        m_functions.cmdPipelineBarrier(
            m_commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0u, 0u, nullptr, 1u, &stagingDownloadBarrier,
            0u, nullptr);

        if (!submitCommands(
                operationName, 1u,
                4u +
                    ((memoryOperation || auxiliaryOperation) ? 1u : 0u) +
                    (auxiliaryResultOperation ? 1u : 0u) +
                    (memoryOperation ? 1u : 0u) +
                    (feedbackOperation ? 2u : 0u),
                1u,
                report, statistics, error) ||
            !invalidateMappedAllocation(
                m_staging, "VRAM", report, error) ||
            (memoryOperation &&
             !invalidateMappedAllocation(
                 m_memoryResults, "memory results", report, error)) ||
            !finishOperation(
                operationName, validationErrorsBefore,
                report, statistics, error))
        {
            return false;
        }

        std::vector<uint8_t> completedVram(GS_VULKAN_VRAM_SIZE);
        std::memcpy(
            completedVram.data(), m_stagingMap,
            completedVram.size());
        std::vector<GsVulkanMemoryResult> completedResults;
        if (memoryOperation)
        {
            completedResults.resize(cases.size());
            std::memcpy(
                completedResults.data(), m_memoryResultsMap,
                static_cast<size_t>(resultBytes));
        }

        output = std::move(completedVram);
        if (results)
            *results = std::move(completedResults);
        statistics.bytesUploaded += GS_VULKAN_VRAM_SIZE;
        if (feedbackOperation)
            statistics.bytesUploaded += GS_VULKAN_VRAM_SIZE;
        statistics.bytesDownloaded += GS_VULKAN_VRAM_SIZE;
        error.clear();
        return true;
    }

    void VulkanExecutionContext::shutdown() noexcept
    {
        if (m_device != VK_NULL_HANDLE && m_functions.deviceWaitIdle)
            (void)m_functions.deviceWaitIdle(m_device);
        if (m_memoryResultsMap && m_functions.unmapMemory)
        {
            m_functions.unmapMemory(
                m_device, m_memoryResults.memory);
        }
        m_memoryResultsMap = nullptr;
        if (m_memoryCasesMap && m_functions.unmapMemory)
            m_functions.unmapMemory(m_device, m_memoryCases.memory);
        m_memoryCasesMap = nullptr;
        if (m_stagingMap && m_functions.unmapMemory)
            m_functions.unmapMemory(m_device, m_staging.memory);
        m_stagingMap = nullptr;
        if (m_feedbackStagingMap && m_functions.unmapMemory)
        {
            m_functions.unmapMemory(
                m_device, m_feedbackStaging.memory);
        }
        m_feedbackStagingMap = nullptr;
        if (m_fence != VK_NULL_HANDLE && m_functions.destroyFence)
            m_functions.destroyFence(m_device, m_fence, nullptr);
        m_fence = VK_NULL_HANDLE;
        if (m_commandPool != VK_NULL_HANDLE &&
            m_functions.destroyCommandPool)
        {
            m_functions.destroyCommandPool(
                m_device, m_commandPool, nullptr);
        }
        m_commandPool = VK_NULL_HANDLE;
        m_commandBuffer = VK_NULL_HANDLE;
        if (m_descriptorPool != VK_NULL_HANDLE &&
            m_functions.destroyDescriptorPool)
        {
            m_functions.destroyDescriptorPool(
                m_device, m_descriptorPool, nullptr);
        }
        m_descriptorPool = VK_NULL_HANDLE;
        m_descriptorSet = VK_NULL_HANDLE;
        if (m_feedbackNearestDepthCt32TrianglePipeline !=
                VK_NULL_HANDLE &&
            m_functions.destroyPipeline)
        {
            m_functions.destroyPipeline(
                m_device,
                m_feedbackNearestDepthCt32TrianglePipeline,
                nullptr);
        }
        m_feedbackNearestDepthCt32TrianglePipeline = VK_NULL_HANDLE;
        if (m_feedbackNearestDepthCt32TriangleShaderModule !=
                VK_NULL_HANDLE &&
            m_functions.destroyShaderModule)
        {
            m_functions.destroyShaderModule(
                m_device,
                m_feedbackNearestDepthCt32TriangleShaderModule,
                nullptr);
        }
        m_feedbackNearestDepthCt32TriangleShaderModule = VK_NULL_HANDLE;
        if (m_t8GouraudDepthCt32TrianglePipeline != VK_NULL_HANDLE &&
            m_functions.destroyPipeline)
        {
            m_functions.destroyPipeline(
                m_device,
                m_t8GouraudDepthCt32TrianglePipeline,
                nullptr);
        }
        m_t8GouraudDepthCt32TrianglePipeline = VK_NULL_HANDLE;
        if (m_t8GouraudDepthCt32TriangleShaderModule !=
                VK_NULL_HANDLE &&
            m_functions.destroyShaderModule)
        {
            m_functions.destroyShaderModule(
                m_device,
                m_t8GouraudDepthCt32TriangleShaderModule,
                nullptr);
        }
        m_t8GouraudDepthCt32TriangleShaderModule = VK_NULL_HANDLE;
        if (m_gouraudDepthCt32TrianglePipeline != VK_NULL_HANDLE &&
            m_functions.destroyPipeline)
        {
            m_functions.destroyPipeline(
                m_device,
                m_gouraudDepthCt32TrianglePipeline,
                nullptr);
        }
        m_gouraudDepthCt32TrianglePipeline = VK_NULL_HANDLE;
        if (m_gouraudDepthCt32TriangleShaderModule !=
                VK_NULL_HANDLE &&
            m_functions.destroyShaderModule)
        {
            m_functions.destroyShaderModule(
                m_device,
                m_gouraudDepthCt32TriangleShaderModule,
                nullptr);
        }
        m_gouraudDepthCt32TriangleShaderModule = VK_NULL_HANDLE;
        if (m_trianglePipeline != VK_NULL_HANDLE &&
            m_functions.destroyPipeline)
        {
            m_functions.destroyPipeline(
                m_device, m_trianglePipeline, nullptr);
        }
        m_trianglePipeline = VK_NULL_HANDLE;
        if (m_triangleShaderModule != VK_NULL_HANDLE &&
            m_functions.destroyShaderModule)
        {
            m_functions.destroyShaderModule(
                m_device, m_triangleShaderModule, nullptr);
        }
        m_triangleShaderModule = VK_NULL_HANDLE;
        if (m_feedbackLinearDepthCt32SpritePipeline != VK_NULL_HANDLE &&
            m_functions.destroyPipeline)
        {
            m_functions.destroyPipeline(
                m_device,
                m_feedbackLinearDepthCt32SpritePipeline,
                nullptr);
        }
        m_feedbackLinearDepthCt32SpritePipeline = VK_NULL_HANDLE;
        if (m_feedbackLinearDepthCt32SpriteShaderModule !=
                VK_NULL_HANDLE &&
            m_functions.destroyShaderModule)
        {
            m_functions.destroyShaderModule(
                m_device,
                m_feedbackLinearDepthCt32SpriteShaderModule,
                nullptr);
        }
        m_feedbackLinearDepthCt32SpriteShaderModule = VK_NULL_HANDLE;
        if (m_linearCt32SpritePipeline != VK_NULL_HANDLE &&
            m_functions.destroyPipeline)
        {
            m_functions.destroyPipeline(
                m_device, m_linearCt32SpritePipeline, nullptr);
        }
        m_linearCt32SpritePipeline = VK_NULL_HANDLE;
        if (m_linearCt32SpriteShaderModule != VK_NULL_HANDLE &&
            m_functions.destroyShaderModule)
        {
            m_functions.destroyShaderModule(
                m_device, m_linearCt32SpriteShaderModule, nullptr);
        }
        m_linearCt32SpriteShaderModule = VK_NULL_HANDLE;
        if (m_nearestCt32SpritePipeline != VK_NULL_HANDLE &&
            m_functions.destroyPipeline)
        {
            m_functions.destroyPipeline(
                m_device, m_nearestCt32SpritePipeline, nullptr);
        }
        m_nearestCt32SpritePipeline = VK_NULL_HANDLE;
        if (m_nearestCt32SpriteShaderModule != VK_NULL_HANDLE &&
            m_functions.destroyShaderModule)
        {
            m_functions.destroyShaderModule(
                m_device, m_nearestCt32SpriteShaderModule, nullptr);
        }
        m_nearestCt32SpriteShaderModule = VK_NULL_HANDLE;
        if (m_depthCt32SpritePipeline != VK_NULL_HANDLE &&
            m_functions.destroyPipeline)
        {
            m_functions.destroyPipeline(
                m_device, m_depthCt32SpritePipeline, nullptr);
        }
        m_depthCt32SpritePipeline = VK_NULL_HANDLE;
        if (m_depthCt32SpriteShaderModule != VK_NULL_HANDLE &&
            m_functions.destroyShaderModule)
        {
            m_functions.destroyShaderModule(
                m_device, m_depthCt32SpriteShaderModule, nullptr);
        }
        m_depthCt32SpriteShaderModule = VK_NULL_HANDLE;
        if (m_spritePipeline != VK_NULL_HANDLE &&
            m_functions.destroyPipeline)
        {
            m_functions.destroyPipeline(
                m_device, m_spritePipeline, nullptr);
        }
        m_spritePipeline = VK_NULL_HANDLE;
        if (m_spriteShaderModule != VK_NULL_HANDLE &&
            m_functions.destroyShaderModule)
        {
            m_functions.destroyShaderModule(
                m_device, m_spriteShaderModule, nullptr);
        }
        m_spriteShaderModule = VK_NULL_HANDLE;
        if (m_memoryPipeline != VK_NULL_HANDLE &&
            m_functions.destroyPipeline)
        {
            m_functions.destroyPipeline(
                m_device, m_memoryPipeline, nullptr);
        }
        m_memoryPipeline = VK_NULL_HANDLE;
        if (m_memoryShaderModule != VK_NULL_HANDLE &&
            m_functions.destroyShaderModule)
        {
            m_functions.destroyShaderModule(
                m_device, m_memoryShaderModule, nullptr);
        }
        m_memoryShaderModule = VK_NULL_HANDLE;
        if (m_noopPipeline != VK_NULL_HANDLE &&
            m_functions.destroyPipeline)
        {
            m_functions.destroyPipeline(
                m_device, m_noopPipeline, nullptr);
        }
        m_noopPipeline = VK_NULL_HANDLE;
        if (m_noopShaderModule != VK_NULL_HANDLE &&
            m_functions.destroyShaderModule)
        {
            m_functions.destroyShaderModule(
                m_device, m_noopShaderModule, nullptr);
        }
        m_noopShaderModule = VK_NULL_HANDLE;
        if (m_pipelineLayout != VK_NULL_HANDLE &&
            m_functions.destroyPipelineLayout)
        {
            m_functions.destroyPipelineLayout(
                m_device, m_pipelineLayout, nullptr);
        }
        m_pipelineLayout = VK_NULL_HANDLE;
        if (m_descriptorSetLayout != VK_NULL_HANDLE &&
            m_functions.destroyDescriptorSetLayout)
        {
            m_functions.destroyDescriptorSetLayout(
                m_device, m_descriptorSetLayout, nullptr);
        }
        m_descriptorSetLayout = VK_NULL_HANDLE;

        auto destroyAllocation = [&](BufferAllocation &allocation)
        {
            if (allocation.buffer != VK_NULL_HANDLE &&
                m_functions.destroyBuffer)
            {
                m_functions.destroyBuffer(
                    m_device, allocation.buffer, nullptr);
            }
            allocation.buffer = VK_NULL_HANDLE;
            if (allocation.memory != VK_NULL_HANDLE &&
                m_functions.freeMemory)
            {
                m_functions.freeMemory(
                    m_device, allocation.memory, nullptr);
            }
            allocation.memory = VK_NULL_HANDLE;
            allocation.allocationSize = 0u;
            allocation.coherent = false;
        };
        destroyAllocation(m_memoryResults);
        destroyAllocation(m_memoryCases);
        destroyAllocation(m_feedbackStaging);
        destroyAllocation(m_feedbackSnapshot);
        destroyAllocation(m_staging);
        destroyAllocation(m_vram);

        if (m_device != VK_NULL_HANDLE && m_functions.destroyDevice)
            m_functions.destroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
        m_queue = VK_NULL_HANDLE;
        m_physicalDevice = VK_NULL_HANDLE;
        m_instance.reset();
        m_healthy = false;
    }
}
#endif

#if PS2X_HAS_GS_VULKAN
struct GsVulkanService::Impl final
{
    explicit Impl(const GsVulkanServiceConfig &serviceConfig)
        : config(serviceConfig)
    {
    }

    void threadMain()
    {
        ThreadNaming::SetCurrentThreadName("PS2GsVulkan");
        try
        {
            threadMainImpl();
        }
        catch (const std::exception &exception)
        {
            failUnexpectedWorkerExit(
                std::string("Vulkan GS worker exception: ") +
                exception.what());
        }
        catch (...)
        {
            failUnexpectedWorkerExit(
                "Vulkan GS worker exited with an unknown exception");
        }
    }

    void failUnexpectedWorkerExit(std::string message)
    {
        {
            std::lock_guard lock(stateMutex);
            capabilities.compiled = true;
            capabilities.status =
                GsVulkanProbeStatus::ResourceCreationFailed;
            capabilities.detail = message;
            initializationError = message;
            healthy = false;
            initialized = true;
            workerFinished = true;
            available.store(false, std::memory_order_release);
            requestPending = false;
            requestPageUploadSource = {};
            requestPageDownloadDestination = {};
            if (requestInFlight)
            {
                if (activeRequestKind ==
                    GsVulkanRequestKind::MemoryCases)
                {
                    ++statistics.memoryBatchesFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::Ct32Sprite)
                {
                    ++statistics.spriteDrawsFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::DepthCt32Sprite)
                {
                    ++statistics.depthCt32SpriteDrawsFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::NearestCt32Sprite)
                {
                    ++statistics.nearestCt32SpriteDrawsFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::LinearCt32Sprite)
                {
                    ++statistics.linearCt32SpriteDrawsFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::FeedbackLinearDepthCt32Sprite)
                {
                    ++statistics
                          .feedbackLinearDepthCt32SpriteDrawsFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::Ct32Triangle)
                {
                    ++statistics.triangleDrawsFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::GouraudDepthCt32Triangle)
                {
                    ++statistics
                          .gouraudDepthCt32TriangleDrawsFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::T8GouraudDepthCt32Triangle)
                {
                    ++statistics
                          .t8GouraudDepthCt32TriangleDrawsFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::ResidentCt32Sprites)
                {
                    statistics.spriteDrawsFailed +=
                        activeRequestSpriteCount;
                    ++statistics.residentSpriteBatchesFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::ResidentDepthCt32Sprites)
                {
                    statistics.depthCt32SpriteDrawsFailed +=
                        activeRequestDepthCt32SpriteCount;
                    ++statistics.residentDepthCt32SpriteBatchesFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::
                             ResidentFeedbackLinearDepthCt32Sprites)
                {
                    statistics
                        .feedbackLinearDepthCt32SpriteDrawsFailed +=
                        activeRequestFeedbackLinearDepthCt32SpriteCount;
                    ++statistics
                          .residentFeedbackLinearDepthCt32SpriteBatchesFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::ResidentNearestCt32Sprites)
                {
                    statistics.nearestCt32SpriteDrawsFailed +=
                        activeRequestNearestCt32SpriteCount;
                    ++statistics.residentNearestCt32SpriteBatchesFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::ResidentLinearCt32Sprites)
                {
                    statistics.linearCt32SpriteDrawsFailed +=
                        activeRequestLinearCt32SpriteCount;
                    ++statistics.residentLinearCt32SpriteBatchesFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::ResidentCt32Triangles)
                {
                    statistics.triangleDrawsFailed +=
                        activeRequestTriangleCount;
                    ++statistics.residentTriangleBatchesFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::
                             ResidentGouraudDepthCt32Triangles)
                {
                    statistics.gouraudDepthCt32TriangleDrawsFailed +=
                        activeRequestGouraudTriangleCount;
                    ++statistics
                          .residentGouraudDepthCt32TriangleBatchesFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::
                             ResidentT8GouraudDepthCt32Triangles)
                {
                    statistics.t8GouraudDepthCt32TriangleDrawsFailed +=
                        activeRequestT8TriangleCount;
                    ++statistics
                          .residentT8GouraudDepthCt32TriangleBatchesFailed;
                    if (activeRequestHasPageUpload)
                        ++statistics.pageUploadOperationsFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::
                             ResidentFeedbackNearestDepthCt32Triangles)
                {
                    statistics
                        .feedbackNearestDepthCt32TriangleDrawsFailed +=
                        activeRequestFeedbackNearestTriangleCount;
                    ++statistics
                          .residentFeedbackNearestDepthCt32TriangleBatchesFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::UploadPages)
                {
                    ++statistics.pageUploadOperationsFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::DownloadPages)
                {
                    ++statistics.pageDownloadOperationsFailed;
                }
                else if (activeRequestKind ==
                         GsVulkanRequestKind::DownloadFeedbackSnapshot)
                {
                    ++statistics
                          .residentFeedbackSnapshotDownloadsFailed;
                }
                else
                {
                    ++statistics.roundTripsFailed;
                }
                responseOutput.clear();
                responseResults.clear();
                responseError = message;
                responseSucceeded = false;
                responseReady = true;
                requestInFlight = false;
            }
        }
        stateChanged.notify_all();
    }

    void threadMainImpl()
    {
        VulkanExecutionContext context;
        GsVulkanCapabilityReport localCapabilities =
            probeGsVulkanCapabilities(config.probe);
        GsVulkanServiceStatistics localStatistics{};
        std::string localError = localCapabilities.ready()
            ? std::string{}
            : localCapabilities.detail;
        const bool initializedSuccessfully =
            localCapabilities.ready() &&
            context.initialize(config, localCapabilities, localError);

        if (!initializedSuccessfully)
            context.shutdown();
        context.refreshDiagnostics(
            localCapabilities, localStatistics);

        {
            std::lock_guard lock(stateMutex);
            capabilities = localCapabilities;
            statistics = localStatistics;
            initializationError = localError;
            healthy = initializedSuccessfully;
            initialized = true;
            if (!initializedSuccessfully)
                workerFinished = true;
            available.store(
                initializedSuccessfully,
                std::memory_order_release);
        }
        stateChanged.notify_all();
        if (!initializedSuccessfully)
            return;

        for (;;)
        {
            std::vector<uint8_t> input;
            std::vector<uint8_t> feedbackSnapshot;
            std::vector<GsVulkanMemoryCase> memoryCases;
            std::vector<GsVulkanCt32Sprite> sprites;
            std::vector<GsVulkanDepthCt32Sprite> depthCt32Sprites;
            std::vector<GsVulkanNearestCt32Sprite> nearestCt32Sprites;
            std::vector<GsVulkanLinearCt32Sprite> linearCt32Sprites;
            std::vector<GsVulkanFeedbackLinearDepthCt32Sprite>
                feedbackLinearDepthCt32Sprites;
            std::vector<GsVulkanCt32Triangle> triangles;
            std::vector<GsVulkanGouraudDepthCt32Triangle>
                gouraudDepthCt32Triangles;
            std::vector<GsVulkanT8GouraudDepthCt32Triangle>
                t8GouraudDepthCt32Triangles;
            std::vector<GsVulkanResidentT8GouraudDepthCt32Triangle>
                residentT8GouraudDepthCt32Triangles;
            std::vector<GsVulkanFeedbackNearestDepthCt32Triangle>
                feedbackNearestDepthCt32Triangles;
            std::vector<GsVulkanT8Palette> t8Palettes;
            GsVulkanFeedbackSnapshotMode feedbackSnapshotMode =
                GsVulkanFeedbackSnapshotMode::UploadHost;
            std::span<const uint8_t> pageUploadSource;
            std::span<uint8_t> pageDownloadDestination;
            GsVramPageMask pages;
            GsVulkanRequestKind kind =
                GsVulkanRequestKind::RoundTrip;
            {
                std::unique_lock lock(stateMutex);
                stateChanged.wait(lock, [this]
                {
                    return stopping || requestPending;
                });
                if (stopping && !requestPending)
                    break;
                input = std::move(requestInput);
                feedbackSnapshot =
                    std::move(requestFeedbackSnapshot);
                memoryCases = std::move(requestMemoryCases);
                sprites = std::move(requestSprites);
                depthCt32Sprites =
                    std::move(requestDepthCt32Sprites);
                nearestCt32Sprites =
                    std::move(requestNearestCt32Sprites);
                linearCt32Sprites =
                    std::move(requestLinearCt32Sprites);
                feedbackLinearDepthCt32Sprites =
                    std::move(
                        requestFeedbackLinearDepthCt32Sprites);
                triangles = std::move(requestTriangles);
                gouraudDepthCt32Triangles =
                    std::move(requestGouraudDepthCt32Triangles);
                t8GouraudDepthCt32Triangles =
                    std::move(requestT8GouraudDepthCt32Triangles);
                residentT8GouraudDepthCt32Triangles =
                    std::move(
                        requestResidentT8GouraudDepthCt32Triangles);
                feedbackNearestDepthCt32Triangles =
                    std::move(
                        requestFeedbackNearestDepthCt32Triangles);
                t8Palettes = std::move(requestT8Palettes);
                feedbackSnapshotMode = requestFeedbackSnapshotMode;
                pageUploadSource = requestPageUploadSource;
                pageDownloadDestination =
                    requestPageDownloadDestination;
                requestPageUploadSource = {};
                requestPageDownloadDestination = {};
                pages = requestPages;
                kind = requestKind;
                activeRequestKind = kind;
                activeRequestHasPageUpload =
                    !pageUploadSource.empty();
                activeRequestSpriteCount = sprites.size();
                activeRequestDepthCt32SpriteCount =
                    depthCt32Sprites.size();
                activeRequestNearestCt32SpriteCount =
                    nearestCt32Sprites.size();
                activeRequestLinearCt32SpriteCount =
                    linearCt32Sprites.size();
                activeRequestFeedbackLinearDepthCt32SpriteCount =
                    feedbackLinearDepthCt32Sprites.size();
                activeRequestTriangleCount = triangles.size();
                activeRequestGouraudTriangleCount =
                    gouraudDepthCt32Triangles.size();
                activeRequestT8TriangleCount =
                    kind == GsVulkanRequestKind::
                                ResidentT8GouraudDepthCt32Triangles
                        ? residentT8GouraudDepthCt32Triangles.size()
                        : t8GouraudDepthCt32Triangles.size();
                activeRequestFeedbackNearestTriangleCount =
                    feedbackNearestDepthCt32Triangles.size();
                requestPending = false;
                requestInFlight = true;
            }

            std::vector<uint8_t> output;
            std::vector<GsVulkanMemoryResult> memoryResults;
            std::string operationError;
            bool succeeded = false;
            const uint64_t fenceWaitsBefore =
                localStatistics.fenceWaits;
            const uint64_t fenceWaitNanosecondsBefore =
                localStatistics.fenceWaitNanoseconds;
            if (kind == GsVulkanRequestKind::MemoryCases)
            {
                succeeded = context.executeMemoryCases(
                    input, memoryCases, output, memoryResults,
                    localCapabilities, localStatistics,
                    operationError);
            }
            else if (kind == GsVulkanRequestKind::Ct32Sprite)
            {
                succeeded = context.executeCt32Sprite(
                    input, sprites.front(), output,
                    localCapabilities, localStatistics,
                    operationError);
            }
            else if (kind == GsVulkanRequestKind::DepthCt32Sprite)
            {
                succeeded = context.executeDepthCt32Sprite(
                    input, depthCt32Sprites.front(), output,
                    localCapabilities, localStatistics,
                    operationError);
            }
            else if (kind == GsVulkanRequestKind::NearestCt32Sprite)
            {
                succeeded = context.executeNearestCt32Sprite(
                    input, nearestCt32Sprites.front(), output,
                    localCapabilities, localStatistics,
                    operationError);
            }
            else if (kind == GsVulkanRequestKind::LinearCt32Sprite)
            {
                succeeded = context.executeLinearCt32Sprite(
                    input, linearCt32Sprites.front(), output,
                    localCapabilities, localStatistics,
                    operationError);
            }
            else if (kind ==
                     GsVulkanRequestKind::FeedbackLinearDepthCt32Sprite)
            {
                succeeded =
                    context.executeFeedbackLinearDepthCt32Sprite(
                        input,
                        feedbackSnapshot,
                        feedbackLinearDepthCt32Sprites.front(),
                        output,
                        localCapabilities,
                        localStatistics,
                        operationError);
            }
            else if (kind == GsVulkanRequestKind::Ct32Triangle)
            {
                succeeded = context.executeCt32Triangle(
                    input, triangles.front(), output,
                    localCapabilities, localStatistics,
                    operationError);
            }
            else if (kind ==
                     GsVulkanRequestKind::GouraudDepthCt32Triangle)
            {
                succeeded = context.executeGouraudDepthCt32Triangle(
                    input, gouraudDepthCt32Triangles.front(), output,
                    localCapabilities, localStatistics,
                    operationError);
            }
            else if (kind ==
                     GsVulkanRequestKind::T8GouraudDepthCt32Triangle)
            {
                succeeded = context.executeT8GouraudDepthCt32Triangle(
                    input, t8GouraudDepthCt32Triangles.front(), output,
                    localCapabilities, localStatistics,
                    operationError);
            }
            else if (kind == GsVulkanRequestKind::UploadPages)
            {
                succeeded = context.uploadVramPages(
                    pageUploadSource, pages, localCapabilities,
                    localStatistics, operationError);
            }
            else if (kind == GsVulkanRequestKind::DownloadPages)
            {
                succeeded = context.downloadVramPages(
                    pageDownloadDestination, pages, localCapabilities,
                    localStatistics, operationError);
            }
            else if (kind ==
                     GsVulkanRequestKind::DownloadFeedbackSnapshot)
            {
                succeeded = context.downloadFeedbackSnapshot(
                    pageDownloadDestination, localCapabilities,
                    localStatistics, operationError);
            }
            else if (kind == GsVulkanRequestKind::ResidentCt32Sprites)
            {
                succeeded = context.executeResidentCt32Sprites(
                    sprites, localCapabilities, localStatistics,
                    operationError);
            }
            else if (kind ==
                     GsVulkanRequestKind::ResidentDepthCt32Sprites)
            {
                succeeded = context.executeResidentDepthCt32Sprites(
                    depthCt32Sprites, localCapabilities,
                    localStatistics, operationError);
            }
            else if (kind ==
                     GsVulkanRequestKind::ResidentNearestCt32Sprites)
            {
                succeeded = context.executeResidentNearestCt32Sprites(
                    nearestCt32Sprites, localCapabilities,
                    localStatistics, operationError);
            }
            else if (kind == GsVulkanRequestKind::ResidentCt32Triangles)
            {
                succeeded = context.executeResidentCt32Triangles(
                    triangles, localCapabilities, localStatistics,
                    operationError);
            }
            else if (kind ==
                     GsVulkanRequestKind::
                         ResidentGouraudDepthCt32Triangles)
            {
                succeeded = context.
                    executeResidentGouraudDepthCt32Triangles(
                        gouraudDepthCt32Triangles,
                        localCapabilities, localStatistics,
                        operationError);
            }
            else if (kind ==
                     GsVulkanRequestKind::
                         ResidentT8GouraudDepthCt32Triangles)
            {
                succeeded = context.
                    executeResidentT8GouraudDepthCt32Triangles(
                        residentT8GouraudDepthCt32Triangles,
                        t8Palettes,
                        pageUploadSource, pages,
                        localCapabilities, localStatistics,
                        operationError);
            }
            else if (kind ==
                     GsVulkanRequestKind::
                         ResidentFeedbackNearestDepthCt32Triangles)
            {
                succeeded = context.
                    executeResidentFeedbackNearestDepthCt32Triangles(
                        feedbackSnapshot,
                        feedbackSnapshotMode,
                        feedbackNearestDepthCt32Triangles,
                        localCapabilities, localStatistics,
                        operationError);
            }
            else if (kind ==
                     GsVulkanRequestKind::
                         ResidentFeedbackLinearDepthCt32Sprites)
            {
                succeeded = context.
                    executeResidentFeedbackLinearDepthCt32Sprites(
                        feedbackSnapshot,
                        feedbackLinearDepthCt32Sprites,
                        localCapabilities,
                        localStatistics,
                        operationError);
            }
            else if (kind ==
                     GsVulkanRequestKind::ResidentLinearCt32Sprites)
            {
                succeeded = context.executeResidentLinearCt32Sprites(
                    linearCt32Sprites, localCapabilities,
                    localStatistics, operationError);
            }
            else
            {
                succeeded = context.roundTrip(
                    input, output, localCapabilities,
                    localStatistics, operationError);
            }
            if (kind == GsVulkanRequestKind::MemoryCases)
            {
                if (succeeded)
                {
                    ++localStatistics.memoryBatchesCompleted;
                    localStatistics.memoryCasesExecuted +=
                        memoryCases.size();
                }
                else
                {
                    ++localStatistics.memoryBatchesFailed;
                }
            }
            else if (kind == GsVulkanRequestKind::Ct32Sprite)
            {
                if (succeeded)
                {
                    ++localStatistics.spriteDrawsCompleted;
                    localStatistics.spritePixelsExecuted +=
                        static_cast<uint64_t>(
                            sprites.front().x1 - sprites.front().x0) *
                        static_cast<uint64_t>(
                            sprites.front().y1 - sprites.front().y0);
                }
                else
                {
                    ++localStatistics.spriteDrawsFailed;
                }
            }
            else if (kind == GsVulkanRequestKind::DepthCt32Sprite)
            {
                if (succeeded)
                {
                    const GsVulkanDepthCt32Sprite &sprite =
                        depthCt32Sprites.front();
                    ++localStatistics.depthCt32SpriteDrawsCompleted;
                    localStatistics.depthCt32SpritePixelsExecuted +=
                        static_cast<uint64_t>(
                            sprite.boundsX1 - sprite.boundsX0) *
                        static_cast<uint64_t>(
                            sprite.boundsY1 - sprite.boundsY0);
                }
                else
                {
                    ++localStatistics.depthCt32SpriteDrawsFailed;
                }
            }
            else if (kind == GsVulkanRequestKind::NearestCt32Sprite)
            {
                if (succeeded)
                {
                    const GsVulkanNearestCt32Sprite &sprite =
                        nearestCt32Sprites.front();
                    ++localStatistics.nearestCt32SpriteDrawsCompleted;
                    localStatistics.nearestCt32SpritePixelsExecuted +=
                        static_cast<uint64_t>(
                            sprite.boundsX1 - sprite.boundsX0) *
                        static_cast<uint64_t>(
                            sprite.boundsY1 - sprite.boundsY0);
                }
                else
                {
                    ++localStatistics.nearestCt32SpriteDrawsFailed;
                }
            }
            else if (kind == GsVulkanRequestKind::LinearCt32Sprite)
            {
                if (succeeded)
                {
                    const GsVulkanLinearCt32Sprite &sprite =
                        linearCt32Sprites.front();
                    ++localStatistics.linearCt32SpriteDrawsCompleted;
                    localStatistics.linearCt32SpritePixelsExecuted +=
                        static_cast<uint64_t>(
                            sprite.boundsX1 - sprite.boundsX0) *
                        static_cast<uint64_t>(
                            sprite.boundsY1 - sprite.boundsY0);
                }
                else
                {
                    ++localStatistics.linearCt32SpriteDrawsFailed;
                }
            }
            else if (kind ==
                     GsVulkanRequestKind::FeedbackLinearDepthCt32Sprite)
            {
                if (succeeded)
                {
                    const GsVulkanFeedbackLinearDepthCt32Sprite &sprite =
                        feedbackLinearDepthCt32Sprites.front();
                    ++localStatistics
                          .feedbackLinearDepthCt32SpriteDrawsCompleted;
                    localStatistics
                        .feedbackLinearDepthCt32SpritePixelsExecuted +=
                        static_cast<uint64_t>(
                            sprite.boundsX1 - sprite.boundsX0) *
                        static_cast<uint64_t>(
                            sprite.boundsY1 - sprite.boundsY0);
                }
                else
                {
                    ++localStatistics
                          .feedbackLinearDepthCt32SpriteDrawsFailed;
                }
            }
            else if (kind == GsVulkanRequestKind::Ct32Triangle)
            {
                if (succeeded)
                {
                    const GsVulkanCt32Triangle &triangle = triangles.front();
                    ++localStatistics.triangleDrawsCompleted;
                    localStatistics.triangleCandidatePixelsExecuted +=
                        static_cast<uint64_t>(
                            triangle.boundsX1 - triangle.boundsX0) *
                        static_cast<uint64_t>(
                            triangle.boundsY1 - triangle.boundsY0);
                }
                else
                {
                    ++localStatistics.triangleDrawsFailed;
                }
            }
            else if (kind ==
                     GsVulkanRequestKind::GouraudDepthCt32Triangle)
            {
                if (succeeded)
                {
                    const GsVulkanGouraudDepthCt32Triangle &triangle =
                        gouraudDepthCt32Triangles.front();
                    ++localStatistics
                          .gouraudDepthCt32TriangleDrawsCompleted;
                    localStatistics
                        .gouraudDepthCt32TriangleCandidatePixelsExecuted +=
                        static_cast<uint64_t>(
                            triangle.boundsX1 - triangle.boundsX0) *
                        static_cast<uint64_t>(
                            triangle.boundsY1 - triangle.boundsY0);
                }
                else
                {
                    ++localStatistics
                          .gouraudDepthCt32TriangleDrawsFailed;
                }
            }
            else if (kind ==
                     GsVulkanRequestKind::T8GouraudDepthCt32Triangle)
            {
                if (succeeded)
                {
                    const GsVulkanT8GouraudDepthCt32Triangle &triangle =
                        t8GouraudDepthCt32Triangles.front();
                    ++localStatistics
                          .t8GouraudDepthCt32TriangleDrawsCompleted;
                    localStatistics
                        .t8GouraudDepthCt32TriangleCandidatePixelsExecuted +=
                        static_cast<uint64_t>(
                            triangle.boundsX1 - triangle.boundsX0) *
                        static_cast<uint64_t>(
                            triangle.boundsY1 - triangle.boundsY0);
                }
                else
                {
                    ++localStatistics
                          .t8GouraudDepthCt32TriangleDrawsFailed;
                }
            }
            else if (kind == GsVulkanRequestKind::ResidentCt32Sprites)
            {
                if (succeeded)
                {
                    localStatistics.spriteDrawsCompleted += sprites.size();
                    for (const GsVulkanCt32Sprite &sprite : sprites)
                    {
                        localStatistics.spritePixelsExecuted +=
                            static_cast<uint64_t>(sprite.x1 - sprite.x0) *
                            static_cast<uint64_t>(sprite.y1 - sprite.y0);
                    }
                    ++localStatistics.residentSpriteBatchesCompleted;
                    localStatistics.largestResidentSpriteBatch = std::max(
                        localStatistics.largestResidentSpriteBatch,
                        static_cast<uint64_t>(sprites.size()));
                }
                else
                {
                    localStatistics.spriteDrawsFailed += sprites.size();
                    ++localStatistics.residentSpriteBatchesFailed;
                }
            }
            else if (kind ==
                     GsVulkanRequestKind::ResidentDepthCt32Sprites)
            {
                if (succeeded)
                {
                    localStatistics.depthCt32SpriteDrawsCompleted +=
                        depthCt32Sprites.size();
                    for (const GsVulkanDepthCt32Sprite &sprite :
                         depthCt32Sprites)
                    {
                        localStatistics.depthCt32SpritePixelsExecuted +=
                            static_cast<uint64_t>(
                                sprite.boundsX1 - sprite.boundsX0) *
                            static_cast<uint64_t>(
                                sprite.boundsY1 - sprite.boundsY0);
                    }
                    ++localStatistics
                          .residentDepthCt32SpriteBatchesCompleted;
                    localStatistics.largestResidentDepthCt32SpriteBatch =
                        std::max(
                            localStatistics
                                .largestResidentDepthCt32SpriteBatch,
                            static_cast<uint64_t>(
                                depthCt32Sprites.size()));
                }
                else
                {
                    localStatistics.depthCt32SpriteDrawsFailed +=
                        depthCt32Sprites.size();
                    ++localStatistics
                          .residentDepthCt32SpriteBatchesFailed;
                }
            }
            else if (kind ==
                     GsVulkanRequestKind::
                         ResidentFeedbackLinearDepthCt32Sprites)
            {
                if (succeeded)
                {
                    localStatistics
                        .feedbackLinearDepthCt32SpriteDrawsCompleted +=
                        feedbackLinearDepthCt32Sprites.size();
                    for (const GsVulkanFeedbackLinearDepthCt32Sprite &sprite :
                         feedbackLinearDepthCt32Sprites)
                    {
                        localStatistics
                            .feedbackLinearDepthCt32SpritePixelsExecuted +=
                            static_cast<uint64_t>(
                                sprite.boundsX1 - sprite.boundsX0) *
                            static_cast<uint64_t>(
                                sprite.boundsY1 - sprite.boundsY0);
                    }
                    ++localStatistics
                          .residentFeedbackLinearDepthCt32SpriteBatchesCompleted;
                    localStatistics
                        .largestResidentFeedbackLinearDepthCt32SpriteBatch =
                        std::max(
                            localStatistics
                                .largestResidentFeedbackLinearDepthCt32SpriteBatch,
                            static_cast<uint64_t>(
                                feedbackLinearDepthCt32Sprites.size()));
                }
                else
                {
                    localStatistics
                        .feedbackLinearDepthCt32SpriteDrawsFailed +=
                        feedbackLinearDepthCt32Sprites.size();
                    ++localStatistics
                          .residentFeedbackLinearDepthCt32SpriteBatchesFailed;
                }
            }
            else if (kind ==
                     GsVulkanRequestKind::ResidentNearestCt32Sprites)
            {
                if (succeeded)
                {
                    localStatistics.nearestCt32SpriteDrawsCompleted +=
                        nearestCt32Sprites.size();
                    for (const GsVulkanNearestCt32Sprite &sprite :
                         nearestCt32Sprites)
                    {
                        localStatistics.nearestCt32SpritePixelsExecuted +=
                            static_cast<uint64_t>(
                                sprite.boundsX1 - sprite.boundsX0) *
                            static_cast<uint64_t>(
                                sprite.boundsY1 - sprite.boundsY0);
                    }
                    ++localStatistics
                          .residentNearestCt32SpriteBatchesCompleted;
                    localStatistics.largestResidentNearestCt32SpriteBatch =
                        std::max(
                            localStatistics
                                .largestResidentNearestCt32SpriteBatch,
                            static_cast<uint64_t>(
                                nearestCt32Sprites.size()));
                }
                else
                {
                    localStatistics.nearestCt32SpriteDrawsFailed +=
                        nearestCt32Sprites.size();
                    ++localStatistics
                          .residentNearestCt32SpriteBatchesFailed;
                }
            }
            else if (kind ==
                     GsVulkanRequestKind::ResidentLinearCt32Sprites)
            {
                if (succeeded)
                {
                    localStatistics.linearCt32SpriteDrawsCompleted +=
                        linearCt32Sprites.size();
                    for (const GsVulkanLinearCt32Sprite &sprite :
                         linearCt32Sprites)
                    {
                        localStatistics.linearCt32SpritePixelsExecuted +=
                            static_cast<uint64_t>(
                                sprite.boundsX1 - sprite.boundsX0) *
                            static_cast<uint64_t>(
                                sprite.boundsY1 - sprite.boundsY0);
                    }
                    ++localStatistics
                          .residentLinearCt32SpriteBatchesCompleted;
                    localStatistics.largestResidentLinearCt32SpriteBatch =
                        std::max(
                            localStatistics
                                .largestResidentLinearCt32SpriteBatch,
                            static_cast<uint64_t>(
                                linearCt32Sprites.size()));
                }
                else
                {
                    localStatistics.linearCt32SpriteDrawsFailed +=
                        linearCt32Sprites.size();
                    ++localStatistics
                          .residentLinearCt32SpriteBatchesFailed;
                }
            }
            else if (kind == GsVulkanRequestKind::ResidentCt32Triangles)
            {
                if (succeeded)
                {
                    localStatistics.triangleDrawsCompleted += triangles.size();
                    for (const GsVulkanCt32Triangle &triangle : triangles)
                    {
                        localStatistics.triangleCandidatePixelsExecuted +=
                            static_cast<uint64_t>(
                                triangle.boundsX1 - triangle.boundsX0) *
                            static_cast<uint64_t>(
                                triangle.boundsY1 - triangle.boundsY0);
                    }
                    ++localStatistics.residentTriangleBatchesCompleted;
                    localStatistics.largestResidentTriangleBatch = std::max(
                        localStatistics.largestResidentTriangleBatch,
                        static_cast<uint64_t>(triangles.size()));
                }
                else
                {
                    localStatistics.triangleDrawsFailed += triangles.size();
                    ++localStatistics.residentTriangleBatchesFailed;
                }
            }
            else if (kind ==
                     GsVulkanRequestKind::
                         ResidentGouraudDepthCt32Triangles)
            {
                if (succeeded)
                {
                    localStatistics
                        .gouraudDepthCt32TriangleDrawsCompleted +=
                        gouraudDepthCt32Triangles.size();
                    for (const auto &triangle :
                         gouraudDepthCt32Triangles)
                    {
                        localStatistics
                            .gouraudDepthCt32TriangleCandidatePixelsExecuted +=
                            static_cast<uint64_t>(
                                triangle.boundsX1 - triangle.boundsX0) *
                            static_cast<uint64_t>(
                                triangle.boundsY1 - triangle.boundsY0);
                    }
                    ++localStatistics
                          .residentGouraudDepthCt32TriangleBatchesCompleted;
                    localStatistics
                        .largestResidentGouraudDepthCt32TriangleBatch =
                        std::max(
                            localStatistics
                                .largestResidentGouraudDepthCt32TriangleBatch,
                            static_cast<uint64_t>(
                                gouraudDepthCt32Triangles.size()));
                }
                else
                {
                    localStatistics
                        .gouraudDepthCt32TriangleDrawsFailed +=
                        gouraudDepthCt32Triangles.size();
                    ++localStatistics
                          .residentGouraudDepthCt32TriangleBatchesFailed;
                }
            }
            else if (kind ==
                     GsVulkanRequestKind::
                         ResidentT8GouraudDepthCt32Triangles)
            {
                if (succeeded)
                {
                    localStatistics
                        .t8GouraudDepthCt32TriangleDrawsCompleted +=
                        residentT8GouraudDepthCt32Triangles.size();
                    for (const auto &triangle :
                         residentT8GouraudDepthCt32Triangles)
                    {
                        localStatistics
                            .t8GouraudDepthCt32TriangleCandidatePixelsExecuted +=
                            static_cast<uint64_t>(
                                triangle.boundsX1 - triangle.boundsX0) *
                            static_cast<uint64_t>(
                                triangle.boundsY1 - triangle.boundsY0);
                    }
                    ++localStatistics
                          .residentT8GouraudDepthCt32TriangleBatchesCompleted;
                    localStatistics
                        .largestResidentT8GouraudDepthCt32TriangleBatch =
                        std::max(
                            localStatistics
                                .largestResidentT8GouraudDepthCt32TriangleBatch,
                            static_cast<uint64_t>(
                                residentT8GouraudDepthCt32Triangles.size()));
                }
                else
                {
                    localStatistics
                        .t8GouraudDepthCt32TriangleDrawsFailed +=
                        residentT8GouraudDepthCt32Triangles.size();
                    ++localStatistics
                          .residentT8GouraudDepthCt32TriangleBatchesFailed;
                }
            }
            else if (kind ==
                     GsVulkanRequestKind::
                         ResidentFeedbackNearestDepthCt32Triangles)
            {
                if (succeeded)
                {
                    localStatistics
                        .feedbackNearestDepthCt32TriangleDrawsCompleted +=
                        feedbackNearestDepthCt32Triangles.size();
                    for (const auto &triangle :
                         feedbackNearestDepthCt32Triangles)
                    {
                        localStatistics
                            .feedbackNearestDepthCt32TriangleCandidatePixelsExecuted +=
                            static_cast<uint64_t>(
                                triangle.boundsX1 - triangle.boundsX0) *
                            static_cast<uint64_t>(
                                triangle.boundsY1 - triangle.boundsY0);
                    }
                    ++localStatistics
                          .residentFeedbackNearestDepthCt32TriangleBatchesCompleted;
                    localStatistics
                        .largestResidentFeedbackNearestDepthCt32TriangleBatch =
                        std::max(
                            localStatistics
                                .largestResidentFeedbackNearestDepthCt32TriangleBatch,
                            static_cast<uint64_t>(
                                feedbackNearestDepthCt32Triangles.size()));
                }
                else
                {
                    localStatistics
                        .feedbackNearestDepthCt32TriangleDrawsFailed +=
                        feedbackNearestDepthCt32Triangles.size();
                    ++localStatistics
                          .residentFeedbackNearestDepthCt32TriangleBatchesFailed;
                }
            }
            else if (kind == GsVulkanRequestKind::UploadPages)
            {
                if (succeeded)
                    ++localStatistics.pageUploadOperationsCompleted;
                else
                    ++localStatistics.pageUploadOperationsFailed;
            }
            else if (kind == GsVulkanRequestKind::DownloadPages)
            {
                if (succeeded)
                    ++localStatistics.pageDownloadOperationsCompleted;
                else
                    ++localStatistics.pageDownloadOperationsFailed;
            }
            else if (kind ==
                     GsVulkanRequestKind::DownloadFeedbackSnapshot)
            {
                if (succeeded)
                {
                    ++localStatistics
                          .residentFeedbackSnapshotsDownloaded;
                }
                else
                {
                    ++localStatistics
                          .residentFeedbackSnapshotDownloadsFailed;
                }
            }
            else if (succeeded)
            {
                ++localStatistics.roundTripsCompleted;
            }
            else
            {
                ++localStatistics.roundTripsFailed;
            }
            if (kind ==
                    GsVulkanRequestKind::
                        ResidentT8GouraudDepthCt32Triangles &&
                !pageUploadSource.empty())
            {
                if (succeeded)
                    ++localStatistics.pageUploadOperationsCompleted;
                else
                    ++localStatistics.pageUploadOperationsFailed;
            }
            GsVulkanRequestStatistics &requestStatistics =
                localStatistics.requestsByKind[
                    gsVulkanRequestKindIndex(kind)];
            requestStatistics.fenceWaits +=
                localStatistics.fenceWaits - fenceWaitsBefore;
            requestStatistics.fenceWaitNanoseconds +=
                localStatistics.fenceWaitNanoseconds -
                fenceWaitNanosecondsBefore;
            context.refreshDiagnostics(
                localCapabilities, localStatistics);

            {
                std::lock_guard lock(stateMutex);
                capabilities = localCapabilities;
                statistics = localStatistics;
                healthy = context.healthy();
                available.store(healthy, std::memory_order_release);
                responseOutput = std::move(output);
                responseResults = std::move(memoryResults);
                responseError = std::move(operationError);
                responseSucceeded = succeeded;
                responseReady = true;
                requestInFlight = false;
            }
            stateChanged.notify_all();
        }

        context.refreshDiagnostics(localCapabilities, localStatistics);
        context.shutdown();
        context.refreshDiagnostics(localCapabilities, localStatistics);
        if (localCapabilities.status == GsVulkanProbeStatus::Ready &&
            localStatistics.validationErrors != 0u)
        {
            localCapabilities.status =
                GsVulkanProbeStatus::ValidationError;
            localCapabilities.detail =
                "Vulkan validation reported an error during service shutdown";
        }
        {
            std::lock_guard lock(stateMutex);
            capabilities = std::move(localCapabilities);
            statistics = localStatistics;
            healthy = false;
            workerFinished = true;
            available.store(false, std::memory_order_release);
        }
        stateChanged.notify_all();
    }

    bool executeRequest(
        GsVulkanRequestKind kind,
        std::vector<uint8_t> input,
        std::vector<uint8_t> feedbackSnapshot,
        std::vector<GsVulkanMemoryCase> memoryCases,
        std::vector<GsVulkanCt32Sprite> sprites,
        std::vector<GsVulkanDepthCt32Sprite> depthCt32Sprites,
        std::vector<GsVulkanNearestCt32Sprite> nearestCt32Sprites,
        std::vector<GsVulkanLinearCt32Sprite> linearCt32Sprites,
        std::vector<GsVulkanFeedbackLinearDepthCt32Sprite>
            feedbackLinearDepthCt32Sprites,
        std::vector<GsVulkanCt32Triangle> triangles,
        GsVramPageMask pages,
        std::vector<uint8_t> &output,
        std::vector<GsVulkanMemoryResult> *memoryResults,
        std::string *error)
    {
        return executeRequest(
            kind,
            std::move(input),
            std::move(feedbackSnapshot),
            std::move(memoryCases),
            std::move(sprites),
            std::move(depthCt32Sprites),
            std::move(nearestCt32Sprites),
            std::move(linearCt32Sprites),
            std::move(feedbackLinearDepthCt32Sprites),
            std::move(triangles),
            {},
            {},
            pages,
            output,
            memoryResults,
            error);
    }

    bool executeRequest(
        GsVulkanRequestKind kind,
        std::vector<uint8_t> input,
        std::vector<uint8_t> feedbackSnapshot,
        std::vector<GsVulkanMemoryCase> memoryCases,
        std::vector<GsVulkanCt32Sprite> sprites,
        std::vector<GsVulkanDepthCt32Sprite> depthCt32Sprites,
        std::vector<GsVulkanNearestCt32Sprite> nearestCt32Sprites,
        std::vector<GsVulkanLinearCt32Sprite> linearCt32Sprites,
        std::vector<GsVulkanFeedbackLinearDepthCt32Sprite>
            feedbackLinearDepthCt32Sprites,
        std::vector<GsVulkanCt32Triangle> triangles,
        std::vector<GsVulkanGouraudDepthCt32Triangle>
            gouraudDepthCt32Triangles,
        std::vector<GsVulkanT8GouraudDepthCt32Triangle>
            t8GouraudDepthCt32Triangles,
        GsVramPageMask pages,
        std::vector<uint8_t> &output,
        std::vector<GsVulkanMemoryResult> *memoryResults,
        std::string *error,
        std::vector<GsVulkanResidentT8GouraudDepthCt32Triangle>
            residentT8GouraudDepthCt32Triangles = {},
        std::vector<GsVulkanT8Palette> t8Palettes = {},
        std::span<const uint8_t> pageUploadSource = {},
        std::span<uint8_t> pageDownloadDestination = {},
        std::vector<GsVulkanFeedbackNearestDepthCt32Triangle>
            feedbackNearestDepthCt32Triangles = {},
        GsVulkanFeedbackSnapshotMode feedbackSnapshotMode =
            GsVulkanFeedbackSnapshotMode::UploadHost)
    {
        bool pageTransferValid =
            pageUploadSource.empty() &&
            pageDownloadDestination.empty() &&
            !pages.any();
        if (kind == GsVulkanRequestKind::UploadPages)
        {
            pageTransferValid =
                pageUploadSource.size() == GS_VULKAN_VRAM_SIZE &&
                pageDownloadDestination.empty() &&
                pages.any();
        }
        else if (kind ==
                     GsVulkanRequestKind::
                         ResidentT8GouraudDepthCt32Triangles &&
                 !pageUploadSource.empty())
        {
            pageTransferValid =
                pageUploadSource.size() == GS_VULKAN_VRAM_SIZE &&
                pageDownloadDestination.empty() &&
                pages.any();
        }
        else if (kind == GsVulkanRequestKind::DownloadPages)
        {
            pageTransferValid =
                pageUploadSource.empty() &&
                pageDownloadDestination.size() == GS_VULKAN_VRAM_SIZE &&
                pages.any();
        }
        else if (kind ==
                 GsVulkanRequestKind::DownloadFeedbackSnapshot)
        {
            pageTransferValid =
                pageUploadSource.empty() &&
                pageDownloadDestination.size() == GS_VULKAN_VRAM_SIZE &&
                !pages.any();
        }
        if (!pageTransferValid)
        {
            if (error)
                *error = "invalid Vulkan GS page-transfer request storage";
            return false;
        }

        std::lock_guard callLock(callMutex);
        std::unique_lock stateLock(stateMutex);
        if (!healthy || stopping || workerFinished)
        {
            if (error)
            {
                if (stopping || workerFinished)
                    *error = "Vulkan GS service is shut down";
                else
                {
                    *error = capabilities.detail.empty()
                        ? "Vulkan GS service is not healthy"
                        : capabilities.detail;
                }
            }
            return false;
        }

        requestKind = kind;
        requestInput = std::move(input);
        requestFeedbackSnapshot = std::move(feedbackSnapshot);
        requestMemoryCases = std::move(memoryCases);
        requestSprites = std::move(sprites);
        requestDepthCt32Sprites = std::move(depthCt32Sprites);
        requestNearestCt32Sprites = std::move(nearestCt32Sprites);
        requestLinearCt32Sprites = std::move(linearCt32Sprites);
        requestFeedbackLinearDepthCt32Sprites =
            std::move(feedbackLinearDepthCt32Sprites);
        requestTriangles = std::move(triangles);
        requestGouraudDepthCt32Triangles =
            std::move(gouraudDepthCt32Triangles);
        requestT8GouraudDepthCt32Triangles =
            std::move(t8GouraudDepthCt32Triangles);
        requestResidentT8GouraudDepthCt32Triangles =
            std::move(residentT8GouraudDepthCt32Triangles);
        requestT8Palettes = std::move(t8Palettes);
        requestFeedbackNearestDepthCt32Triangles =
            std::move(feedbackNearestDepthCt32Triangles);
        requestFeedbackSnapshotMode = feedbackSnapshotMode;
        requestPageUploadSource = pageUploadSource;
        requestPageDownloadDestination = pageDownloadDestination;
        requestPages = pages;
        responseOutput.clear();
        responseResults.clear();
        responseError.clear();
        responseSucceeded = false;
        responseReady = false;
        requestPending = true;
        const auto requestWaitStart =
            std::chrono::steady_clock::now();
        stateLock.unlock();
        stateChanged.notify_all();
        stateLock.lock();
        stateChanged.wait(stateLock, [this]
        {
            return responseReady || workerFinished;
        });
        const uint64_t requestWaitNanoseconds =
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - requestWaitStart)
                    .count());
        requestWaits.fetch_add(1u, std::memory_order_relaxed);
        requestWaitTotalNanoseconds.fetch_add(
            requestWaitNanoseconds, std::memory_order_relaxed);
        const size_t requestKindIndex =
            gsVulkanRequestKindIndex(kind);
        requestWaitsByKind[requestKindIndex].fetch_add(
            1u, std::memory_order_relaxed);
        requestWaitNanosecondsByKind[requestKindIndex].fetch_add(
            requestWaitNanoseconds, std::memory_order_relaxed);
        uint64_t maximumWait =
            maximumRequestWaitNanoseconds.load(
                std::memory_order_relaxed);
        while (maximumWait < requestWaitNanoseconds &&
               !maximumRequestWaitNanoseconds.compare_exchange_weak(
                   maximumWait,
                   requestWaitNanoseconds,
                   std::memory_order_relaxed,
                   std::memory_order_relaxed))
        {
        }

        if (!responseReady)
        {
            if (error)
            {
                *error =
                    "Vulkan GS worker stopped before completing the request";
            }
            return false;
        }
        const bool succeeded = responseSucceeded;
        std::vector<uint8_t> completedOutput =
            std::move(responseOutput);
        std::vector<GsVulkanMemoryResult> completedResults =
            std::move(responseResults);
        std::string operationError = std::move(responseError);
        responseReady = false;
        stateLock.unlock();

        if (!succeeded)
        {
            if (error)
                *error = std::move(operationError);
            return false;
        }
        output = std::move(completedOutput);
        if (memoryResults)
            *memoryResults = std::move(completedResults);
        if (error)
            error->clear();
        return true;
    }

    void stopAndJoin() noexcept
    {
        {
            std::lock_guard lock(stateMutex);
            stopping = true;
            available.store(false, std::memory_order_release);
        }
        stateChanged.notify_all();
        if (worker.joinable())
            worker.join();
    }

    GsVulkanServiceConfig config;
    mutable std::mutex stateMutex;
    std::mutex callMutex;
    std::condition_variable stateChanged;
    std::thread worker;
    GsVulkanCapabilityReport capabilities;
    GsVulkanServiceStatistics statistics;
    std::string initializationError;
    std::vector<uint8_t> requestInput;
    std::vector<uint8_t> requestFeedbackSnapshot;
    std::vector<GsVulkanMemoryCase> requestMemoryCases;
    std::vector<GsVulkanCt32Sprite> requestSprites;
    std::vector<GsVulkanDepthCt32Sprite> requestDepthCt32Sprites;
    std::vector<GsVulkanNearestCt32Sprite> requestNearestCt32Sprites;
    std::vector<GsVulkanLinearCt32Sprite> requestLinearCt32Sprites;
    std::vector<GsVulkanFeedbackLinearDepthCt32Sprite>
        requestFeedbackLinearDepthCt32Sprites;
    std::vector<GsVulkanCt32Triangle> requestTriangles;
    std::vector<GsVulkanGouraudDepthCt32Triangle>
        requestGouraudDepthCt32Triangles;
    std::vector<GsVulkanT8GouraudDepthCt32Triangle>
        requestT8GouraudDepthCt32Triangles;
    std::vector<GsVulkanResidentT8GouraudDepthCt32Triangle>
        requestResidentT8GouraudDepthCt32Triangles;
    std::vector<GsVulkanT8Palette> requestT8Palettes;
    std::vector<GsVulkanFeedbackNearestDepthCt32Triangle>
        requestFeedbackNearestDepthCt32Triangles;
    GsVulkanFeedbackSnapshotMode requestFeedbackSnapshotMode =
        GsVulkanFeedbackSnapshotMode::UploadHost;
    // Page transfer calls are synchronous: the posting thread remains blocked
    // under callMutex until the owning Vulkan worker finishes consuming these
    // borrowed canonical-VRAM views.
    std::span<const uint8_t> requestPageUploadSource;
    std::span<uint8_t> requestPageDownloadDestination;
    GsVramPageMask requestPages;
    std::vector<uint8_t> responseOutput;
    std::vector<GsVulkanMemoryResult> responseResults;
    std::string responseError;
    GsVulkanRequestKind requestKind =
        GsVulkanRequestKind::RoundTrip;
    GsVulkanRequestKind activeRequestKind =
        GsVulkanRequestKind::RoundTrip;
    size_t activeRequestSpriteCount = 0u;
    size_t activeRequestDepthCt32SpriteCount = 0u;
    size_t activeRequestNearestCt32SpriteCount = 0u;
    size_t activeRequestLinearCt32SpriteCount = 0u;
    size_t activeRequestFeedbackLinearDepthCt32SpriteCount = 0u;
    size_t activeRequestTriangleCount = 0u;
    size_t activeRequestGouraudTriangleCount = 0u;
    size_t activeRequestT8TriangleCount = 0u;
    size_t activeRequestFeedbackNearestTriangleCount = 0u;
    bool activeRequestHasPageUpload = false;
    bool initialized = false;
    bool healthy = false;
    bool stopping = false;
    bool workerFinished = false;
    bool requestPending = false;
    bool requestInFlight = false;
    bool responseReady = false;
    bool responseSucceeded = false;
    std::atomic<uint64_t> requestWaits{0u};
    std::atomic<uint64_t> requestWaitTotalNanoseconds{0u};
    std::atomic<uint64_t> maximumRequestWaitNanoseconds{0u};
    std::array<
        std::atomic<uint64_t>,
        GS_VULKAN_REQUEST_KIND_COUNT> requestWaitsByKind{};
    std::array<
        std::atomic<uint64_t>,
        GS_VULKAN_REQUEST_KIND_COUNT> requestWaitNanosecondsByKind{};
    // Classification only needs a current fail-fast signal. Request posting
    // still rechecks the mutex-protected lifecycle state, so publishing this
    // mirror removes a per-draw mutex without weakening execution safety.
    std::atomic<bool> available{false};
};
#else
struct GsVulkanService::Impl final
{
};
#endif

GsVulkanService::GsVulkanService(std::unique_ptr<Impl> impl)
    : m_impl(std::move(impl))
{
}

GsVulkanService::~GsVulkanService()
{
    shutdown();
}

void GsVulkanService::shutdown() noexcept
{
#if PS2X_HAS_GS_VULKAN
    if (m_impl)
        m_impl->stopAndJoin();
#endif
}

std::unique_ptr<GsVulkanService> GsVulkanService::create(
    const GsVulkanServiceConfig &config,
    GsVulkanCapabilityReport *report,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    GsVulkanCapabilityReport unavailable =
        probeGsVulkanCapabilities(config.probe);
    if (report)
        *report = unavailable;
    if (error)
        *error = unavailable.detail;
    return nullptr;
#else
    auto impl = std::make_unique<Impl>(config);
    try
    {
        impl->worker = std::thread(&Impl::threadMain, impl.get());
    }
    catch (const std::exception &exception)
    {
        GsVulkanCapabilityReport failed{};
        failed.compiled = true;
        failed.status = GsVulkanProbeStatus::ResourceCreationFailed;
        failed.detail =
            std::string("failed to start Vulkan GS worker: ") +
            exception.what();
        if (report)
            *report = failed;
        if (error)
            *error = failed.detail;
        return nullptr;
    }

    GsVulkanCapabilityReport initializedCapabilities;
    std::string initializationError;
    bool initializedSuccessfully = false;
    {
        std::unique_lock lock(impl->stateMutex);
        impl->stateChanged.wait(lock, [&impl]
        {
            return impl->initialized;
        });
        initializedCapabilities = impl->capabilities;
        initializationError = impl->initializationError;
        initializedSuccessfully = impl->healthy;
    }

    if (report)
        *report = initializedCapabilities;
    if (error)
        *error = initializationError;
    if (!initializedSuccessfully)
    {
        impl->stopAndJoin();
        return nullptr;
    }
    return std::unique_ptr<GsVulkanService>(
        new GsVulkanService(std::move(impl)));
#endif
}

bool GsVulkanService::roundTripVram(
    std::span<const uint8_t> input,
    std::vector<uint8_t> &output,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)input;
    (void)output;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (input.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
            *error = "Vulkan VRAM round trip requires exactly 4 MiB";
        return false;
    }

    return m_impl->executeRequest(
        GsVulkanRequestKind::RoundTrip,
        std::vector<uint8_t>(input.begin(), input.end()),
        {}, {}, {}, {}, {}, {}, {}, {}, {},
        output, nullptr, error);
#endif
}

bool GsVulkanService::executeMemoryCases(
    std::span<const uint8_t> input,
    std::span<const GsVulkanMemoryCase> cases,
    std::vector<uint8_t> &output,
    std::vector<GsVulkanMemoryResult> &results,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)input;
    (void)cases;
    (void)output;
    (void)results;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (input.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
            *error =
                "Vulkan memory cases require exactly 4 MiB of VRAM";
        return false;
    }
    if (cases.empty() ||
        cases.size() > GS_VULKAN_MAX_MEMORY_CASES)
    {
        if (error)
        {
            *error = "Vulkan memory cases require between 1 and " +
                     std::to_string(GS_VULKAN_MAX_MEMORY_CASES) +
                     " records";
        }
        return false;
    }

    for (size_t index = 0u; index < cases.size(); ++index)
    {
        const GsVulkanMemoryCase &memoryCase = cases[index];
        const auto psm = static_cast<GSMem::PixelStorageMode>(
            memoryCase.pixelStorageMode);
        if (!GSMem::IsValidPsm(psm))
        {
            if (error)
            {
                *error = "Vulkan memory case " +
                         std::to_string(index) +
                         " has an unsupported pixel storage mode";
            }
            return false;
        }
        if (memoryCase.operation !=
                GsVulkanMemoryOperation::Read &&
            memoryCase.operation !=
                GsVulkanMemoryOperation::Write)
        {
            if (error)
            {
                *error = "Vulkan memory case " +
                         std::to_string(index) +
                         " has an unsupported operation";
            }
            return false;
        }
        if (memoryCase.reserved != 0u)
        {
            if (error)
            {
                *error = "Vulkan memory case " +
                         std::to_string(index) +
                         " has non-zero reserved data";
            }
            return false;
        }
    }

    return m_impl->executeRequest(
        GsVulkanRequestKind::MemoryCases,
        std::vector<uint8_t>(input.begin(), input.end()),
        {},
        std::vector<GsVulkanMemoryCase>(
            cases.begin(), cases.end()),
        {}, {}, {}, {}, {}, {}, {},
        output, &results, error);
#endif
}

bool GsVulkanService::executeCt32Sprite(
    std::span<const uint8_t> input,
    const GsVulkanCt32Sprite &sprite,
    std::vector<uint8_t> &output,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)input;
    (void)sprite;
    (void)output;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (input.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
            *error =
                "Vulkan CT32 sprite requires exactly 4 MiB of VRAM";
        return false;
    }
    if (const char *validationError =
            ct32SpriteValidationError(sprite))
    {
        if (error)
            *error = validationError;
        return false;
    }

    return m_impl->executeRequest(
        GsVulkanRequestKind::Ct32Sprite,
        std::vector<uint8_t>(input.begin(), input.end()),
        {}, {}, std::vector<GsVulkanCt32Sprite>{sprite},
        {}, {}, {}, {}, {}, {},
        output, nullptr, error);
#endif
}

bool GsVulkanService::executeDepthCt32Sprite(
    std::span<const uint8_t> input,
    const GsVulkanDepthCt32Sprite &sprite,
    std::vector<uint8_t> &output,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)input;
    (void)sprite;
    (void)output;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (input.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
        {
            *error =
                "Vulkan depth CT32 sprite requires exactly 4 MiB of VRAM";
        }
        return false;
    }
    if (const char *validationError =
            depthCt32SpriteValidationError(sprite))
    {
        if (error)
            *error = validationError;
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected || !selected->exactDepthCt32Sprite)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact depth CT32 sprites";
            }
            return false;
        }
    }

    return m_impl->executeRequest(
        GsVulkanRequestKind::DepthCt32Sprite,
        std::vector<uint8_t>(input.begin(), input.end()),
        {}, {}, {}, std::vector<GsVulkanDepthCt32Sprite>{sprite},
        {}, {}, {}, {}, {},
        output, nullptr, error);
#endif
}

bool GsVulkanService::executeNearestCt32Sprite(
    std::span<const uint8_t> input,
    const GsVulkanNearestCt32Sprite &sprite,
    std::vector<uint8_t> &output,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)input;
    (void)sprite;
    (void)output;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (input.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
        {
            *error =
                "Vulkan nearest CT32 sprite requires exactly 4 MiB of VRAM";
        }
        return false;
    }
    if (const char *validationError =
            nearestCt32SpriteValidationError(sprite))
    {
        if (error)
            *error = validationError;
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected || !selected->exactNearestCt32Sprite)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact nearest CT32 sprites";
            }
            return false;
        }
    }

    return m_impl->executeRequest(
        GsVulkanRequestKind::NearestCt32Sprite,
        std::vector<uint8_t>(input.begin(), input.end()),
        {}, {}, {}, {},
        std::vector<GsVulkanNearestCt32Sprite>{sprite},
        {}, {}, {}, {},
        output, nullptr, error);
#endif
}

bool GsVulkanService::executeLinearCt32Sprite(
    std::span<const uint8_t> input,
    const GsVulkanLinearCt32Sprite &sprite,
    std::vector<uint8_t> &output,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)input;
    (void)sprite;
    (void)output;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (input.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
        {
            *error =
                "Vulkan linear CT32 sprite requires exactly 4 MiB of VRAM";
        }
        return false;
    }
    if (const char *validationError =
            linearCt32SpriteValidationError(sprite))
    {
        if (error)
            *error = validationError;
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected || !selected->exactLinearCt32Sprite)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact linear CT32 sprites";
            }
            return false;
        }
    }

    return m_impl->executeRequest(
        GsVulkanRequestKind::LinearCt32Sprite,
        std::vector<uint8_t>(input.begin(), input.end()),
        {}, {}, {}, {}, {},
        std::vector<GsVulkanLinearCt32Sprite>{sprite},
        {}, {}, {},
        output, nullptr, error);
#endif
}

bool GsVulkanService::executeFeedbackLinearDepthCt32Sprite(
    std::span<const uint8_t> input,
    std::span<const uint8_t> feedbackSnapshot,
    const GsVulkanFeedbackLinearDepthCt32Sprite &sprite,
    std::vector<uint8_t> &output,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)input;
    (void)feedbackSnapshot;
    (void)sprite;
    (void)output;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (input.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
        {
            *error =
                "Vulkan feedback linear depth CT32 sprite requires exactly 4 MiB of canonical VRAM";
        }
        return false;
    }
    if (feedbackSnapshot.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
        {
            *error =
                "Vulkan feedback linear depth CT32 sprite requires exactly 4 MiB of snapshot VRAM";
        }
        return false;
    }
    if (const char *validationError =
            feedbackLinearDepthCt32SpriteValidationError(sprite))
    {
        if (error)
            *error = validationError;
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected ||
            !selected->exactFeedbackLinearDepthCt32Sprite)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact feedback linear depth CT32 sprites";
            }
            return false;
        }
    }

    return m_impl->executeRequest(
        GsVulkanRequestKind::FeedbackLinearDepthCt32Sprite,
        std::vector<uint8_t>(input.begin(), input.end()),
        std::vector<uint8_t>(
            feedbackSnapshot.begin(), feedbackSnapshot.end()),
        {}, {}, {}, {}, {},
        std::vector<GsVulkanFeedbackLinearDepthCt32Sprite>{sprite},
        {}, {},
        output, nullptr, error);
#endif
}

bool GsVulkanService::executeFeedbackNearestDepthCt32Triangle(
    std::span<const uint8_t> input,
    std::span<const uint8_t> feedbackSnapshot,
    const GsVulkanFeedbackNearestDepthCt32Triangle &triangle,
    std::vector<uint8_t> &output,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)input;
    (void)feedbackSnapshot;
    (void)triangle;
    (void)output;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (input.size() != GS_VULKAN_VRAM_SIZE ||
        feedbackSnapshot.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
        {
            *error =
                "Vulkan feedback nearest depth CT32 triangle requires exact 4 MiB canonical and snapshot images";
        }
        return false;
    }
    if (const char *validationError =
            feedbackNearestDepthCt32TriangleValidationError(triangle))
    {
        if (error)
            *error = validationError;
        return false;
    }
    GsVramPageMask allPages;
    allPages.setAll();
    if (!uploadVramPages(input, allPages, error) ||
        !executeResidentFeedbackNearestDepthCt32Triangles(
            feedbackSnapshot,
            GsVulkanFeedbackSnapshotMode::UploadHost,
            std::span<const GsVulkanFeedbackNearestDepthCt32Triangle>(
                &triangle, 1u),
            error))
    {
        return false;
    }
    std::vector<uint8_t> completed(GS_VULKAN_VRAM_SIZE);
    if (!downloadVramPages(completed, allPages, error))
        return false;
    output = std::move(completed);
    if (error)
        error->clear();
    return true;
#endif
}

bool GsVulkanService::executeCt32Triangle(
    std::span<const uint8_t> input,
    const GsVulkanCt32Triangle &triangle,
    std::vector<uint8_t> &output,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)input;
    (void)triangle;
    (void)output;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (input.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
        {
            *error =
                "Vulkan CT32 triangle requires exactly 4 MiB of VRAM";
        }
        return false;
    }
    if (const char *validationError =
            ct32TriangleValidationError(triangle))
    {
        if (error)
            *error = validationError;
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected || !selected->exactCt32Triangle)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact CT32 triangles";
            }
            return false;
        }
    }

    return m_impl->executeRequest(
        GsVulkanRequestKind::Ct32Triangle,
        std::vector<uint8_t>(input.begin(), input.end()),
        {}, {}, {}, {}, {}, {}, {},
        std::vector<GsVulkanCt32Triangle>{triangle}, {},
        output, nullptr, error);
#endif
}

bool GsVulkanService::executeGouraudDepthCt32Triangle(
    std::span<const uint8_t> input,
    const GsVulkanGouraudDepthCt32Triangle &triangle,
    std::vector<uint8_t> &output,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)input;
    (void)triangle;
    (void)output;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (input.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
        {
            *error =
                "Vulkan Gouraud depth CT32 triangle requires exactly 4 MiB of VRAM";
        }
        return false;
    }
    if (const char *validationError =
            gouraudDepthCt32TriangleValidationError(triangle))
    {
        if (error)
            *error = validationError;
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected ||
            !selected->exactGouraudDepthCt32Triangle)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact Gouraud depth CT32 triangles";
            }
            return false;
        }
    }

    return m_impl->executeRequest(
        GsVulkanRequestKind::GouraudDepthCt32Triangle,
        std::vector<uint8_t>(input.begin(), input.end()),
        {}, {}, {}, {}, {}, {}, {}, {},
        std::vector<GsVulkanGouraudDepthCt32Triangle>{triangle},
        {}, {}, output, nullptr, error);
#endif
}

bool GsVulkanService::executeT8GouraudDepthCt32Triangle(
    std::span<const uint8_t> input,
    const GsVulkanT8GouraudDepthCt32Triangle &triangle,
    std::vector<uint8_t> &output,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)input;
    (void)triangle;
    (void)output;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (input.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
        {
            *error =
                "Vulkan T8 Gouraud depth CT32 triangle requires exactly 4 MiB of VRAM";
        }
        return false;
    }
    if (const char *validationError =
            t8GouraudDepthCt32TriangleValidationError(triangle))
    {
        if (error)
            *error = validationError;
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected ||
            !selected->exactT8GouraudDepthCt32Triangle)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact T8 Gouraud depth CT32 triangles";
            }
            return false;
        }
    }

    return m_impl->executeRequest(
        GsVulkanRequestKind::T8GouraudDepthCt32Triangle,
        std::vector<uint8_t>(input.begin(), input.end()),
        {}, {}, {}, {}, {}, {}, {}, {}, {},
        std::vector<GsVulkanT8GouraudDepthCt32Triangle>{triangle},
        {}, output, nullptr, error);
#endif
}

bool GsVulkanService::uploadVramPages(
    std::span<const uint8_t> source,
    const GsVramPageMask &pages,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)source;
    (void)pages;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (source.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
            *error = "Vulkan page upload requires exactly 4 MiB of VRAM";
        return false;
    }
    if (!pages.any())
    {
        if (error)
            *error = "Vulkan page upload requires at least one page";
        return false;
    }

    std::vector<uint8_t> unusedOutput;
    return m_impl->executeRequest(
        GsVulkanRequestKind::UploadPages,
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, pages,
        unusedOutput, nullptr, error, {}, {}, source, {});
#endif
}

bool GsVulkanService::downloadVramPages(
    std::span<uint8_t> destination,
    const GsVramPageMask &pages,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)destination;
    (void)pages;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (destination.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
            *error = "Vulkan page download requires exactly 4 MiB of VRAM";
        return false;
    }
    if (!pages.any())
    {
        if (error)
            *error = "Vulkan page download requires at least one page";
        return false;
    }

    std::vector<uint8_t> unusedOutput;
    return m_impl->executeRequest(
        GsVulkanRequestKind::DownloadPages,
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, pages,
        unusedOutput, nullptr, error, {}, {}, {}, destination);
#endif
}

bool GsVulkanService::downloadFeedbackSnapshot(
    std::span<uint8_t> destination,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)destination;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (destination.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
        {
            *error =
                "Vulkan feedback snapshot download requires exactly 4 MiB";
        }
        return false;
    }

    std::vector<uint8_t> unusedOutput;
    return m_impl->executeRequest(
        GsVulkanRequestKind::DownloadFeedbackSnapshot,
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
        unusedOutput, nullptr, error, {}, {}, {}, destination);
#endif
}

bool GsVulkanService::executeResidentCt32Sprite(
    const GsVulkanCt32Sprite &sprite,
    std::string *error)
{
    return executeResidentCt32Sprites(
        std::span<const GsVulkanCt32Sprite>(&sprite, 1u), error);
}

bool GsVulkanService::executeResidentCt32Sprites(
    std::span<const GsVulkanCt32Sprite> sprites,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)sprites;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    std::string validationError;
    if (!validateResidentCt32SpriteBatch(sprites, validationError))
    {
        if (error)
            *error = std::move(validationError);
        return false;
    }

    std::vector<uint8_t> unusedOutput;
    return m_impl->executeRequest(
        GsVulkanRequestKind::ResidentCt32Sprites,
        {}, {}, {},
        std::vector<GsVulkanCt32Sprite>(
            sprites.begin(), sprites.end()),
        {}, {}, {}, {}, {}, {},
        unusedOutput, nullptr, error);
#endif
}

bool GsVulkanService::executeResidentNearestCt32Sprite(
    const GsVulkanNearestCt32Sprite &sprite,
    std::string *error)
{
    return executeResidentNearestCt32Sprites(
        std::span<const GsVulkanNearestCt32Sprite>(&sprite, 1u),
        error);
}

bool GsVulkanService::executeResidentNearestCt32Sprites(
    std::span<const GsVulkanNearestCt32Sprite> sprites,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)sprites;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    std::string validationError;
    if (!validateResidentNearestCt32SpriteBatch(
            sprites, validationError))
    {
        if (error)
            *error = std::move(validationError);
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected || !selected->exactNearestCt32Sprite)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact nearest CT32 sprites";
            }
            return false;
        }
    }

    std::vector<uint8_t> unusedOutput;
    return m_impl->executeRequest(
        GsVulkanRequestKind::ResidentNearestCt32Sprites,
        {}, {}, {}, {}, {},
        std::vector<GsVulkanNearestCt32Sprite>(
            sprites.begin(), sprites.end()),
        {}, {}, {}, {},
        unusedOutput, nullptr, error);
#endif
}

bool GsVulkanService::executeResidentLinearCt32Sprite(
    const GsVulkanLinearCt32Sprite &sprite,
    std::string *error)
{
    return executeResidentLinearCt32Sprites(
        std::span<const GsVulkanLinearCt32Sprite>(&sprite, 1u),
        error);
}

bool GsVulkanService::executeResidentDepthCt32Sprite(
    const GsVulkanDepthCt32Sprite &sprite,
    std::string *error)
{
    return executeResidentDepthCt32Sprites(
        std::span<const GsVulkanDepthCt32Sprite>(&sprite, 1u),
        error);
}

bool GsVulkanService::executeResidentDepthCt32Sprites(
    std::span<const GsVulkanDepthCt32Sprite> sprites,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)sprites;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    std::string validationError;
    if (!validateResidentDepthCt32SpriteBatch(sprites, validationError))
    {
        if (error)
            *error = std::move(validationError);
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected || !selected->exactDepthCt32Sprite)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact depth CT32 sprites";
            }
            return false;
        }
    }

    std::vector<uint8_t> unusedOutput;
    return m_impl->executeRequest(
        GsVulkanRequestKind::ResidentDepthCt32Sprites,
        {}, {}, {}, {},
        std::vector<GsVulkanDepthCt32Sprite>(
            sprites.begin(), sprites.end()),
        {}, {}, {}, {}, {},
        unusedOutput, nullptr, error);
#endif
}

bool GsVulkanService::executeResidentFeedbackLinearDepthCt32Sprite(
    std::span<const uint8_t> feedbackSnapshot,
    const GsVulkanFeedbackLinearDepthCt32Sprite &sprite,
    std::string *error)
{
    return executeResidentFeedbackLinearDepthCt32Sprites(
        feedbackSnapshot,
        std::span<const GsVulkanFeedbackLinearDepthCt32Sprite>(
            &sprite, 1u),
        error);
}

bool GsVulkanService::executeResidentFeedbackLinearDepthCt32Sprites(
    std::span<const uint8_t> feedbackSnapshot,
    std::span<const GsVulkanFeedbackLinearDepthCt32Sprite> sprites,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)feedbackSnapshot;
    (void)sprites;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    if (feedbackSnapshot.size() != GS_VULKAN_VRAM_SIZE)
    {
        if (error)
        {
            *error =
                "Vulkan resident feedback linear depth CT32 sprites require exactly 4 MiB of snapshot VRAM";
        }
        return false;
    }
    std::string validationError;
    if (!validateResidentFeedbackLinearDepthCt32SpriteBatch(
            sprites, validationError))
    {
        if (error)
            *error = std::move(validationError);
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected ||
            !selected->exactFeedbackLinearDepthCt32Sprite)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact feedback linear depth CT32 sprites";
            }
            return false;
        }
    }

    std::vector<uint8_t> unusedOutput;
    return m_impl->executeRequest(
        GsVulkanRequestKind::
            ResidentFeedbackLinearDepthCt32Sprites,
        {},
        std::vector<uint8_t>(
            feedbackSnapshot.begin(), feedbackSnapshot.end()),
        {}, {}, {}, {}, {},
        std::vector<GsVulkanFeedbackLinearDepthCt32Sprite>(
            sprites.begin(), sprites.end()),
        {}, {},
        unusedOutput, nullptr, error);
#endif
}

bool GsVulkanService::executeResidentFeedbackNearestDepthCt32Triangles(
    std::span<const uint8_t> feedbackSnapshot,
    GsVulkanFeedbackSnapshotMode snapshotMode,
    std::span<const GsVulkanFeedbackNearestDepthCt32Triangle> triangles,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)feedbackSnapshot;
    (void)snapshotMode;
    (void)triangles;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    const bool uploadsHostSnapshot =
        snapshotMode == GsVulkanFeedbackSnapshotMode::UploadHost;
    if ((uploadsHostSnapshot &&
         feedbackSnapshot.size() != GS_VULKAN_VRAM_SIZE) ||
        (!uploadsHostSnapshot && !feedbackSnapshot.empty()))
    {
        if (error)
        {
            *error =
                "Vulkan resident feedback nearest depth CT32 triangles received an invalid snapshot source";
        }
        return false;
    }
    std::string validationError;
    if (!validateResidentFeedbackNearestDepthCt32TriangleBatch(
            triangles, validationError))
    {
        if (error)
            *error = std::move(validationError);
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected ||
            !selected->exactFeedbackNearestDepthCt32Triangle)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact feedback nearest depth CT32 triangles";
            }
            return false;
        }
    }

    std::vector<uint8_t> unusedOutput;
    return m_impl->executeRequest(
        GsVulkanRequestKind::
            ResidentFeedbackNearestDepthCt32Triangles,
        {},
        std::vector<uint8_t>(
            feedbackSnapshot.begin(), feedbackSnapshot.end()),
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
        unusedOutput, nullptr, error,
        {}, {}, {}, {},
        std::vector<GsVulkanFeedbackNearestDepthCt32Triangle>(
            triangles.begin(), triangles.end()),
        snapshotMode);
#endif
}

bool GsVulkanService::executeResidentLinearCt32Sprites(
    std::span<const GsVulkanLinearCt32Sprite> sprites,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)sprites;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    std::string validationError;
    if (!validateResidentLinearCt32SpriteBatch(
            sprites, validationError))
    {
        if (error)
            *error = std::move(validationError);
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected || !selected->exactLinearCt32Sprite)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact linear CT32 sprites";
            }
            return false;
        }
    }

    std::vector<uint8_t> unusedOutput;
    return m_impl->executeRequest(
        GsVulkanRequestKind::ResidentLinearCt32Sprites,
        {}, {}, {}, {}, {}, {},
        std::vector<GsVulkanLinearCt32Sprite>(
            sprites.begin(), sprites.end()),
        {}, {}, {},
        unusedOutput, nullptr, error);
#endif
}

bool GsVulkanService::executeResidentCt32Triangle(
    const GsVulkanCt32Triangle &triangle,
    std::string *error)
{
    return executeResidentCt32Triangles(
        std::span<const GsVulkanCt32Triangle>(&triangle, 1u), error);
}

bool GsVulkanService::executeResidentCt32Triangles(
    std::span<const GsVulkanCt32Triangle> triangles,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)triangles;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    std::string validationError;
    if (!validateResidentCt32TriangleBatch(triangles, validationError))
    {
        if (error)
            *error = std::move(validationError);
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected || !selected->exactCt32Triangle)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact CT32 triangles";
            }
            return false;
        }
    }

    std::vector<uint8_t> unusedOutput;
    return m_impl->executeRequest(
        GsVulkanRequestKind::ResidentCt32Triangles,
        {}, {}, {}, {}, {}, {}, {}, {},
        std::vector<GsVulkanCt32Triangle>(
            triangles.begin(), triangles.end()),
        {}, unusedOutput, nullptr, error);
#endif
}

bool GsVulkanService::executeResidentGouraudDepthCt32Triangles(
    std::span<const GsVulkanGouraudDepthCt32Triangle> triangles,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)triangles;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    std::string validationError;
    if (!validateResidentGouraudDepthCt32TriangleBatch(
            triangles, validationError))
    {
        if (error)
            *error = std::move(validationError);
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected || !selected->exactGouraudDepthCt32Triangle)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact Gouraud depth CT32 triangles";
            }
            return false;
        }
    }

    std::vector<uint8_t> unusedOutput;
    return m_impl->executeRequest(
        GsVulkanRequestKind::ResidentGouraudDepthCt32Triangles,
        {}, {}, {}, {}, {}, {}, {}, {}, {},
        std::vector<GsVulkanGouraudDepthCt32Triangle>(
            triangles.begin(), triangles.end()),
        {}, {}, unusedOutput, nullptr, error);
#endif
}

bool GsVulkanService::executeResidentT8GouraudDepthCt32Triangles(
    std::span<const GsVulkanResidentT8GouraudDepthCt32Triangle> triangles,
    std::span<const GsVulkanT8Palette> palettes,
    std::string *error)
{
    return executeResidentT8GouraudDepthCt32TrianglesImpl(
        std::vector<GsVulkanResidentT8GouraudDepthCt32Triangle>(
            triangles.begin(), triangles.end()),
        std::vector<GsVulkanT8Palette>(
            palettes.begin(), palettes.end()),
        true, {}, {}, error);
}

bool GsVulkanService::
    executePreparedResidentT8GouraudDepthCt32Triangles(
        std::vector<GsVulkanResidentT8GouraudDepthCt32Triangle> triangles,
        std::vector<GsVulkanT8Palette> palettes,
        std::string *error)
{
    return executeResidentT8GouraudDepthCt32TrianglesImpl(
        std::move(triangles), std::move(palettes), false,
        {}, {}, error);
}

bool GsVulkanService::
    uploadVramPagesAndExecutePreparedResidentT8GouraudDepthCt32Triangles(
        std::span<const uint8_t> source,
        const GsVramPageMask &pages,
        std::vector<GsVulkanResidentT8GouraudDepthCt32Triangle> triangles,
        std::vector<GsVulkanT8Palette> palettes,
        std::string *error)
{
    if (source.size() != GS_VULKAN_VRAM_SIZE || !pages.any())
    {
        if (error)
        {
            *error =
                "invalid Vulkan GS resident T8 fused page upload";
        }
        return false;
    }
    return executeResidentT8GouraudDepthCt32TrianglesImpl(
        std::move(triangles), std::move(palettes), false,
        source, pages, error);
}

bool GsVulkanService::executeResidentT8GouraudDepthCt32TrianglesImpl(
    std::vector<GsVulkanResidentT8GouraudDepthCt32Triangle> triangles,
    std::vector<GsVulkanT8Palette> palettes,
    bool validateResourceViews,
    std::span<const uint8_t> pageUploadSource,
    const GsVramPageMask &pageUploadPages,
    std::string *error)
{
#if !PS2X_HAS_GS_VULKAN
    (void)triangles;
    (void)palettes;
    (void)validateResourceViews;
    (void)pageUploadSource;
    (void)pageUploadPages;
    if (error)
        *error = "Vulkan GS support was compiled out";
    return false;
#else
    std::string validationError;
    if (!validateResidentT8GouraudDepthCt32TriangleBatch(
            triangles, palettes, validateResourceViews,
            validateResourceViews,
            validationError))
    {
        if (error)
            *error = std::move(validationError);
        return false;
    }
    {
        std::lock_guard lock(m_impl->stateMutex);
        const GsVulkanDeviceReport *selected =
            m_impl->capabilities.selectedDevice();
        if (!selected ||
            !selected->exactT8GouraudDepthCt32Triangle)
        {
            if (error)
            {
                *error =
                    "Vulkan device does not support exact T8 Gouraud depth CT32 triangles";
            }
            return false;
        }
    }

    std::vector<uint8_t> unusedOutput;
    return m_impl->executeRequest(
        GsVulkanRequestKind::ResidentT8GouraudDepthCt32Triangles,
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
        {}, pageUploadPages, unusedOutput, nullptr, error,
        std::move(triangles), std::move(palettes),
        pageUploadSource);
#endif
}

GsVulkanCapabilityReport GsVulkanService::capabilities() const
{
#if !PS2X_HAS_GS_VULKAN
    return probeGsVulkanCapabilities();
#else
    std::lock_guard lock(m_impl->stateMutex);
    return m_impl->capabilities;
#endif
}

GsVulkanServiceStatistics GsVulkanService::statistics() const
{
#if !PS2X_HAS_GS_VULKAN
    return {};
#else
    std::lock_guard lock(m_impl->stateMutex);
    GsVulkanServiceStatistics statistics = m_impl->statistics;
    statistics.requestWaits = m_impl->requestWaits.load(
        std::memory_order_relaxed);
    statistics.requestWaitNanoseconds =
        m_impl->requestWaitTotalNanoseconds.load(
            std::memory_order_relaxed);
    statistics.maximumRequestWaitNanoseconds =
        m_impl->maximumRequestWaitNanoseconds.load(
            std::memory_order_relaxed);
    for (size_t index = 0u;
         index < GS_VULKAN_REQUEST_KIND_COUNT; ++index)
    {
        statistics.requestsByKind[index].requestWaits =
            m_impl->requestWaitsByKind[index].load(
                std::memory_order_relaxed);
        statistics.requestsByKind[index].requestWaitNanoseconds =
            m_impl->requestWaitNanosecondsByKind[index].load(
                std::memory_order_relaxed);
    }
    return statistics;
#endif
}

bool GsVulkanService::healthy() const
{
#if !PS2X_HAS_GS_VULKAN
    return false;
#else
    return m_impl->available.load(std::memory_order_acquire);
#endif
}
