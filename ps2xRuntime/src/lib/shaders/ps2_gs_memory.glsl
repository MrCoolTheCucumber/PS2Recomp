// Exact integer GS-local-memory addressing shared by Vulkan compute shaders.
// The compact block/column formulation follows the GS page/block/column
// layout; exhaustive GPU-vs-CPU tests cover every supported PSM.

const uint GS_MEMORY_SIZE_BYTES = 4u * 1024u * 1024u;
const uint GS_MEMORY_WORD_COUNT = GS_MEMORY_SIZE_BYTES / 4u;
const uint GS_MEMORY_PAGE_BYTES = 8192u;
const uint GS_MEMORY_BLOCK_BYTES = 256u;
const uint GS_MEMORY_COLUMN_BYTES = 64u;

const uint GS_LAYOUT_C32 = 0u;
const uint GS_LAYOUT_Z32 = 1u;
const uint GS_LAYOUT_C16 = 2u;
const uint GS_LAYOUT_C16S = 3u;
const uint GS_LAYOUT_Z16 = 4u;
const uint GS_LAYOUT_Z16S = 5u;
const uint GS_LAYOUT_P8 = 6u;
const uint GS_LAYOUT_P4 = 7u;
const uint GS_LAYOUT_INVALID = 0xffffffffu;

const uint gs_block_c32[32] = uint[32](
     0u,  1u,  4u,  5u, 16u, 17u, 20u, 21u,
     2u,  3u,  6u,  7u, 18u, 19u, 22u, 23u,
     8u,  9u, 12u, 13u, 24u, 25u, 28u, 29u,
    10u, 11u, 14u, 15u, 26u, 27u, 30u, 31u);

const uint gs_block_z32[32] = uint[32](
    24u, 25u, 28u, 29u,  8u,  9u, 12u, 13u,
    26u, 27u, 30u, 31u, 10u, 11u, 14u, 15u,
    16u, 17u, 20u, 21u,  0u,  1u,  4u,  5u,
    18u, 19u, 22u, 23u,  2u,  3u,  6u,  7u);

const uint gs_block_c16[32] = uint[32](
     0u,  2u,  8u, 10u,
     1u,  3u,  9u, 11u,
     4u,  6u, 12u, 14u,
     5u,  7u, 13u, 15u,
    16u, 18u, 24u, 26u,
    17u, 19u, 25u, 27u,
    20u, 22u, 28u, 30u,
    21u, 23u, 29u, 31u);

const uint gs_block_c16s[32] = uint[32](
     0u,  2u, 16u, 18u,
     1u,  3u, 17u, 19u,
     8u, 10u, 24u, 26u,
     9u, 11u, 25u, 27u,
     4u,  6u, 20u, 22u,
     5u,  7u, 21u, 23u,
    12u, 14u, 28u, 30u,
    13u, 15u, 29u, 31u);

const uint gs_block_z16[32] = uint[32](
    24u, 26u, 16u, 18u,
    25u, 27u, 17u, 19u,
    28u, 30u, 20u, 22u,
    29u, 31u, 21u, 23u,
     8u, 10u,  0u,  2u,
     9u, 11u,  1u,  3u,
    12u, 14u,  4u,  6u,
    13u, 15u,  5u,  7u);

const uint gs_block_z16s[32] = uint[32](
    24u, 26u,  8u, 10u,
    25u, 27u,  9u, 11u,
    16u, 18u,  0u,  2u,
    17u, 19u,  1u,  3u,
    28u, 30u, 12u, 14u,
    29u, 31u, 13u, 15u,
    20u, 22u,  4u,  6u,
    21u, 23u,  5u,  7u);

const uint gs_column32[16] = uint[16](
     0u,  1u,  4u,  5u,  8u,  9u, 12u, 13u,
     2u,  3u,  6u,  7u, 10u, 11u, 14u, 15u);

const uint gs_column16[32] = uint[32](
     0u,  2u,  8u, 10u, 16u, 18u, 24u, 26u,
     1u,  3u,  9u, 11u, 17u, 19u, 25u, 27u,
     4u,  6u, 12u, 14u, 20u, 22u, 28u, 30u,
     5u,  7u, 13u, 15u, 21u, 23u, 29u, 31u);

// P8 and P4 use the same pair of 2x8 word tables.
const uint gs_column_word[32] = uint[32](
     0u,  1u,  4u,  5u,  8u,  9u, 12u, 13u,
     2u,  3u,  6u,  7u, 10u, 11u, 14u, 15u,
     8u,  9u, 12u, 13u,  0u,  1u,  4u,  5u,
    10u, 11u, 14u, 15u,  2u,  3u,  6u,  7u);

uint gs_layout_for_psm(uint psm)
{
    switch (psm)
    {
    case 0x00u: // PSMCT32
    case 0x01u: // PSMCT24
    case 0x1bu: // PSMT8H
    case 0x24u: // PSMT4HL
    case 0x2cu: // PSMT4HH
        return GS_LAYOUT_C32;
    case 0x30u: // PSMZ32
    case 0x31u: // PSMZ24
        return GS_LAYOUT_Z32;
    case 0x02u: // PSMCT16
        return GS_LAYOUT_C16;
    case 0x0au: // PSMCT16S
        return GS_LAYOUT_C16S;
    case 0x32u: // PSMZ16
        return GS_LAYOUT_Z16;
    case 0x3au: // PSMZ16S
        return GS_LAYOUT_Z16S;
    case 0x13u: // PSMT8
        return GS_LAYOUT_P8;
    case 0x14u: // PSMT4
        return GS_LAYOUT_P4;
    default:
        return GS_LAYOUT_INVALID;
    }
}

uint gs_packed_bit_width(uint psm)
{
    switch (psm)
    {
    case 0x00u:
    case 0x30u:
        return 32u;
    case 0x01u:
    case 0x31u:
        return 24u;
    case 0x02u:
    case 0x0au:
    case 0x32u:
    case 0x3au:
        return 16u;
    case 0x13u:
    case 0x1bu:
        return 8u;
    case 0x14u:
    case 0x24u:
    case 0x2cu:
        return 4u;
    default:
        return 0u;
    }
}

uint gs_block_id(uint layout_id, uint x, uint y)
{
    if (layout_id == GS_LAYOUT_C32 || layout_id == GS_LAYOUT_Z32)
    {
        uint index = ((y / 8u) % 4u) * 8u + ((x / 8u) % 8u);
        return layout_id == GS_LAYOUT_Z32
            ? gs_block_z32[index]
            : gs_block_c32[index];
    }
    if (layout_id == GS_LAYOUT_P8)
    {
        uint index = ((y / 16u) % 4u) * 8u + ((x / 16u) % 8u);
        return gs_block_c32[index];
    }

    uint block_width = layout_id == GS_LAYOUT_P4 ? 32u : 16u;
    uint block_height = layout_id == GS_LAYOUT_P4 ? 16u : 8u;
    uint index = ((y / block_height) % 8u) * 4u +
                 ((x / block_width) % 4u);
    if (layout_id == GS_LAYOUT_C16S)
        return gs_block_c16s[index];
    if (layout_id == GS_LAYOUT_Z16)
        return gs_block_z16[index];
    if (layout_id == GS_LAYOUT_Z16S)
        return gs_block_z16s[index];
    return gs_block_c16[index];
}

uint gs_page_base_byte(uint layout_id, uint bp, uint bw, uint x, uint y)
{
    uint page_width = layout_id >= GS_LAYOUT_P8 ? 128u : 64u;
    uint page_height = 32u;
    if (layout_id >= GS_LAYOUT_C16 && layout_id <= GS_LAYOUT_Z16S)
        page_height = 64u;
    else if (layout_id == GS_LAYOUT_P8)
        page_height = 64u;
    else if (layout_id == GS_LAYOUT_P4)
        page_height = 128u;

    uint pages_per_row = (bw * 64u) / page_width;
    uint page = bp / 32u + x / page_width +
                (y / page_height) * pages_per_row;
    uint block = bp % 32u + gs_block_id(layout_id, x, y);
    return page * GS_MEMORY_PAGE_BYTES + block * GS_MEMORY_BLOCK_BYTES;
}

uint gs_pixel_bit_address(uint psm, uint bp, uint bw, uint x, uint y)
{
    uint layout_id = gs_layout_for_psm(psm);
    if (layout_id == GS_LAYOUT_INVALID)
        return 0xffffffffu;

    uint base_byte = gs_page_base_byte(layout_id, bp, bw, x, y);
    if (layout_id == GS_LAYOUT_C32 || layout_id == GS_LAYOUT_Z32)
    {
        uint local_x = x % 8u;
        uint local_y = y % 8u;
        uint column = local_y / 2u;
        uint word = gs_column32[(local_y % 2u) * 8u + local_x];
        uint bit_offset = 0u;
        if (psm == 0x1bu || psm == 0x24u)
            bit_offset = 24u;
        else if (psm == 0x2cu)
            bit_offset = 28u;
        return (base_byte + column * GS_MEMORY_COLUMN_BYTES + word * 4u) *
                   8u +
               bit_offset;
    }
    if (layout_id >= GS_LAYOUT_C16 && layout_id <= GS_LAYOUT_Z16S)
    {
        uint local_x = x % 16u;
        uint local_y = y % 8u;
        uint column = local_y / 2u;
        uint half_index = gs_column16[(local_y % 2u) * 16u + local_x];
        return (base_byte + column * GS_MEMORY_COLUMN_BYTES +
                half_index * 2u) * 8u;
    }
    if (layout_id == GS_LAYOUT_P8)
    {
        uint local_x = x % 16u;
        uint local_y = y % 16u;
        uint column = local_y / 4u;
        local_y %= 4u;
        uint table = ((local_y & 2u) >> 1u) ^ (column & 1u);
        uint byte_index = ((local_x & 8u) >> 2u) +
                          ((local_y & 2u) >> 1u);
        uint word = gs_column_word[
            table * 16u + (local_y & 1u) * 8u + (local_x & 7u)];
        return (base_byte + column * GS_MEMORY_COLUMN_BYTES +
                word * 4u + byte_index) * 8u;
    }

    uint local_x = x % 32u;
    uint local_y = y % 16u;
    uint column = local_y / 4u;
    local_y %= 4u;
    uint bit_shift = (local_x & 24u) + ((local_y & 2u) << 1u);
    uint table = ((local_y & 2u) >> 1u) ^ (column & 1u);
    uint word = gs_column_word[
        table * 16u + (local_y & 1u) * 8u + (local_x & 7u)];
    return (base_byte + column * GS_MEMORY_COLUMN_BYTES + word * 4u) * 8u +
           bit_shift;
}

uint gs_value_mask(uint bit_width)
{
    if (bit_width == 32u)
        return 0xffffffffu;
    return (1u << bit_width) - 1u;
}

uint gs_extract_value(uint word, uint bit_shift, uint bit_width)
{
    return (word >> bit_shift) & gs_value_mask(bit_width);
}
