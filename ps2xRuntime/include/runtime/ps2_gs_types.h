#pragma once

#include <cstdint>

enum GSPrimType : uint8_t
{
    GS_PRIM_POINT = 0,
    GS_PRIM_LINE = 1,
    GS_PRIM_LINESTRIP = 2,
    GS_PRIM_TRIANGLE = 3,
    GS_PRIM_TRISTRIP = 4,
    GS_PRIM_TRIFAN = 5,
    GS_PRIM_SPRITE = 6,
};

enum GSPsm : uint8_t
{
    GS_PSM_CT32 = 0,
    GS_PSM_CT24 = 1,
    GS_PSM_CT16 = 2,
    GS_PSM_CT16S = 10,
    GS_PSM_T8 = 19,
    GS_PSM_T4 = 20,
    GS_PSM_T8H = 27,
    GS_PSM_T4HL = 36,
    GS_PSM_T4HH = 44,
    GS_PSM_Z32 = 48,
    GS_PSM_Z24 = 49,
    GS_PSM_Z16 = 50,
    GS_PSM_Z16S = 58,
};

enum GSGifFormat : uint8_t
{
    GIF_FMT_PACKED = 0,
    GIF_FMT_REGLIST = 1,
    GIF_FMT_IMAGE = 2,
    GIF_FMT_IMAGE2 = 3,
};

enum GSRegId : uint8_t
{
    GS_REG_PRIM = 0x00,
    GS_REG_RGBAQ = 0x01,
    GS_REG_ST = 0x02,
    GS_REG_UV = 0x03,
    GS_REG_XYZF2 = 0x04,
    GS_REG_XYZ2 = 0x05,
    GS_REG_TEX0_1 = 0x06,
    GS_REG_TEX0_2 = 0x07,
    GS_REG_CLAMP_1 = 0x08,
    GS_REG_CLAMP_2 = 0x09,
    GS_REG_FOG = 0x0A,
    GS_REG_XYZF3 = 0x0C,
    GS_REG_XYZ3 = 0x0D,
    GS_REG_AD = 0x0F,

    GS_REG_TEX1_1 = 0x14,
    GS_REG_TEX1_2 = 0x15,
    GS_REG_TEX2_1 = 0x16,
    GS_REG_TEX2_2 = 0x17,
    GS_REG_XYOFFSET_1 = 0x18,
    GS_REG_XYOFFSET_2 = 0x19,
    GS_REG_PRMODECONT = 0x1A,
    GS_REG_PRMODE = 0x1B,
    GS_REG_TEXCLUT = 0x1C,
    GS_REG_SCANMSK = 0x22,
    GS_REG_MIPTBP1_1 = 0x34,
    GS_REG_MIPTBP1_2 = 0x35,
    GS_REG_MIPTBP2_1 = 0x36,
    GS_REG_MIPTBP2_2 = 0x37,
    GS_REG_TEXA = 0x3B,
    GS_REG_FOGCOL = 0x3D,
    GS_REG_TEXFLUSH = 0x3F,
    GS_REG_SCISSOR_1 = 0x40,
    GS_REG_SCISSOR_2 = 0x41,
    GS_REG_ALPHA_1 = 0x42,
    GS_REG_ALPHA_2 = 0x43,
    GS_REG_DIMX = 0x44,
    GS_REG_DTHE = 0x45,
    GS_REG_COLCLAMP = 0x46,
    GS_REG_TEST_1 = 0x47,
    GS_REG_TEST_2 = 0x48,
    GS_REG_PABE = 0x49,
    GS_REG_FBA_1 = 0x4A,
    GS_REG_FBA_2 = 0x4B,
    GS_REG_FRAME_1 = 0x4C,
    GS_REG_FRAME_2 = 0x4D,
    GS_REG_ZBUF_1 = 0x4E,
    GS_REG_ZBUF_2 = 0x4F,
    GS_REG_BITBLTBUF = 0x50,
    GS_REG_TRXPOS = 0x51,
    GS_REG_TRXREG = 0x52,
    GS_REG_TRXDIR = 0x53,
    GS_REG_HWREG = 0x54,
    GS_REG_SIGNAL = 0x60,
    GS_REG_FINISH = 0x61,
    GS_REG_LABEL = 0x62,
};

struct GSVertex
{
    float x = 0.0f;
    float y = 0.0f;
    // Kept as double because float is not accurate enough near UINT32_MAX.
    double z = 0.0;
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
    uint8_t a = 0;
    float q = 0.0f;
    float s = 0.0f;
    float t = 0.0f;
    uint16_t u = 0;
    uint16_t v = 0;
    uint8_t fog = 0;

    // Preserve the integer GIF payload alongside the compatibility values
    // above. Backends must not reconstruct 12.4 positions or integer Z from
    // host floating-point values.
    uint16_t x12_4 = 0;
    uint16_t y12_4 = 0;
    uint32_t zInteger = 0;
};

struct GSFrameReg
{
    uint32_t fbp;
    uint32_t fbw;
    uint8_t psm;
    uint32_t fbmsk;
};

struct GSZbufReg
{
    uint32_t zbp;
    uint8_t psm;
    bool zmask;
};

struct GSScissorReg
{
    uint16_t x0, x1, y0, y1;
};

struct GSTex0Reg
{
    uint32_t tbp0;
    uint8_t tbw;
    uint8_t psm;
    uint8_t tw;
    uint8_t th;
    uint8_t tcc;
    uint8_t tfx;
    uint32_t cbp;
    uint8_t cpsm;
    uint8_t csm;
    uint8_t csa;
    uint8_t cld;
};

struct GSXYOffsetReg
{
    uint16_t ofx;
    uint16_t ofy;
};

struct GSTexaReg
{
    uint8_t ta0;
    bool aem;
    uint8_t ta1;
};

struct GSTexClutReg
{
    uint8_t cbw;
    uint8_t cou;
    uint16_t cov;
};

struct GSContext
{
    GSFrameReg frame;
    GSScissorReg scissor;
    GSTex0Reg tex0;
    GSXYOffsetReg xyoffset;
    GSZbufReg zbuf;
    uint64_t tex1;
    uint64_t miptbp1;
    uint64_t miptbp2;
    uint64_t clamp;
    uint64_t alpha;
    uint64_t test;
    uint64_t fba;
};

struct GSPrimReg
{
    GSPrimType type;
    bool iip;
    bool tme;
    bool fge;
    bool abe;
    bool aa1;
    bool fst;
    bool ctxt;
    bool fix;
};

struct GSBitBltBuf
{
    uint32_t sbp;
    uint8_t sbw;
    uint8_t spsm;
    uint32_t dbp;
    uint8_t dbw;
    uint8_t dpsm;
};

struct GSTrxPos
{
    uint16_t ssax, ssay;
    uint16_t dsax, dsay;
    uint8_t dir;
};

struct GSTrxReg
{
    uint16_t rrw, rrh;
};
