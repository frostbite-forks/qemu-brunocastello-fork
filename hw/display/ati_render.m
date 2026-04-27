/*
 * QEMU ATI Rage 128 — inline Metal 3D renderer
 *
 * Parses PM4 command packets (already byte-swapped to host LE by ati_3d.c)
 * and executes them on the host Metal GPU.  Rendered pixels are written
 * directly into s->vga.vram_ptr so the existing VGA display path shows them.
 *
 * Vertex format (our canonical definition, used by both this file and the
 * ati-vga-ndrv NDRV compiled in OS 9):
 *
 *   XY position : always present, 2 × float32 (screen pixels, top-left origin)
 *   Z depth     : optional (VFMT_Z,    bit 0), 1 × float32
 *   W           : optional (VFMT_W,    bit 1), 1 × float32  [ignored]
 *   ARGB colour : optional (VFMT_COLOR,bit 3), 1 × uint32   A[31:24] R[23:16] G[15:8] B[7:0]
 *   Specular    : optional (VFMT_SPEC, bit 4), 1 × uint32   [ignored]
 *   Fog         : optional (VFMT_FOG,  bit 5), 1 × float32  [ignored]
 *   S0 T0       : optional (VFMT_ST0,  bit 6), 2 × float32  [ignored for now]
 *   S1 T1       : optional (VFMT_ST1,  bit 7), 2 × float32  [ignored]
 *
 * PM4 packets (Rage 128 / R128 encoding):
 *   Type 0  hdr[31:30]=00  hdr[29:16]=cnt  hdr[8:0]=reg>>2
 *           cnt+1 dwords follow, written to consecutive registers
 *   Type 3  hdr[31:30]=11  hdr[29:16]=cnt  hdr[15:8]=opcode
 *           cnt dwords follow
 *
 * 3D_DRAW_IMMD (opcode 0x29) payload:
 *   dw[0]  vertex format flags (VFMT_*)
 *   dw[1]  (nvert << 16) | walk_type | prim_type
 *   dw[2…] nvert × stride dwords of vertex data
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ati_render.h"
#include "ati_regs.h"
#include <string.h>
#include <stdlib.h>

/* ---------- PM4 packet macros -------------------------------------------- */

#define PM4_TYPE(h)       (((h) >> 30) & 0x3)
#define PM4_PKT0_REG(h)   (((h) & 0x1ffu) << 2)
#define PM4_PKT0_CNT(h)   (((h) >> 16) & 0x3fffu)
#define PM4_PKT3_OP(h)    (((h) >> 8)  & 0xffu)
#define PM4_PKT3_CNT(h)   (((h) >> 16) & 0x3fffu)

/* Type-3 opcodes we care about */
#define R128_OP_NOP           0x10u
#define R128_OP_WAIT_IDLE     0x14u
#define R128_OP_3D_DRAW_VBUF  0x28u
#define R128_OP_3D_DRAW_IMMD  0x29u
#define R128_OP_3D_DRAW_INDX  0x2Au

/* Custom ATI3D passthrough opcodes */
#define ATI_OP_TEXTURE_UPLOAD 0x30u  /* slot,w,h,stride,vram_offset */
#define ATI_OP_SET_TEXTURE    0x31u  /* slot */

#define ATI_TEX_SLOTS         16u

/* Vertex format bits */
#define VFMT_Z     (1u << 0)
#define VFMT_W     (1u << 1)
#define VFMT_COLOR (1u << 3)
#define VFMT_SPEC  (1u << 4)
#define VFMT_FOG   (1u << 5)
#define VFMT_ST0   (1u << 6)
#define VFMT_ST1   (1u << 7)

/* Primitive type field in dw[1] bits[3:0] */
#define PRIM_POINTS     0x1u
#define PRIM_LINES      0x2u
#define PRIM_POLY_LINE  0x3u
#define PRIM_TRI_LIST   0x4u
#define PRIM_TRI_FAN    0x5u
#define PRIM_TRI_STRIP  0x6u

/* ---------- internal types ------------------------------------------------ */

#define MAX_VERTS  65536u
#define MAX_DRAWS  1024u

typedef struct {
    float x, y, z;    /* screen XY + depth */
    float r, g, b, a;
    float u, v;
} AtiVertex;  /* 36 bytes — must match Metal shader struct */

typedef struct {
    uint32_t           vert_start;
    uint32_t           vert_count;
    MTLPrimitiveType   prim;
    uint32_t           tex_slot;
    bool               has_tex;
} DrawCmd;

struct ATIRenderState {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLRenderPipelineState> pipeline;       /* vertex color */
    id<MTLRenderPipelineState> pipeline_tex;   /* texture * vertex color */
    id<MTLDepthStencilState>   depth_state;
    id<MTLTexture>             fb_tex;
    id<MTLTexture>             depth_tex;
    id<MTLTexture>             tex_pool[ATI_TEX_SLOTS];
    id<MTLBuffer>              vbuf;   /* MAX_VERTS × sizeof(AtiVertex) */

    uint8_t  *vram_ptr;
    uint32_t  vram_size;
    uint32_t  fb_width;
    uint32_t  fb_height;
    uint32_t  fb_stride;    /* Metal texture row stride: fb_width * 4 (always BGRA8) */
    uint32_t  vram_stride;  /* guest VRAM row stride: fb_width * bypp */
    uint32_t  fb_bpp;       /* guest color depth: 15, 16, or 32 */
    uint32_t *conv_buf;   /* scratch buffer for BGRA<->guest conversion */
    uint32_t  active_tex; /* last slot set by ATI_OP_SET_TEXTURE */

    /* 3D state updated by PM4 type-0 register writes */
    uint32_t  vc_fpu_setup;
};

/* ---------- Metal shader source ------------------------------------------- */

static NSString *kShaderSrc = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct AtiVertex {\n"
"    float x, y, z;\n"       /* 12 bytes — separate floats, no float3 padding */
"    float r, g, b, a;\n"    /* 16 bytes — matches 36-byte C struct exactly */
"    float u, v;\n"           /* 8 bytes */
"};\n"
"\n"
"struct VOut {\n"
"    float4 pos [[position]];\n"
"    float4 col;\n"
"    float2 uv;\n"
"};\n"
"\n"
"vertex VOut vert_main(\n"
"    device const AtiVertex *v [[buffer(0)]],\n"
"    uint vid [[vertex_id]],\n"
"    constant float2 &fbsz [[buffer(1)]])\n"
"{\n"
"    VOut out;\n"
"    out.pos.x =  2.0 * v[vid].x / fbsz.x - 1.0;\n"
"    out.pos.y = -2.0 * v[vid].y / fbsz.y + 1.0;\n"
"    out.pos.z = v[vid].z;\n"
"    out.pos.w = 1.0;\n"
"    out.col   = float4(v[vid].r, v[vid].g, v[vid].b, v[vid].a);\n"
"    out.uv    = float2(v[vid].u, v[vid].v);\n"
"    return out;\n"
"}\n"
"\n"
"fragment float4 frag_main(VOut in [[stage_in]]) {\n"
"    return in.col;\n"
"}\n"
"\n"
"fragment float4 frag_tex(VOut in [[stage_in]],\n"
"    texture2d<float> tex [[texture(0)]])\n"
"{\n"
"    constexpr sampler smp(filter::linear, address::repeat);\n"
"    return tex.sample(smp, in.uv) * in.col;\n"
"}\n";

/* ---------- helpers -------------------------------------------------------- */

static id<MTLRenderPipelineState> make_pipeline(id<MTLDevice> dev,
                                                 id<MTLLibrary> lib)
{
    NSError *err = nil;
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction   = [lib newFunctionWithName:@"vert_main"];
    d.fragmentFunction = [lib newFunctionWithName:@"frag_main"];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    d.depthAttachmentPixelFormat      = MTLPixelFormatDepth32Float;
    d.colorAttachments[0].blendingEnabled             = YES;
    d.colorAttachments[0].rgbBlendOperation           = MTLBlendOperationAdd;
    d.colorAttachments[0].alphaBlendOperation         = MTLBlendOperationAdd;
    d.colorAttachments[0].sourceRGBBlendFactor        = MTLBlendFactorSourceAlpha;
    d.colorAttachments[0].destinationRGBBlendFactor   = MTLBlendFactorOneMinusSourceAlpha;
    d.colorAttachments[0].sourceAlphaBlendFactor      = MTLBlendFactorOne;
    d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    id<MTLRenderPipelineState> ps =
        [dev newRenderPipelineStateWithDescriptor:d error:&err];
    if (!ps) { NSLog(@"ati_render: pipeline error: %@", err); }
    return ps;
}

static id<MTLRenderPipelineState> make_pipeline_tex(id<MTLDevice> dev,
                                                     id<MTLLibrary> lib)
{
    NSError *err = nil;
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction   = [lib newFunctionWithName:@"vert_main"];
    d.fragmentFunction = [lib newFunctionWithName:@"frag_tex"];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    d.depthAttachmentPixelFormat      = MTLPixelFormatDepth32Float;
    d.colorAttachments[0].blendingEnabled             = YES;
    d.colorAttachments[0].rgbBlendOperation           = MTLBlendOperationAdd;
    d.colorAttachments[0].alphaBlendOperation         = MTLBlendOperationAdd;
    d.colorAttachments[0].sourceRGBBlendFactor        = MTLBlendFactorSourceAlpha;
    d.colorAttachments[0].destinationRGBBlendFactor   = MTLBlendFactorOneMinusSourceAlpha;
    d.colorAttachments[0].sourceAlphaBlendFactor      = MTLBlendFactorOne;
    d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    id<MTLRenderPipelineState> ps =
        [dev newRenderPipelineStateWithDescriptor:d error:&err];
    if (!ps) { NSLog(@"ati_render: pipeline_tex error: %@", err); }
    return ps;
}

static id<MTLTexture> make_fb_texture(id<MTLDevice> dev, uint32_t w, uint32_t h)
{
    MTLTextureDescriptor *td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                          width:w
                                                         height:h
                                                      mipmapped:NO];
    td.usage       = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;  /* CPU-readable for readback */
    return [dev newTextureWithDescriptor:td];
}

static id<MTLTexture> make_depth_texture(id<MTLDevice> dev, uint32_t w, uint32_t h)
{
    MTLTextureDescriptor *td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                          width:w
                                                         height:h
                                                      mipmapped:NO];
    td.usage       = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModePrivate;  /* GPU-only, no CPU readback needed */
    return [dev newTextureWithDescriptor:td];
}

static id<MTLDepthStencilState> make_depth_stencil_state(id<MTLDevice> dev)
{
    MTLDepthStencilDescriptor *d = [MTLDepthStencilDescriptor new];
    d.depthCompareFunction = MTLCompareFunctionLessEqual;
    d.depthWriteEnabled    = YES;
    return [dev newDepthStencilStateWithDescriptor:d];
}

/* ---------- public API ---------------------------------------------------- */

ATIRenderState *ati_metal_init(uint8_t *vram_ptr, uint32_t vram_size)
{
    ATIRenderState *rs = (ATIRenderState *)calloc(1, sizeof(*rs));
    if (!rs) return NULL;

    rs->device = MTLCreateSystemDefaultDevice();
    if (!rs->device) {
        NSLog(@"ati_render: no Metal device");
        free(rs);
        return NULL;
    }

    rs->queue = [rs->device newCommandQueue];

    {
        NSError *err = nil;
        id<MTLLibrary> lib = [rs->device newLibraryWithSource:kShaderSrc
                                                       options:nil
                                                         error:&err];
        if (!lib) { NSLog(@"ati_render: shader error: %@", err); free(rs); return NULL; }
        rs->pipeline     = make_pipeline(rs->device, lib);
        rs->pipeline_tex = make_pipeline_tex(rs->device, lib);
    }
    if (!rs->pipeline || !rs->pipeline_tex) { free(rs); return NULL; }
    rs->depth_state = make_depth_stencil_state(rs->device);

    rs->vbuf = [rs->device newBufferWithLength:MAX_VERTS * sizeof(AtiVertex)
                                       options:MTLResourceStorageModeShared];

    rs->vram_ptr  = vram_ptr;
    rs->vram_size = vram_size;
    rs->conv_buf  = NULL;

    NSLog(@"ati_render: Metal renderer ready (%s)",
          [[rs->device name] UTF8String]);
    return rs;
}

void ati_metal_set_fb(ATIRenderState *rs,
                      uint32_t width, uint32_t height, uint32_t stride,
                      uint32_t bpp)
{
    if (!rs || !width || !height) return;
    if (!bpp) bpp = 32;
    if (rs->fb_width == width && rs->fb_height == height &&
        rs->fb_bpp == bpp) return;

    rs->fb_width    = width;
    rs->fb_height   = height;
    rs->fb_stride   = width * 4;            /* Metal texture: always BGRA8 */
    rs->vram_stride = stride ? stride : width * (bpp <= 16 ? 2u : 4u);
    rs->fb_bpp      = bpp;
    rs->fb_tex      = make_fb_texture(rs->device, width, height);
    rs->depth_tex   = make_depth_texture(rs->device, width, height);

    free(rs->conv_buf);
    rs->conv_buf = (uint32_t *)malloc(width * height * sizeof(uint32_t));

    NSLog(@"ati_render: framebuffer %ux%u bpp=%u vram_stride=%u",
          width, height, bpp, rs->vram_stride);
}

void ati_metal_destroy(ATIRenderState *rs)
{
    if (!rs) return;
    {
        uint32_t _i;
        for (_i = 0; _i < ATI_TEX_SLOTS; _i++) rs->tex_pool[_i] = nil;
    }
    rs->device      = nil;
    rs->queue       = nil;
    rs->pipeline    = nil;
    rs->pipeline_tex = nil;
    rs->depth_state = nil;
    rs->fb_tex      = nil;
    rs->depth_tex   = nil;
    rs->vbuf        = nil;
    free(rs->conv_buf);
    free(rs);
}

/* ---------- PM4 parsing --------------------------------------------------- */

/* Dwords per vertex given format flags (XY always adds 2 implicit dwords). */
static uint32_t vert_stride(uint32_t fmt)
{
    uint32_t n = 2;  /* XY always present */
    if (fmt & VFMT_Z)     n += 1;
    if (fmt & VFMT_W)     n += 1;
    if (fmt & VFMT_COLOR) n += 1;
    if (fmt & VFMT_SPEC)  n += 1;
    if (fmt & VFMT_FOG)   n += 1;
    if (fmt & VFMT_ST0)   n += 2;
    if (fmt & VFMT_ST1)   n += 2;
    return n;
}

static void unpack_vertex(const uint32_t *d, uint32_t fmt, AtiVertex *v)
{
    uint32_t off = 0;

    /* XY: two floats (screen pixel coordinates) */
    v->x = *(const float *)&d[off++];
    v->y = *(const float *)&d[off++];

    v->z = 0.5f;  /* default mid-depth if no Z in packet */
    if (fmt & VFMT_Z) { v->z = *(const float *)&d[off++]; }
    if (fmt & VFMT_W) { off++; }

    /* Packed ARGB colour */
    v->r = v->g = v->b = 1.0f;
    v->a = 1.0f;
    if (fmt & VFMT_COLOR) {
        uint32_t c = d[off++];
        v->a = ((c >> 24) & 0xffu) / 255.0f;
        v->r = ((c >> 16) & 0xffu) / 255.0f;
        v->g = ((c >>  8) & 0xffu) / 255.0f;
        v->b = ( c        & 0xffu) / 255.0f;
    }

    if (fmt & VFMT_SPEC) { off++; }
    if (fmt & VFMT_FOG)  { off++; }

    v->u = v->v = 0.0f;
    if (fmt & VFMT_ST0) {
        v->u = *(const float *)&d[off++];
        v->v = *(const float *)&d[off++];
    }
    if (fmt & VFMT_ST1) { off += 2; }
}

static MTLPrimitiveType map_prim(uint32_t p)
{
    switch (p) {
    case PRIM_POINTS:    return MTLPrimitiveTypePoint;
    case PRIM_LINES:     return MTLPrimitiveTypeLine;
    case PRIM_POLY_LINE: return MTLPrimitiveTypeLineStrip;
    case PRIM_TRI_LIST:  return MTLPrimitiveTypeTriangle;
    case PRIM_TRI_FAN:   return MTLPrimitiveTypeTriangle;   /* fan→list approx */
    case PRIM_TRI_STRIP: return MTLPrimitiveTypeTriangleStrip;
    default:             return MTLPrimitiveTypeTriangle;
    }
}

/* ---------- main submit entry point --------------------------------------- */

void ati_metal_submit(ATIRenderState *rs,
                      const uint32_t *pm4, uint32_t ndwords)
{
    if (!rs || !pm4 || !ndwords || !rs->fb_tex) return;

    @autoreleasepool {

    AtiVertex *verts      = (AtiVertex *)[rs->vbuf contents];
    DrawCmd    draws[MAX_DRAWS];
    uint32_t   ndraw      = 0;
    uint32_t   ntotal     = 0;  /* vertices written so far */
    uint32_t   active_tex = rs->active_tex;
    uint32_t   i          = 0;

    /* ---- first pass: parse PM4, fill vertex buffer and draw list ---- */
    while (i < ndwords) {
        uint32_t h    = pm4[i];
        uint32_t type = PM4_TYPE(h);

        if (type == 0) {
            /* Type-0: register write(s) — update tracked state */
            uint32_t reg = PM4_PKT0_REG(h);
            uint32_t cnt = PM4_PKT0_CNT(h);
            if (i + 1 + cnt >= ndwords) break;
            for (uint32_t k = 0; k <= cnt; k++) {
                switch (reg + k * 4) {
                case PM4_VC_FPU_SETUP:
                    rs->vc_fpu_setup = pm4[i + 1 + k];
                    break;
                default:
                    break;
                }
            }
            i += 2 + cnt;

        } else if (type == 3) {
            uint32_t op  = PM4_PKT3_OP(h);
            uint32_t cnt = PM4_PKT3_CNT(h);
            if (i + 1 + cnt > ndwords) break;

            const uint32_t *dw = pm4 + i + 1;

            if (op == R128_OP_3D_DRAW_IMMD && cnt >= 2) {
                uint32_t fmt       = dw[0];
                uint32_t prim_info = dw[1];
                uint32_t prim_type = prim_info & 0xfu;
                uint32_t nvert     = (prim_info >> 16) & 0xffffu;
                uint32_t vstride   = vert_stride(fmt);

                if (nvert && ndraw < MAX_DRAWS &&
                    ntotal + nvert <= MAX_VERTS &&
                    2 + nvert * vstride <= cnt + 1)
                {
                    const uint32_t *src = dw + 2;
                    for (uint32_t v = 0; v < nvert; v++) {
                        unpack_vertex(src, fmt, &verts[ntotal + v]);
                        src += vstride;
                    }
                    draws[ndraw].vert_start = ntotal;
                    draws[ndraw].vert_count = nvert;
                    draws[ndraw].prim       = map_prim(prim_type);
                    draws[ndraw].tex_slot   = active_tex;
                    draws[ndraw].has_tex    = (fmt & VFMT_ST0) != 0;
                    ndraw++;
                    ntotal += nvert;
                }
            } else if (op == ATI_OP_SET_TEXTURE && cnt >= 1) {
                active_tex = dw[0] & (ATI_TEX_SLOTS - 1);
                rs->active_tex = active_tex;

            } else if (op == ATI_OP_TEXTURE_UPLOAD && cnt >= 5) {
                uint32_t slot        = dw[0] & (ATI_TEX_SLOTS - 1);
                uint32_t tw          = dw[1];
                uint32_t th          = dw[2];
                uint32_t tstride     = dw[3];
                uint32_t vram_offset = dw[4];

                if (tw && th && vram_offset + (uint64_t)th * tstride <= rs->vram_size) {
                    MTLTextureDescriptor *td =
                        [MTLTextureDescriptor
                            texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                         width:tw height:th mipmapped:NO];
                    td.usage       = MTLTextureUsageShaderRead;
                    td.storageMode = MTLStorageModeShared;
                    rs->tex_pool[slot] = [rs->device newTextureWithDescriptor:td];

                    /* guest VRAM: big-endian ARGB → Metal BGRA via bswap32 */
                    uint32_t npx = tw * th;
                    const uint32_t *src = (const uint32_t *)(rs->vram_ptr + vram_offset);
                    uint32_t *tmp = (uint32_t *)malloc(npx * sizeof(uint32_t));
                    if (tmp) {
                        for (uint32_t _p = 0; _p < npx; _p++)
                            tmp[_p] = __builtin_bswap32(src[_p]);
                        MTLRegion full = MTLRegionMake2D(0, 0, tw, th);
                        [rs->tex_pool[slot] replaceRegion:full mipmapLevel:0
                                                withBytes:tmp bytesPerRow:tstride];
                        free(tmp);
                    }
                    NSLog(@"ati_render: texture slot %u uploaded %ux%u", slot, tw, th);
                }
            }
            /* 3D_DRAW_VBUF / 3D_DRAW_INDX: vertices in VRAM — not yet */

            i += 1 + cnt;

        } else if (type == 2) {
            i++;  /* type-2: filler/NOP */
        } else {
            /* type-1: two-register write */
            if (i + 2 >= ndwords) break;
            i += 3;
        }
    }

    /* Dump raw PM4 dwords to diagnose parse failures */
    {
        NSMutableString *s = [NSMutableString stringWithFormat:@"ati_render: %u dwords:", ndwords];
        for (uint32_t _d = 0; _d < ndwords && _d < 16; _d++) {
            [s appendFormat:@" %08x", pm4[_d]];
        }
        NSLog(@"%@", s);
    }
    NSLog(@"ati_render: ndraw=%u ntotal=%u", ndraw, ntotal);
    if (!ndraw) return;

    /* ---- upload current guest VRAM into Metal texture (BGRA8Unorm) ----
     * 32-bit: guest big-endian ARGB → bswap32 → host LE BGRA8.
     * 16-bit: guest big-endian xRGB555 → expand to host LE BGRA8. */
    if (rs->conv_buf) {
        uint32_t npx = rs->fb_width * rs->fb_height;
        MTLRegion full = MTLRegionMake2D(0, 0, rs->fb_width, rs->fb_height);
        if (rs->fb_bpp <= 16) {
            const uint16_t *src16 = (const uint16_t *)rs->vram_ptr;
            for (uint32_t p = 0; p < npx; p++) {
                uint16_t px = __builtin_bswap16(src16[p]);
                uint8_t r = (uint8_t)((px >> 10) & 0x1fu); r = (r << 3) | (r >> 2);
                uint8_t g = (uint8_t)((px >>  5) & 0x1fu); g = (g << 3) | (g >> 2);
                uint8_t b = (uint8_t)( px        & 0x1fu); b = (b << 3) | (b >> 2);
                rs->conv_buf[p] = (0xffu << 24) | ((uint32_t)r << 16) |
                                  ((uint32_t)g << 8) | b;
            }
        } else {
            const uint32_t *src32 = (const uint32_t *)rs->vram_ptr;
            for (uint32_t p = 0; p < npx; p++)
                rs->conv_buf[p] = __builtin_bswap32(src32[p]);
        }
        [rs->fb_tex replaceRegion:full mipmapLevel:0
                        withBytes:rs->conv_buf
                      bytesPerRow:rs->fb_stride];  /* fb_stride = fb_width * 4 */
    }

    /* ---- second pass: issue all draws in one command buffer pass ---- */
    id<MTLCommandBuffer>      cb  = [rs->queue commandBuffer];
    MTLRenderPassDescriptor  *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture     = rs->fb_tex;
    rpd.colorAttachments[0].loadAction  = MTLLoadActionLoad;   /* preserve background */
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.depthAttachment.texture     = rs->depth_tex;
    rpd.depthAttachment.loadAction  = MTLLoadActionClear;  /* reset depth each flush */
    rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
    rpd.depthAttachment.clearDepth  = 1.0;

    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
    [enc setDepthStencilState:rs->depth_state];
    [enc setVertexBuffer:rs->vbuf offset:0 atIndex:0];

    float fbsz[2] = { (float)rs->fb_width, (float)rs->fb_height };
    [enc setVertexBytes:fbsz length:sizeof(fbsz) atIndex:1];

    for (uint32_t d = 0; d < ndraw; d++) {
        bool use_tex = draws[d].has_tex &&
                       draws[d].tex_slot < ATI_TEX_SLOTS &&
                       rs->tex_pool[draws[d].tex_slot] != nil;
        if (use_tex) {
            [enc setRenderPipelineState:rs->pipeline_tex];
            [enc setFragmentTexture:rs->tex_pool[draws[d].tex_slot] atIndex:0];
        } else {
            [enc setRenderPipelineState:rs->pipeline];
        }
        [enc drawPrimitives:draws[d].prim
                vertexStart:draws[d].vert_start
                vertexCount:draws[d].vert_count];
    }

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    /* ---- readback rendered pixels into guest VRAM ----
     * Metal BGRA8 → bswap32 → 32-bit big-endian ARGB, or
     * Metal BGRA8 → pack → 16-bit big-endian xRGB555. */
    {
        MTLRegion full = MTLRegionMake2D(0, 0, rs->fb_width, rs->fb_height);
        uint32_t npx = rs->fb_width * rs->fb_height;
        if (rs->conv_buf) {
            [rs->fb_tex getBytes:rs->conv_buf
                    bytesPerRow:rs->fb_stride  /* fb_stride = fb_width * 4 */
                     fromRegion:full
                    mipmapLevel:0];
            if (rs->fb_bpp <= 16) {
                uint16_t *dst16 = (uint16_t *)rs->vram_ptr;
                for (uint32_t p = 0; p < npx; p++) {
                    uint32_t bgra = rs->conv_buf[p];
                    uint8_t b8 =  bgra        & 0xffu;
                    uint8_t g8 = (bgra >>  8) & 0xffu;
                    uint8_t r8 = (bgra >> 16) & 0xffu;
                    uint16_t px = ((uint16_t)(r8 >> 3) << 10) |
                                  ((uint16_t)(g8 >> 3) <<  5) |
                                   (uint16_t)(b8 >> 3);
                    dst16[p] = __builtin_bswap16(px);
                }
            } else {
                uint32_t *dst32 = (uint32_t *)rs->vram_ptr;
                for (uint32_t p = 0; p < npx; p++)
                    dst32[p] = __builtin_bswap32(rs->conv_buf[p]);
            }
        }

        /* Probe pixel (320,240) — meaningful only when in bounds */
        if (rs->fb_width > 320 && rs->fb_height > 240) {
            uint32_t probe_px;
            if (rs->fb_bpp <= 16) {
                uint16_t px16 = __builtin_bswap16(
                    ((const uint16_t *)rs->vram_ptr)[240 * rs->fb_width + 320]);
                probe_px = px16;
            } else {
                probe_px = ((const uint32_t *)rs->vram_ptr)[240 * rs->fb_width + 320];
            }
            NSLog(@"ati_render: readback done, pixel(320,240)=0x%08x", probe_px);
        }
    }

    } /* @autoreleasepool */
}
