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
    float x, y;
    float r, g, b, a;
    float u, v;
} AtiVertex;  /* 32 bytes — must match Metal shader struct */

typedef struct {
    uint32_t           vert_start;
    uint32_t           vert_count;
    MTLPrimitiveType   prim;
} DrawCmd;

struct ATIRenderState {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLRenderPipelineState> pipeline;
    id<MTLTexture>             fb_tex;
    id<MTLBuffer>              vbuf;   /* MAX_VERTS × sizeof(AtiVertex) */

    uint8_t  *vram_ptr;
    uint32_t  vram_size;
    uint32_t  fb_width;
    uint32_t  fb_height;
    uint32_t  fb_stride;  /* bytes per row */

    /* 3D state updated by PM4 type-0 register writes */
    uint32_t  vc_fpu_setup;
};

/* ---------- Metal shader source ------------------------------------------- */

static NSString *kShaderSrc = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct AtiVertex {\n"
"    float2 pos;\n"
"    float4 col;\n"
"    float2 uv;\n"
"};\n"
"\n"
"struct VOut {\n"
"    float4 pos [[position]];\n"
"    float4 col;\n"
"};\n"
"\n"
"vertex VOut vert_main(\n"
"    device const AtiVertex *v [[buffer(0)]],\n"
"    uint vid [[vertex_id]],\n"
"    constant float2 &fbsz [[buffer(1)]])\n"
"{\n"
"    VOut out;\n"
"    out.pos.x =  2.0 * v[vid].pos.x / fbsz.x - 1.0;\n"
"    out.pos.y = -2.0 * v[vid].pos.y / fbsz.y + 1.0;\n"
"    out.pos.z = 0.0;\n"
"    out.pos.w = 1.0;\n"
"    out.col   = v[vid].col;\n"
"    return out;\n"
"}\n"
"\n"
"fragment float4 frag_main(VOut in [[stage_in]]) {\n"
"    return in.col;\n"
"}\n";

/* ---------- helpers -------------------------------------------------------- */

static id<MTLRenderPipelineState> make_pipeline(id<MTLDevice> dev)
{
    NSError *err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:kShaderSrc
                                           options:nil
                                             error:&err];
    if (!lib) {
        NSLog(@"ati_render: shader error: %@", err);
        return nil;
    }

    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    d.vertexFunction   = [lib newFunctionWithName:@"vert_main"];
    d.fragmentFunction = [lib newFunctionWithName:@"frag_main"];
    d.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;

    /* Standard src-alpha blending */
    d.colorAttachments[0].blendingEnabled             = YES;
    d.colorAttachments[0].rgbBlendOperation           = MTLBlendOperationAdd;
    d.colorAttachments[0].alphaBlendOperation         = MTLBlendOperationAdd;
    d.colorAttachments[0].sourceRGBBlendFactor        = MTLBlendFactorSourceAlpha;
    d.colorAttachments[0].destinationRGBBlendFactor   = MTLBlendFactorOneMinusSourceAlpha;
    d.colorAttachments[0].sourceAlphaBlendFactor      = MTLBlendFactorOne;
    d.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

    id<MTLRenderPipelineState> ps =
        [dev newRenderPipelineStateWithDescriptor:d error:&err];
    if (!ps) {
        NSLog(@"ati_render: pipeline error: %@", err);
    }
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

    rs->queue    = [rs->device newCommandQueue];
    rs->pipeline = make_pipeline(rs->device);
    if (!rs->pipeline) { free(rs); return NULL; }

    rs->vbuf = [rs->device newBufferWithLength:MAX_VERTS * sizeof(AtiVertex)
                                       options:MTLResourceStorageModeShared];

    rs->vram_ptr  = vram_ptr;
    rs->vram_size = vram_size;

    NSLog(@"ati_render: Metal renderer ready (%s)",
          [[rs->device name] UTF8String]);
    return rs;
}

void ati_metal_set_fb(ATIRenderState *rs,
                      uint32_t width, uint32_t height, uint32_t stride)
{
    if (!rs || !width || !height) return;
    if (rs->fb_width == width && rs->fb_height == height) return;

    rs->fb_width  = width;
    rs->fb_height = height;
    rs->fb_stride = stride ? stride : width * 4;
    rs->fb_tex    = make_fb_texture(rs->device, width, height);

    NSLog(@"ati_render: framebuffer %ux%u stride %u",
          width, height, rs->fb_stride);
}

void ati_metal_destroy(ATIRenderState *rs)
{
    if (!rs) return;
    rs->device   = nil;
    rs->queue    = nil;
    rs->pipeline = nil;
    rs->fb_tex   = nil;
    rs->vbuf     = nil;
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

    /* Z */
    if (fmt & VFMT_Z)  { off++; }
    if (fmt & VFMT_W)  { off++; }

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

    AtiVertex *verts  = (AtiVertex *)[rs->vbuf contents];
    DrawCmd    draws[MAX_DRAWS];
    uint32_t   ndraw   = 0;
    uint32_t   ntotal  = 0;  /* vertices written so far */
    uint32_t   i       = 0;

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
                    ndraw++;
                    ntotal += nvert;
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

    /* ---- second pass: issue all draws in one command buffer pass ---- */
    id<MTLCommandBuffer>      cb  = [rs->queue commandBuffer];
    MTLRenderPassDescriptor  *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture     = rs->fb_tex;
    rpd.colorAttachments[0].loadAction  = MTLLoadActionLoad;   /* accumulate */
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
    [enc setRenderPipelineState:rs->pipeline];
    [enc setVertexBuffer:rs->vbuf offset:0 atIndex:0];

    float fbsz[2] = { (float)rs->fb_width, (float)rs->fb_height };
    [enc setVertexBytes:fbsz length:sizeof(fbsz) atIndex:1];

    for (uint32_t d = 0; d < ndraw; d++) {
        [enc drawPrimitives:draws[d].prim
                vertexStart:draws[d].vert_start
                vertexCount:draws[d].vert_count];
    }

    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    /* ---- readback rendered pixels into guest VRAM ---- */
    MTLRegion region = MTLRegionMake2D(0, 0, rs->fb_width, rs->fb_height);
    [rs->fb_tex getBytes:rs->vram_ptr
            bytesPerRow:rs->fb_stride
             fromRegion:region
            mipmapLevel:0];

    /* Debug: log first pixel value to verify readback and check format */
    {
        uint32_t px = ((uint32_t *)rs->vram_ptr)[0];
        NSLog(@"ati_render: readback done, vram[0]=0x%08x", px);
    }

    } /* @autoreleasepool */
}
