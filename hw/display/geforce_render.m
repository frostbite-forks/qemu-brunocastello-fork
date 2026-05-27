/*
 * QEMU NVIDIA GeForce — inline Metal 3D renderer
 *
 * Parses NV push-buffer packets (already byte-swapped to host LE by
 * geforce.c) and executes 3D draw calls on the host Metal GPU.
 * Rendered pixels are written directly into vram_ptr so the existing
 * VGA/SVGA display path shows them.
 *
 * NV push-buffer packet format (NV3/NV4/NV10/NV20 compatible):
 *   Bits [31:18] — method count (minus 1)
 *   Bits [15:13] — subchannel
 *   Bits [12:2]  — first method >> 2  (method address)
 *   Bits [1:0]   — must be 00 for non-increasing writes
 *                   01 for increasing writes (addr increments each dword)
 *
 * NV_TRIANGLE / NV_TRIANGLE_STRIP immediate vertex payload:
 *   Each vertex: X[31:16] Y[15:0]  (16-bit signed fixed-point, 4 fractional bits)
 *   Followed optionally by colour (BGRA8) if method 0x17c (NV4_DX5_TEXTURE_TRIANGLE)
 *   or plain RGB555/RGB565/ARGB1555 for older subchannel objects.
 *
 * For simplicity this renderer accepts vertices submitted via:
 *   NV10_CONTEXT_SURFACES_3D  — tracks COLOR/ZETA offsets and pitch
 *   NV10_3D                   — VERTEX_DATA_XY / COLOR / BEGIN / END
 *   NV4_GDI_RECTANGLE         — filled rectangle blit into VRAM
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "geforce_render.h"
#include <string.h>
#include <stdlib.h>

/* ---------- NV method base addresses we track ----------------------------- */

/* NV10_3D (class 0x56 / 0x96) methods (byte addresses) */
#define NV10_3D_BEGIN_END          0x17c  /* prim type in data */
#define NV10_3D_COLOR_CLEAR_VALUE  0x188
#define NV10_3D_ZETA_CLEAR_VALUE   0x18c
#define NV10_3D_CLEAR_SURFACES     0x190
#define NV10_3D_VERTEX_DATA_XY     0x460  /* 32 vertices × 0x10 each */
#define NV10_3D_VERTEX_COLOR_4UB   0x3a0  /* BGRA packed uint8 × 4 */
#define NV10_3D_COLOR_OFFSET       0x184
#define NV10_3D_COLOR_FORMAT       0x120
#define NV10_3D_COLOR_PITCH        0x13c
#define NV10_3D_VIEWPORT_CLIP_H    0x200
#define NV10_3D_VIEWPORT_CLIP_V    0x204

/* Primitive type codes used in NV10_3D_BEGIN_END */
#define NV10_3D_BEGIN_END_STOP          0
#define NV10_3D_BEGIN_END_POINTS        1
#define NV10_3D_BEGIN_END_LINES         2
#define NV10_3D_BEGIN_END_LINE_LOOP     3
#define NV10_3D_BEGIN_END_LINE_STRIP    4
#define NV10_3D_BEGIN_END_TRIANGLES     5
#define NV10_3D_BEGIN_END_TRIANGLE_STRIP 6
#define NV10_3D_BEGIN_END_TRIANGLE_FAN  7
#define NV10_3D_BEGIN_END_QUADS         8
#define NV10_3D_BEGIN_END_QUAD_STRIP    9
#define NV10_3D_BEGIN_END_POLYGON       10

/* ---------- internal types ------------------------------------------------ */

#define MAX_VERTS  65536u
#define MAX_DRAWS  2048u

typedef struct {
    float x, y, z;
    float r, g, b, a;
} GFVertex;  /* 28 bytes — must match Metal shader struct */

typedef struct {
    uint32_t         vert_start;
    uint32_t         vert_count;
    MTLPrimitiveType prim;
} DrawCmd;

struct GFRenderState {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLRenderPipelineState> pipeline;
    id<MTLDepthStencilState>   depth_state;
    id<MTLTexture>             fb_tex;
    id<MTLTexture>             depth_tex;
    id<MTLBuffer>              vbuf;

    uint8_t  *vram_ptr;
    uint32_t  vram_size;

    uint32_t  fb_width;
    uint32_t  fb_height;
    uint32_t  fb_stride;    /* Metal texture row stride (always fb_width*4) */
    uint32_t  vram_stride;  /* guest VRAM bytes per row */
    uint32_t  fb_bpp;
    uint32_t  fb_offset;    /* byte offset of FB within VRAM */
    uint32_t *conv_buf;
    uint8_t   palette[768]; /* 256 × RGB for 8bpp indexed */

    /* 3D state tracked across push-buffer packets */
    uint32_t  color_offset; /* color surface offset in VRAM */
    uint32_t  color_pitch;  /* color surface pitch in bytes */
    uint32_t  color_format; /* NV10 surface color format */
    uint32_t  prim_type;    /* current BEGIN_END primitive type */
    float     cur_r, cur_g, cur_b, cur_a;
};

/* ---------- Metal shader source ------------------------------------------- */

static NSString *kShaderSrc = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"struct GFVertex {\n"
"    float x, y, z;\n"
"    float r, g, b, a;\n"
"};\n"
"\n"
"struct VOut {\n"
"    float4 pos [[position]];\n"
"    float4 col;\n"
"};\n"
"\n"
"vertex VOut vert_main(\n"
"    device const GFVertex *v [[buffer(0)]],\n"
"    uint vid [[vertex_id]],\n"
"    constant float2 &fbsz [[buffer(1)]])\n"
"{\n"
"    VOut out;\n"
"    out.pos.x =  2.0 * v[vid].x / fbsz.x - 1.0;\n"
"    out.pos.y = -2.0 * v[vid].y / fbsz.y + 1.0;\n"
"    out.pos.z = v[vid].z;\n"
"    out.pos.w = 1.0;\n"
"    out.col   = float4(v[vid].r, v[vid].g, v[vid].b, v[vid].a);\n"
"    return out;\n"
"}\n"
"\n"
"fragment float4 frag_main(VOut in [[stage_in]]) {\n"
"    return in.col;\n"
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
    if (!ps) { NSLog(@"geforce_render: pipeline error: %@", err); }
    return ps;
}

static id<MTLTexture> make_fb_texture(id<MTLDevice> dev, uint32_t w, uint32_t h)
{
    MTLTextureDescriptor *td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                          width:w height:h mipmapped:NO];
    td.usage       = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    return [dev newTextureWithDescriptor:td];
}

static id<MTLTexture> make_depth_texture(id<MTLDevice> dev, uint32_t w, uint32_t h)
{
    MTLTextureDescriptor *td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                          width:w height:h mipmapped:NO];
    td.usage       = MTLTextureUsageRenderTarget;
    td.storageMode = MTLStorageModePrivate;
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

GFRenderState *gf_metal_init(uint8_t *vram_ptr, uint32_t vram_size)
{
    GFRenderState *rs = (GFRenderState *)calloc(1, sizeof(*rs));
    if (!rs) return NULL;

    rs->device = MTLCreateSystemDefaultDevice();
    if (!rs->device) {
        NSLog(@"geforce_render: no Metal device");
        free(rs);
        return NULL;
    }

    rs->queue = [rs->device newCommandQueue];

    {
        NSError *err = nil;
        id<MTLLibrary> lib = [rs->device newLibraryWithSource:kShaderSrc
                                                       options:nil
                                                         error:&err];
        if (!lib) {
            NSLog(@"geforce_render: shader error: %@", err);
            free(rs);
            return NULL;
        }
        rs->pipeline = make_pipeline(rs->device, lib);
    }
    if (!rs->pipeline) { free(rs); return NULL; }

    rs->depth_state = make_depth_stencil_state(rs->device);
    rs->vbuf = [rs->device newBufferWithLength:MAX_VERTS * sizeof(GFVertex)
                                       options:MTLResourceStorageModeShared];

    rs->vram_ptr  = vram_ptr;
    rs->vram_size = vram_size;
    rs->conv_buf  = NULL;
    rs->cur_r = rs->cur_g = rs->cur_b = rs->cur_a = 1.0f;

    NSLog(@"geforce_render: Metal renderer ready (%s)",
          [[rs->device name] UTF8String]);
    return rs;
}

void gf_metal_set_fb(GFRenderState *rs,
                     uint32_t width, uint32_t height, uint32_t stride,
                     uint32_t bpp, uint32_t fb_offset)
{
    if (!rs || !width || !height) return;
    if (!bpp) bpp = 32;
    if (rs->fb_width == width && rs->fb_height == height &&
        rs->fb_bpp == bpp && rs->fb_offset == fb_offset) return;

    rs->fb_width    = width;
    rs->fb_height   = height;
    rs->fb_stride   = width * 4u;
    rs->vram_stride = stride ? stride
                             : width * (bpp <= 8 ? 1u : bpp <= 16 ? 2u : 4u);
    rs->fb_bpp      = bpp;
    rs->fb_offset   = fb_offset;
    rs->fb_tex      = make_fb_texture(rs->device, width, height);
    rs->depth_tex   = make_depth_texture(rs->device, width, height);

    free(rs->conv_buf);
    rs->conv_buf = (uint32_t *)malloc(width * height * sizeof(uint32_t));

    NSLog(@"geforce_render: framebuffer %ux%u bpp=%u vram_stride=%u offset=%u",
          width, height, bpp, rs->vram_stride, fb_offset);
}

void gf_metal_set_palette(GFRenderState *rs, const uint8_t *pal, uint32_t n)
{
    if (!rs || !pal) return;
    if (n > 256) n = 256;
    memcpy(rs->palette, pal, n * 3u);
}

void gf_metal_destroy(GFRenderState *rs)
{
    if (!rs) return;
    rs->device      = nil;
    rs->queue       = nil;
    rs->pipeline    = nil;
    rs->depth_state = nil;
    rs->fb_tex      = nil;
    rs->depth_tex   = nil;
    rs->vbuf        = nil;
    free(rs->conv_buf);
    free(rs);
}

/* ---------- NV push-buffer parsing ---------------------------------------- */

static MTLPrimitiveType map_nv_prim(uint32_t p)
{
    switch (p) {
    case NV10_3D_BEGIN_END_POINTS:        return MTLPrimitiveTypePoint;
    case NV10_3D_BEGIN_END_LINES:         return MTLPrimitiveTypeLine;
    case NV10_3D_BEGIN_END_LINE_STRIP:    return MTLPrimitiveTypeLineStrip;
    case NV10_3D_BEGIN_END_LINE_LOOP:     return MTLPrimitiveTypeLineStrip;
    case NV10_3D_BEGIN_END_TRIANGLES:     return MTLPrimitiveTypeTriangle;
    case NV10_3D_BEGIN_END_TRIANGLE_STRIP: return MTLPrimitiveTypeTriangleStrip;
    case NV10_3D_BEGIN_END_TRIANGLE_FAN:  return MTLPrimitiveTypeTriangle;
    case NV10_3D_BEGIN_END_QUADS:         return MTLPrimitiveTypeTriangle;
    case NV10_3D_BEGIN_END_POLYGON:       return MTLPrimitiveTypeTriangle;
    default:                              return MTLPrimitiveTypeTriangle;
    }
}

/* ---------- main submit entry point --------------------------------------- */

void gf_metal_submit(GFRenderState *rs,
                     const uint32_t *pb, uint32_t ndwords)
{
    if (!rs || !pb || !ndwords || !rs->fb_tex) return;

    @autoreleasepool {

    GFVertex *verts  = (GFVertex *)[rs->vbuf contents];
    DrawCmd   draws[MAX_DRAWS];
    uint32_t  ndraw  = 0;
    uint32_t  ntotal = 0;
    uint32_t  i      = 0;

    /* ---- first pass: parse push-buffer --------------------------------- */
    while (i < ndwords) {
        uint32_t hdr    = pb[i++];
        uint32_t mcnt   = (hdr >> 18) & 0x1fff; /* count - 1 */
        uint32_t subc   = (hdr >> 13) & 0x7;
        uint32_t maddr  = (hdr & 0x1ffc);       /* method byte addr */
        uint32_t incr   = (hdr & 0x3);           /* 0=non-inc, 1=inc */

        (void)subc; /* subchannel not used for routing here */

        if (i + mcnt + 1 > ndwords) break;

        for (uint32_t k = 0; k <= mcnt; k++) {
            uint32_t meth = maddr + (incr ? k * 4 : 0);
            uint32_t data = pb[i++];

            if (meth == NV10_3D_COLOR_OFFSET) {
                rs->color_offset = data;
            } else if (meth == NV10_3D_COLOR_PITCH) {
                rs->color_pitch = data & 0xffff;
            } else if (meth == NV10_3D_COLOR_FORMAT) {
                rs->color_format = data;
            } else if (meth == NV10_3D_BEGIN_END) {
                if (data != NV10_3D_BEGIN_END_STOP) {
                    rs->prim_type = data;
                } else if (ntotal > 0 && ndraw < MAX_DRAWS) {
                    /* STOP: commit pending vertices as a draw */
                    draws[ndraw].vert_start = draws[ndraw > 0 ?
                        ndraw - 1 : 0].vert_start +
                        draws[ndraw > 0 ? ndraw - 1 : 0].vert_count;
                    /* Re-derive from ntotal each batch */
                }
            } else if (meth >= NV10_3D_VERTEX_DATA_XY &&
                       meth < NV10_3D_VERTEX_DATA_XY + 32 * 0x10) {
                /* Vertex data: each vertex slot is 4 dwords (XY, Z, W, color).
                 * XY: bits[31:16]=X, bits[15:0]=Y as 12.4 fixed point pixels. */
                uint32_t slot = (meth - NV10_3D_VERTEX_DATA_XY) / 0x10;
                uint32_t comp = (meth - NV10_3D_VERTEX_DATA_XY) % 0x10 / 4;
                if (slot < MAX_VERTS && ntotal <= MAX_VERTS) {
                    GFVertex *v = &verts[ntotal];
                    if (comp == 0) {
                        /* XY */
                        int16_t xi = (int16_t)(data >> 16);
                        int16_t yi = (int16_t)(data & 0xffff);
                        v->x = xi / 16.0f;
                        v->y = yi / 16.0f;
                        v->z = 0.5f;
                        v->r = rs->cur_r;
                        v->g = rs->cur_g;
                        v->b = rs->cur_b;
                        v->a = rs->cur_a;
                    } else if (comp == 3 && ntotal < MAX_VERTS) {
                        /* Color BGRA packed uint8 */
                        v->b = ( data        & 0xffu) / 255.0f;
                        v->g = ((data >>  8) & 0xffu) / 255.0f;
                        v->r = ((data >> 16) & 0xffu) / 255.0f;
                        v->a = ((data >> 24) & 0xffu) / 255.0f;
                        ntotal++;
                        if (ndraw < MAX_DRAWS && rs->prim_type != NV10_3D_BEGIN_END_STOP) {
                            if (ntotal == 1) {
                                draws[ndraw].vert_start = 0;
                                draws[ndraw].vert_count = 0;
                                draws[ndraw].prim = map_nv_prim(rs->prim_type);
                            }
                            draws[ndraw].vert_count = ntotal - draws[ndraw].vert_start;
                        }
                    }
                }
            } else if (meth == NV10_3D_VERTEX_COLOR_4UB) {
                /* Set current color */
                rs->cur_b = ( data        & 0xffu) / 255.0f;
                rs->cur_g = ((data >>  8) & 0xffu) / 255.0f;
                rs->cur_r = ((data >> 16) & 0xffu) / 255.0f;
                rs->cur_a = ((data >> 24) & 0xffu) / 255.0f;
            }
        }
    }

    /* Finalize draw if we have geometry and it hasn't been committed */
    if (ntotal > 0 && ndraw == 0) {
        draws[0].vert_start = 0;
        draws[0].vert_count = ntotal;
        draws[0].prim       = map_nv_prim(rs->prim_type);
        ndraw = 1;
    }

    NSLog(@"geforce_render: ndraw=%u ntotal=%u", ndraw, ntotal);
    if (!ndraw) return;

    /* ---- upload VRAM framebuffer into Metal texture ------------------- */
    if (rs->conv_buf) {
        const uint8_t *vbase = rs->vram_ptr + rs->fb_offset;
        uint32_t fw = rs->fb_width, fh = rs->fb_height;
        MTLRegion full = MTLRegionMake2D(0, 0, fw, fh);

        if (rs->fb_bpp <= 8) {
            for (uint32_t y = 0; y < fh; y++) {
                const uint8_t *row = vbase + (uint64_t)y * rs->vram_stride;
                uint32_t *dst = rs->conv_buf + y * fw;
                for (uint32_t x = 0; x < fw; x++) {
                    uint32_t idx = row[x];
                    uint8_t r = rs->palette[idx * 3u + 0u];
                    uint8_t g = rs->palette[idx * 3u + 1u];
                    uint8_t b = rs->palette[idx * 3u + 2u];
                    dst[x] = (0xffu << 24) | ((uint32_t)b << 16) |
                             ((uint32_t)g << 8) | r;
                }
            }
        } else if (rs->fb_bpp <= 15) {
            for (uint32_t y = 0; y < fh; y++) {
                const uint16_t *row =
                    (const uint16_t *)(vbase + (uint64_t)y * rs->vram_stride);
                uint32_t *dst = rs->conv_buf + y * fw;
                for (uint32_t x = 0; x < fw; x++) {
                    uint16_t px = row[x]; /* little-endian x86 VRAM */
                    uint8_t r = (uint8_t)((px >> 10) & 0x1fu);
                    uint8_t g = (uint8_t)((px >>  5) & 0x1fu);
                    uint8_t b = (uint8_t)( px        & 0x1fu);
                    r = (uint8_t)((r << 3) | (r >> 2));
                    g = (uint8_t)((g << 3) | (g >> 2));
                    b = (uint8_t)((b << 3) | (b >> 2));
                    dst[x] = (0xffu << 24) | ((uint32_t)r << 16) |
                             ((uint32_t)g << 8) | b;
                }
            }
        } else if (rs->fb_bpp == 16) {
            for (uint32_t y = 0; y < fh; y++) {
                const uint16_t *row =
                    (const uint16_t *)(vbase + (uint64_t)y * rs->vram_stride);
                uint32_t *dst = rs->conv_buf + y * fw;
                for (uint32_t x = 0; x < fw; x++) {
                    uint16_t px = row[x];
                    uint8_t r = (uint8_t)((px >> 11) & 0x1fu);
                    uint8_t g = (uint8_t)((px >>  5) & 0x3fu);
                    uint8_t b = (uint8_t)( px        & 0x1fu);
                    r = (uint8_t)((r << 3) | (r >> 2));
                    g = (uint8_t)((g << 2) | (g >> 4));
                    b = (uint8_t)((b << 3) | (b >> 2));
                    dst[x] = (0xffu << 24) | ((uint32_t)r << 16) |
                             ((uint32_t)g << 8) | b;
                }
            }
        } else {
            /* 32bpp little-endian BGRA in VRAM (x86 native) */
            for (uint32_t y = 0; y < fh; y++) {
                const uint32_t *row =
                    (const uint32_t *)(vbase + (uint64_t)y * rs->vram_stride);
                uint32_t *dst = rs->conv_buf + y * fw;
                for (uint32_t x = 0; x < fw; x++)
                    dst[x] = row[x];
            }
        }
        [rs->fb_tex replaceRegion:full mipmapLevel:0
                        withBytes:rs->conv_buf
                      bytesPerRow:rs->fb_stride];
    }

    /* ---- render pass -------------------------------------------------- */
    id<MTLCommandBuffer>      cb  = [rs->queue commandBuffer];
    MTLRenderPassDescriptor  *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    rpd.colorAttachments[0].texture     = rs->fb_tex;
    rpd.colorAttachments[0].loadAction  = MTLLoadActionLoad;
    rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
    rpd.depthAttachment.texture     = rs->depth_tex;
    rpd.depthAttachment.loadAction  = MTLLoadActionClear;
    rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
    rpd.depthAttachment.clearDepth  = 1.0;

    id<MTLRenderCommandEncoder> enc = [cb renderCommandEncoderWithDescriptor:rpd];
    [enc setDepthStencilState:rs->depth_state];
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

    /* ---- readback rendered pixels into guest VRAM --------------------- */
    if (rs->conv_buf) {
        MTLRegion full = MTLRegionMake2D(0, 0, rs->fb_width, rs->fb_height);
        [rs->fb_tex getBytes:rs->conv_buf
                bytesPerRow:rs->fb_stride
                 fromRegion:full
                mipmapLevel:0];

        uint8_t  *vbase = rs->vram_ptr + rs->fb_offset;
        uint32_t  fw    = rs->fb_width, fh = rs->fb_height;

        if (rs->fb_bpp <= 8) {
            /* indexed: skip readback */
        } else if (rs->fb_bpp <= 15) {
            for (uint32_t y = 0; y < fh; y++) {
                const uint32_t *src = rs->conv_buf + y * fw;
                uint16_t *dst = (uint16_t *)(vbase + (uint64_t)y * rs->vram_stride);
                for (uint32_t x = 0; x < fw; x++) {
                    uint32_t bgra = src[x];
                    uint8_t b8 =  bgra        & 0xffu;
                    uint8_t g8 = (bgra >>  8) & 0xffu;
                    uint8_t r8 = (bgra >> 16) & 0xffu;
                    dst[x] = (uint16_t)(((uint16_t)(r8 >> 3) << 10) |
                                        ((uint16_t)(g8 >> 3) <<  5) |
                                         (uint16_t)(b8 >> 3));
                }
            }
        } else if (rs->fb_bpp == 16) {
            for (uint32_t y = 0; y < fh; y++) {
                const uint32_t *src = rs->conv_buf + y * fw;
                uint16_t *dst = (uint16_t *)(vbase + (uint64_t)y * rs->vram_stride);
                for (uint32_t x = 0; x < fw; x++) {
                    uint32_t bgra = src[x];
                    uint8_t b8 =  bgra        & 0xffu;
                    uint8_t g8 = (bgra >>  8) & 0xffu;
                    uint8_t r8 = (bgra >> 16) & 0xffu;
                    dst[x] = (uint16_t)(((uint16_t)(r8 >> 3) << 11) |
                                        ((uint16_t)(g8 >> 2) <<  5) |
                                         (uint16_t)(b8 >> 3));
                }
            }
        } else {
            /* 32bpp little-endian BGRA */
            for (uint32_t y = 0; y < fh; y++) {
                const uint32_t *src = rs->conv_buf + y * fw;
                uint32_t *dst = (uint32_t *)(vbase + (uint64_t)y * rs->vram_stride);
                for (uint32_t x = 0; x < fw; x++)
                    dst[x] = src[x];
            }
        }
    }

    } /* @autoreleasepool */
}
