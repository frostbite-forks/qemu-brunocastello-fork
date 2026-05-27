/*
 * QEMU NVIDIA GeForce — inline Metal 3D renderer (plain C interface)
 *
 * Called from geforce.c (C) with NV PGRAPH push-buffer dwords already
 * byte-swapped to host LE.
 * Implementation is in geforce_render.m (Objective-C + Metal, darwin only).
 *
 * Non-darwin builds get no-op stubs so geforce.c compiles unchanged.
 */

#ifndef GEFORCE_RENDER_H
#define GEFORCE_RENDER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GFRenderState GFRenderState;

/*
 * Allocate and initialise the Metal renderer.
 * vram_ptr  — pointer to the guest VRAM buffer.
 * vram_size — total VRAM size in bytes.
 * Returns NULL on failure (no Metal device, or non-darwin build via stubs).
 */
GFRenderState *gf_metal_init(uint8_t *vram_ptr, uint32_t vram_size);

/*
 * Update the active framebuffer region inside VRAM.
 * width / height — framebuffer dimensions in pixels.
 * stride         — row stride in bytes.
 * bpp            — guest color depth: 8, 15, 16, or 32.
 * fb_offset      — byte offset of the framebuffer within VRAM.
 */
void gf_metal_set_fb(GFRenderState *rs,
                     uint32_t width, uint32_t height, uint32_t stride,
                     uint32_t bpp, uint32_t fb_offset);

/*
 * Supply the 256-entry RGB palette used when bpp <= 8.
 * pal  — 256 × 3 bytes (R, G, B), each 0-255.
 * n    — number of entries (must be <= 256).
 */
void gf_metal_set_palette(GFRenderState *rs, const uint8_t *pal, uint32_t n);

/*
 * Parse and execute a batch of NV push-buffer dwords.
 * pb      — host-LE dwords read from the guest push buffer.
 * ndwords — number of dwords in the batch.
 * Rendered pixels are written back into vram_ptr after the batch.
 */
void gf_metal_submit(GFRenderState *rs,
                     const uint32_t *pb, uint32_t ndwords);

/* Release all Metal objects. */
void gf_metal_destroy(GFRenderState *rs);

#ifdef __cplusplus
}
#endif

/* ------------------------------------------------------------------ */
/* No-op stubs for non-darwin builds                                   */
/* ------------------------------------------------------------------ */
#ifndef __APPLE__
static inline GFRenderState *gf_metal_init(uint8_t *v, uint32_t s)
    { (void)v; (void)s; return NULL; }
static inline void gf_metal_set_fb(GFRenderState *r,
    uint32_t w, uint32_t h, uint32_t s, uint32_t b, uint32_t o)
    { (void)r; (void)w; (void)h; (void)s; (void)b; (void)o; }
static inline void gf_metal_set_palette(GFRenderState *r,
    const uint8_t *p, uint32_t n)
    { (void)r; (void)p; (void)n; }
static inline void gf_metal_submit(GFRenderState *r,
    const uint32_t *p, uint32_t n)
    { (void)r; (void)p; (void)n; }
static inline void gf_metal_destroy(GFRenderState *r) { (void)r; }
#endif

#endif /* GEFORCE_RENDER_H */
