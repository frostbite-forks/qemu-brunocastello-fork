/*
 * QEMU ATI Rage 128 — inline Metal 3D renderer (plain C interface)
 *
 * Called from ati_3d.c (C) with PM4 dwords already byte-swapped to host LE.
 * Implementation is in ati_render.m (Objective-C + Metal, darwin only).
 *
 * Non-darwin builds get no-op stubs so ati.c / ati_3d.c compile unchanged.
 */

#ifndef ATI_RENDER_H
#define ATI_RENDER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ATIRenderState ATIRenderState;

/*
 * Allocate and initialise the Metal renderer.
 * vram_ptr  — pointer to the guest VRAM buffer (s->vga.vram_ptr).
 * vram_size — total VRAM size in bytes.
 * Returns NULL on failure (no Metal device, or non-darwin build via stubs).
 */
ATIRenderState *ati_metal_init(uint8_t *vram_ptr, uint32_t vram_size);

/*
 * Update the active framebuffer region inside VRAM.
 * width / height — framebuffer dimensions in pixels.
 * stride         — row stride in bytes (fb_width * 4 for 32 bpp).
 * Must be called before the first ati_metal_submit() and whenever the
 * guest changes its display mode.
 */
void ati_metal_set_fb(ATIRenderState *rs,
                      uint32_t width, uint32_t height, uint32_t stride);

/*
 * Parse and execute a batch of PM4 dwords.
 * pm4    — host-LE dwords read from the guest ring buffer.
 * ndwords — number of dwords in the batch.
 * Rendered pixels are written back into vram_ptr after the batch.
 */
void ati_metal_submit(ATIRenderState *rs,
                      const uint32_t *pm4, uint32_t ndwords);

/* Release all Metal objects. */
void ati_metal_destroy(ATIRenderState *rs);

#ifdef __cplusplus
}
#endif

/* ------------------------------------------------------------------ */
/* No-op stubs for non-darwin builds                                   */
/* ------------------------------------------------------------------ */
#ifndef __APPLE__
static inline ATIRenderState *ati_metal_init(uint8_t *v, uint32_t s)
    { (void)v; (void)s; return NULL; }
static inline void ati_metal_set_fb(ATIRenderState *r,
    uint32_t w, uint32_t h, uint32_t s)
    { (void)r; (void)w; (void)h; (void)s; }
static inline void ati_metal_submit(ATIRenderState *r,
    const uint32_t *p, uint32_t n)
    { (void)r; (void)p; (void)n; }
static inline void ati_metal_destroy(ATIRenderState *r) { (void)r; }
#endif

#endif /* ATI_RENDER_H */
