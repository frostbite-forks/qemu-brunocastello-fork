/*
 * QEMU ATI Rage 128 — 3D passthrough (inline Metal renderer)
 *
 * Called when the guest advances PM4_BUFFER_DL_WPTR.
 * Reads new PM4 dwords from the guest ring buffer (guest VRAM, big-endian PPC),
 * byte-swaps them to host LE, and forwards them to ati_render.m for Metal
 * execution.  Rendered pixels land directly in s->vga.vram_ptr.
 *
 * Ring buffer protocol:
 *   Guest writes PM4_BUFFER_OFFSET  → base physical address of ring
 *   Guest writes PM4_BUFFER_CNTL   → size in 4-dword units (bits[27:0])
 *   Guest writes PM4_BUFFER_DL_RPTR_ADDR → guest phys addr to echo rptr
 *   Guest advances PM4_BUFFER_DL_WPTR   → triggers this function
 */

#include "qemu/osdep.h"
#include "ati_int.h"
#include "ati_render.h"
#include "exec/cpu-common.h"
#include "system/memory.h"         /* memory_region_set_dirty */
#include "qemu/bswap.h"   /* cpu_to_le32 for rptr write-back */
#include "hw/display/bochs-vbe.h"  /* VBE_DISPI_INDEX_XRES/YRES */

#define PM4_MAX_BATCH  16384u   /* dwords per flush segment */

void ati_3d_flush(ATIVGAState *s)
{
    uint32_t rptr      = s->pm4.rptr;
    uint32_t wptr      = s->pm4.wptr;
    uint32_t buf_dw    = s->pm4.buf_size;  /* ring size in dwords */

    if (!buf_dw || !s->pm4.buf_addr || !s->render) {
        s->pm4.rptr = wptr;
        return;
    }

    /* Derive framebuffer dimensions from VBE regs (set by ati_vga_switch_mode) */
    uint32_t fb_w = s->vga.vbe_regs[VBE_DISPI_INDEX_XRES];
    uint32_t fb_h = s->vga.vbe_regs[VBE_DISPI_INDEX_YRES];
    uint32_t pitch_px = (s->regs.crtc_pitch & 0x7ffu) * 8u;
    uint32_t fb_stride = (pitch_px ? pitch_px : fb_w) * 4u;  /* 32 bpp */
    if (!fb_w || !fb_h) { s->pm4.rptr = wptr; return; }
    ati_metal_set_fb(s->render, fb_w, fb_h, fb_stride);

    while (rptr != wptr) {
        uint32_t avail = (wptr > rptr)
            ? (wptr - rptr)
            : (buf_dw - rptr);

        if (!avail) break;
        if (avail > PM4_MAX_BATCH) avail = PM4_MAX_BATCH;

        uint32_t bytes = avail * sizeof(uint32_t);
        uint32_t *tmp  = g_malloc(bytes);

        cpu_physical_memory_read(
            s->pm4.buf_addr + (uint64_t)rptr * sizeof(uint32_t),
            tmp, bytes);

        /*
         * The NDRV writes ring buffer dwords in LE (EndianU32_NtoL on PPC),
         * matching ATI hardware convention. No byte-swap needed here;
         * ati_metal_submit() parses them as LE directly.
         */
        ati_metal_submit(s->render, tmp, avail);
        g_free(tmp);

        /* Mark framebuffer VRAM dirty so QEMU's display update redraws it.
         * We wrote pixels directly into vram_ptr, bypassing CPU write tracking. */
        memory_region_set_dirty(&s->vga.vram, 0,
                                (hwaddr)fb_h * fb_stride);

        rptr = (rptr + avail) % buf_dw;
    }

    s->pm4.rptr = wptr;

    /* Echo updated rptr to guest if it requested a write-back address */
    if (s->pm4.rptr_addr) {
        uint32_t val = cpu_to_le32(s->pm4.rptr);
        cpu_physical_memory_write(s->pm4.rptr_addr, &val, sizeof(val));
    }
}
