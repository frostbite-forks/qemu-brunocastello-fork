/*
 * QEMU NVIDIA GeForce VGA emulation (software rendering, no Metal)
 *
 * Fresh port of the Bochs bx_geforce_c engine (Copyright 2025-2026 The Bochs
 * Project, LGPL v2+) into the QEMU PCI device model, using the DingusPPC
 * GeForce port's device structure as a skeleton.  All 2D acceleration is
 * performed in software on the CPU; there is no GPU/Metal back end.
 *
 * Supported models (-device geforce-vga,model=<name>):
 *   geforce2mx — NV11 (10DE:0110), GeForce2 MX,  64 MiB
 *   geforce3   — NV20 (10DE:0200), GeForce3,      64 MiB
 *
 * BAR layout (matches real hardware on Power Mac G4 AGP):
 *   BAR0  — 16 MiB MMIO register window (NV_PNPMMIO)
 *   BAR1  — VRAM linear aperture (prefetchable)
 *   RAMIN is reached through BAR0 0x700000 (no dedicated BAR2 on NV11/NV20).
 *
 * The display path reuses the QEMU VGA common infrastructure
 * (vga_common_init / vga_init) exactly as the ATI VGA emulation does.
 * Extended modes are activated via NV CRTC register 0x28 (depth byte):
 *   1 = 8 bpp, 2 = 16 bpp (565), 3 = 32 bpp (ARGB).
 */

#include "qemu/osdep.h"
#include "geforce_int.h"
#include "vga-access.h"
#include "vga_int.h"
#include "vga_regs.h"
#include "hw/display/bochs-vbe.h"
#include "hw/core/qdev-properties.h"
#include "qemu/bswap.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "qemu/error-report.h"
#include "qapi/error.h"
#include "ui/console.h"
#include "exec/cpu-common.h"
#include "system/memory.h"

/* ======================================================================== */
/* Model table                                                              */
/* ======================================================================== */

static const struct {
    const char *name;
    uint16_t    dev_id;
    uint32_t    card_type;
    uint32_t    vram_mb;
} nv_model_tab[] = {
    { "geforce2mx", PCI_DEVICE_ID_NV_GEFORCE2_MX, NV_CARD_NV11, 64 },
    { "geforce3",   PCI_DEVICE_ID_NV_GEFORCE3,    NV_CARD_NV20, 64 },
};

/* ======================================================================== */
/* Small helpers                                                            */
/* ======================================================================== */

#define NV_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#define NV_VGA_CRTC_LAST 0x18   /* standard VGA CRTC index range end */

static inline uint8_t alpha_wrap(int value)
{
    return (uint8_t)(-(value >> 8) ^ value);
}

static inline uint32_t color_565_to_888(uint16_t value)
{
    uint8_t r = ((value >> 11) & 0x1f);
    uint8_t g = ((value >> 5)  & 0x3f);
    uint8_t b = (value & 0x1f);
    r = (r << 3) | (r >> 2);
    g = (g << 2) | (g >> 4);
    b = (b << 3) | (b >> 2);
    return ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
}

static inline uint16_t color_888_to_565(uint32_t value)
{
    return (((value >> 19) & 0x1f) << 11) |
           (((value >> 10) & 0x3f) <<  5) |
           ((value >> 3) & 0x1f);
}

/*
 * Ternary ROP3.  Replaces the Bochs bitblt.h engine entirely.  For a given
 * 8-bit ROP code, the output for each bit position is the boolean function
 * of the destination (D), source (S) and pattern (P) bits encoded in the
 * code's truth table:
 *
 *   bit index = (P<<2) | (S<<1) | D   →  output bit = (rop >> index) & 1
 *
 * This is applied per colour-byte across cb bytes.
 */
static uint32_t rop3(uint8_t rop, uint32_t d, uint32_t s, uint32_t p, uint32_t cb)
{
    uint32_t out = 0;
    uint32_t nbits = cb * 8u;
    for (uint32_t bit = 0; bit < nbits; bit++) {
        uint32_t db = (d >> bit) & 1u;
        uint32_t sb = (s >> bit) & 1u;
        uint32_t pb = (p >> bit) & 1u;
        uint32_t idx = (pb << 2) | (sb << 1) | db;
        if ((rop >> idx) & 1u) {
            out |= (1u << bit);
        }
    }
    return out;
}

/* ======================================================================== */
/* VRAM / RAMIN access (little-endian backing store)                        */
/* ======================================================================== */

static inline uint8_t nv_vram_read8(NVVGAState *s, uint32_t a)
{
    return s->vga.vram_ptr[a & s->memsize_mask];
}
static inline uint16_t nv_vram_read16(NVVGAState *s, uint32_t a)
{
    return lduw_le_p(s->vga.vram_ptr + (a & s->memsize_mask));
}
static inline uint32_t nv_vram_read32(NVVGAState *s, uint32_t a)
{
    return ldl_le_p(s->vga.vram_ptr + (a & s->memsize_mask));
}
static inline uint64_t nv_vram_read64(NVVGAState *s, uint32_t a)
{
    return ldq_le_p(s->vga.vram_ptr + (a & s->memsize_mask));
}
static inline void nv_vram_write8(NVVGAState *s, uint32_t a, uint8_t v)
{
    s->vga.vram_ptr[a & s->memsize_mask] = v;
}
static inline void nv_vram_write16(NVVGAState *s, uint32_t a, uint16_t v)
{
    stw_le_p(s->vga.vram_ptr + (a & s->memsize_mask), v);
}
static inline void nv_vram_write32(NVVGAState *s, uint32_t a, uint32_t v)
{
    stl_le_p(s->vga.vram_ptr + (a & s->memsize_mask), v);
}
G_GNUC_UNUSED static inline void nv_vram_write64(NVVGAState *s, uint32_t a, uint64_t v)
{
    stq_le_p(s->vga.vram_ptr + (a & s->memsize_mask), v);
}

static inline uint8_t nv_ramin_read8(NVVGAState *s, uint32_t o)
{
    return nv_vram_read8(s, s->ramin_flip + o);
}
G_GNUC_UNUSED static inline uint16_t nv_ramin_read16(NVVGAState *s, uint32_t o)
{
    return nv_vram_read16(s, s->ramin_flip + o);
}
static inline uint32_t nv_ramin_read32(NVVGAState *s, uint32_t o)
{
    return nv_vram_read32(s, s->ramin_flip + o);
}
static inline void nv_ramin_write8(NVVGAState *s, uint32_t o, uint8_t v)
{
    nv_vram_write8(s, s->ramin_flip + o, v);
}
static inline void nv_ramin_write32(NVVGAState *s, uint32_t o, uint32_t v)
{
    nv_vram_write32(s, s->ramin_flip + o, v);
}

/* ======================================================================== */
/* System-RAM physical access via PCI DMA                                   */
/* ======================================================================== */

static uint8_t nv_phys_read8(NVVGAState *s, uint32_t a)
{
    uint8_t v = 0xff;
    pci_dma_read(&s->dev, a, &v, 1);
    return v;
}
static uint16_t nv_phys_read16(NVVGAState *s, uint32_t a)
{
    uint16_t v = 0xffff;
    pci_dma_read(&s->dev, a, &v, 2);
    return le16_to_cpu(v);
}
static uint32_t nv_phys_read32(NVVGAState *s, uint32_t a)
{
    uint32_t v = 0;
    pci_dma_read(&s->dev, a, &v, 4);
    return le32_to_cpu(v);
}
static uint64_t nv_phys_read64(NVVGAState *s, uint32_t a)
{
    return (uint64_t)nv_phys_read32(s, a) |
           ((uint64_t)nv_phys_read32(s, a + 4) << 32);
}
static void nv_phys_write8(NVVGAState *s, uint32_t a, uint8_t v)
{
    pci_dma_write(&s->dev, a, &v, 1);
}
static void nv_phys_write16(NVVGAState *s, uint32_t a, uint16_t v)
{
    uint16_t le = cpu_to_le16(v);
    pci_dma_write(&s->dev, a, &le, 2);
}
static void nv_phys_write32(NVVGAState *s, uint32_t a, uint32_t v)
{
    uint32_t le = cpu_to_le32(v);
    pci_dma_write(&s->dev, a, &le, 4);
}
G_GNUC_UNUSED static void nv_phys_write64(NVVGAState *s, uint32_t a, uint64_t v)
{
    nv_phys_write32(s, a, (uint32_t)v);
    nv_phys_write32(s, a + 4, (uint32_t)(v >> 32));
}

/* ======================================================================== */
/* DMA object translation (Bochs semantics)                                 */
/*   ramin word0 bit 0x2000  set → linear lookup, clear → page-table walk   */
/*   ramin word0 bit 0x20000 set → system RAM,    clear → VRAM              */
/* ======================================================================== */

static uint32_t dma_pt_lookup(NVVGAState *s, uint32_t object, uint32_t address)
{
    uint32_t address_adj = address + (nv_ramin_read32(s, object) >> 20);
    uint32_t pte = nv_ramin_read32(s, object + 8 + (address_adj >> 12) * 4);
    return (pte & 0xfffff000u) + (address_adj & 0xfffu);
}

static uint32_t dma_lin_lookup(NVVGAState *s, uint32_t object, uint32_t address)
{
    uint32_t adjust = nv_ramin_read32(s, object) >> 20;
    uint32_t base = nv_ramin_read32(s, object + 8) & 0xfffff000u;
    return base + adjust + address;
}

static uint32_t dma_addr(NVVGAState *s, uint32_t object, uint32_t address)
{
    uint32_t flags = nv_ramin_read32(s, object);
    return (flags & 0x00002000u) ? dma_lin_lookup(s, object, address)
                                 : dma_pt_lookup(s, object, address);
}

static uint8_t dma_read8(NVVGAState *s, uint32_t object, uint32_t address)
{
    uint32_t flags = nv_ramin_read32(s, object);
    uint32_t a = dma_addr(s, object, address);
    return (flags & 0x00020000u) ? nv_phys_read8(s, a)
                                 : nv_vram_read8(s, a);
}
static uint16_t dma_read16(NVVGAState *s, uint32_t object, uint32_t address)
{
    uint32_t flags = nv_ramin_read32(s, object);
    uint32_t a = dma_addr(s, object, address);
    return (flags & 0x00020000u) ? nv_phys_read16(s, a)
                                 : nv_vram_read16(s, a);
}
static uint32_t dma_read32(NVVGAState *s, uint32_t object, uint32_t address)
{
    uint32_t flags = nv_ramin_read32(s, object);
    uint32_t a = dma_addr(s, object, address);
    return (flags & 0x00020000u) ? nv_phys_read32(s, a)
                                 : nv_vram_read32(s, a);
}
static uint64_t dma_read64(NVVGAState *s, uint32_t object, uint32_t address)
{
    uint32_t flags = nv_ramin_read32(s, object);
    uint32_t a = dma_addr(s, object, address);
    return (flags & 0x00020000u) ? nv_phys_read64(s, a)
                                 : nv_vram_read64(s, a);
}
static void dma_write8(NVVGAState *s, uint32_t object, uint32_t address,
                       uint8_t value)
{
    uint32_t flags = nv_ramin_read32(s, object);
    uint32_t a = dma_addr(s, object, address);
    if (flags & 0x00020000u) {
        nv_phys_write8(s, a, value);
    } else {
        nv_vram_write8(s, a, value);
    }
}
static void dma_write16(NVVGAState *s, uint32_t object, uint32_t address,
                        uint16_t value)
{
    uint32_t flags = nv_ramin_read32(s, object);
    uint32_t a = dma_addr(s, object, address);
    if (flags & 0x00020000u) {
        nv_phys_write16(s, a, value);
    } else {
        nv_vram_write16(s, a, value);
    }
}
static void dma_write32(NVVGAState *s, uint32_t object, uint32_t address,
                        uint32_t value)
{
    uint32_t flags = nv_ramin_read32(s, object);
    uint32_t a = dma_addr(s, object, address);
    if (flags & 0x00020000u) {
        nv_phys_write32(s, a, value);
    } else {
        nv_vram_write32(s, a, value);
    }
}
static void dma_write64(NVVGAState *s, uint32_t object, uint32_t address,
                        uint64_t value)
{
    dma_write32(s, object, address, (uint32_t)value);
    dma_write32(s, object, address + 4, (uint32_t)(value >> 32));
}

static void dma_copy(NVVGAState *s, uint32_t dst_obj, uint32_t dst_addr,
                     uint32_t src_obj, uint32_t src_addr, uint32_t byte_count)
{
    uint32_t dst_flags = nv_ramin_read32(s, dst_obj);
    uint32_t src_flags = nv_ramin_read32(s, src_obj);
    uint8_t buffer[0x1000];
    uint32_t bytes_left = byte_count;

    while (bytes_left) {
        uint32_t dst_a = (dst_flags & 0x00002000u)
                         ? dma_lin_lookup(s, dst_obj, dst_addr)
                         : dma_pt_lookup(s, dst_obj, dst_addr);
        uint32_t src_a = (src_flags & 0x00002000u)
                         ? dma_lin_lookup(s, src_obj, src_addr)
                         : dma_pt_lookup(s, src_obj, src_addr);
        uint32_t chunk = bytes_left;
        if (chunk > 0x1000u - (dst_a & 0xfffu)) {
            chunk = 0x1000u - (dst_a & 0xfffu);
        }
        if (chunk > 0x1000u - (src_a & 0xfffu)) {
            chunk = 0x1000u - (src_a & 0xfffu);
        }
        if (src_flags & 0x00020000u) {
            pci_dma_read(&s->dev, src_a, buffer, chunk);
        } else {
            memcpy(buffer, s->vga.vram_ptr + (src_a & s->memsize_mask), chunk);
        }
        if (dst_flags & 0x00020000u) {
            pci_dma_write(&s->dev, dst_a, buffer, chunk);
        } else {
            memcpy(s->vga.vram_ptr + (dst_a & s->memsize_mask), buffer, chunk);
        }
        dst_addr += chunk;
        src_addr += chunk;
        bytes_left -= chunk;
    }
}

/* ======================================================================== */
/* RAMHT / RAMFC                                                            */
/* ======================================================================== */

static uint32_t ramfc_address(NVVGAState *s, uint32_t chid, uint32_t offset)
{
    uint32_t ramfc = (s->fifo_ramfc & 0xfff) << 8;
    uint32_t ch_size = (s->card_type < 0x20) ? 0x20 : 0x40;
    return ramfc + chid * ch_size + offset;
}
static uint32_t ramfc_read32(NVVGAState *s, uint32_t chid, uint32_t offset)
{
    return nv_ramin_read32(s, ramfc_address(s, chid, offset));
}
static void ramfc_write32(NVVGAState *s, uint32_t chid, uint32_t offset,
                          uint32_t value)
{
    nv_ramin_write32(s, ramfc_address(s, chid, offset), value);
}

static void ramht_lookup(NVVGAState *s, uint32_t handle, uint32_t chid,
                         uint32_t *object, uint8_t *engine)
{
    uint32_t ramht_addr = (s->fifo_ramht & 0xfff) << 8;
    uint32_t ramht_bits = ((s->fifo_ramht >> 16) & 0xff) + 9;
    uint32_t ramht_size = 1u << ramht_bits << 3;

    uint32_t hash = 0;
    uint32_t x = handle;
    while (x) {
        hash ^= (x & ((1u << ramht_bits) - 1));
        x >>= ramht_bits;
    }
    hash ^= (chid & 0xf) << (ramht_bits - 4);
    hash = hash << 3;

    uint32_t it = hash;
    do {
        if (nv_ramin_read32(s, ramht_addr + it) == handle) {
            uint32_t context = nv_ramin_read32(s, ramht_addr + it + 4);
            uint32_t ctx_chid = (context >> 24) & 0x1f;
            if (chid == ctx_chid) {
                if (object) {
                    *object = (context & 0xffff) << 4;
                }
                if (engine) {
                    *engine = (context >> 16) & 0xff;
                }
                return;
            }
        }
        it += 8;
        if (it >= ramht_size) {
            it = 0;
        }
    } while (it != hash);

    qemu_log_mask(LOG_GUEST_ERROR, "geforce: ramht_lookup failed for 0x%08x\n",
                  handle);
}

/* ======================================================================== */
/* Timer / IRQ                                                              */
/* ======================================================================== */

static uint64_t nv_get_time(NVVGAState *s)
{
    uint64_t ns = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    if (s->timer_den != 0 && s->timer_num != 0) {
        return (ns / 74u) * s->timer_num / s->timer_den;
    }
    return ns >> 5;
}

static uint32_t get_mc_intr(NVVGAState *s)
{
    uint32_t v = 0;
    if (s->bus_intr   & s->bus_intr_en)   v |= 0x10000000u;
    if (s->fifo_intr  & s->fifo_intr_en)  v |= 0x00000100u;
    if (s->graph_intr & s->graph_intr_en) v |= 0x00001000u;
    if (s->timer_intr & s->timer_intr_en) v |= 0x00100000u;
    if (s->crtc_intr  & s->crtc_intr_en)  v |= 0x01000000u;
    return v;
}

static void update_irq_level(NVVGAState *s)
{
    bool asserted = (get_mc_intr(s) && (s->mc_intr_en & 1u)) ||
                    (s->mc_soft_intr && (s->mc_intr_en & 2u));
    pci_set_irq(&s->dev, asserted ? 1 : 0);
}

/* ======================================================================== */
/* Display dirty-region notification                                        */
/*                                                                          */
/* Bochs' redraw_area_nd(offset, width, height) marks a rectangle of the    */
/* scanout surface dirty.  In QEMU we translate the affected VRAM byte span */
/* into a memory_region_set_dirty() call so the VGA display path repaints.  */
/* `offset` is relative to the scanout base (disp_offset already removed by  */
/* the callers, matching Bochs).                                            */
/* ======================================================================== */

static uint32_t nv_bytes_per_pixel(NVVGAState *s)
{
    switch (s->crtc_reg[0x28] & 0x7f) {
    case 2:  return 2;
    case 3:  return 4;
    default: return 1;
    }
}

static void nv_redraw_nd(NVVGAState *s, uint32_t offset,
                         uint32_t width, uint32_t height)
{
    uint32_t bpp = nv_bytes_per_pixel(s);
    uint64_t base = (uint64_t)s->disp_offset + offset;
    uint64_t span = (uint64_t)width * height * bpp;

    if (!span) {
        return;
    }
    if (base >= s->vga.vram_size) {
        return;
    }
    if (base + span > s->vga.vram_size) {
        span = s->vga.vram_size - base;
    }
    memory_region_set_dirty(&s->vga.vram, base, span);
}

/* ======================================================================== */
/* Pixel access + raster operations                                         */
/* ======================================================================== */

static uint32_t get_pixel(NVVGAState *s, uint32_t obj, uint32_t ofs,
                          uint32_t x, uint32_t cb)
{
    if (cb == 1) {
        return dma_read8(s, obj, ofs + x);
    } else if (cb == 2) {
        return dma_read16(s, obj, ofs + x * 2);
    } else {
        return dma_read32(s, obj, ofs + x * 4);
    }
}

static void put_pixel(NVVGAState *s, NvChannel *ch, uint32_t ofs,
                      uint32_t x, uint32_t value)
{
    if (ch->s2d_color_bytes == 1) {
        dma_write8(s, ch->s2d_img_dst, ofs + x, value);
    } else if (ch->s2d_color_bytes == 2) {
        dma_write16(s, ch->s2d_img_dst, ofs + x * 2, value);
    } else if (ch->s2d_color_fmt == 6) {
        dma_write32(s, ch->s2d_img_dst, ofs + x * 4, value & 0x00ffffff);
    } else {
        dma_write32(s, ch->s2d_img_dst, ofs + x * 4, value);
    }
}

static void put_pixel_swzs(NVVGAState *s, NvChannel *ch, uint32_t ofs,
                           uint32_t value)
{
    if (ch->swzs_color_bytes == 1) {
        dma_write8(s, ch->swzs_img_obj, ofs, value);
    } else if (ch->swzs_color_bytes == 2) {
        dma_write16(s, ch->swzs_img_obj, ofs, value);
    } else {
        dma_write32(s, ch->swzs_img_obj, ofs, value);
    }
}

static void pixel_operation(NVVGAState *s, NvChannel *ch, uint32_t op,
                            uint32_t *dstcolor, const uint32_t *srccolor,
                            uint32_t cb, uint32_t px, uint32_t py)
{
    if (op == 1) {
        uint8_t rop = ch->rop;
        uint32_t i = (py % 8) * 8 + (px % 8);
        uint32_t patt_color;
        if (ch->patt_type_color) {
            patt_color = ch->patt_data_color[i];
        } else {
            patt_color = ch->patt_data_mono[i] ? ch->patt_fg_color
                                               : ch->patt_bg_color;
        }
        *dstcolor = rop3(rop, *dstcolor, *srccolor, patt_color, cb);
    } else if (op == 5) {
        if (cb == 4) {
            if (*srccolor) {
                uint8_t sb = *srccolor;
                uint8_t sg = *srccolor >> 8;
                uint8_t sr = *srccolor >> 16;
                uint8_t sa = *srccolor >> 24;
                uint32_t beta = ch->beta;
                if (beta != 0xffffffff) {
                    uint8_t bb = beta;
                    uint8_t bg = beta >> 8;
                    uint8_t br = beta >> 16;
                    uint8_t ba = beta >> 24;
                    sb = sb * bb / 0xff;
                    sg = sg * bg / 0xff;
                    sr = sr * br / 0xff;
                    sa = sa * ba / 0xff;
                }
                uint8_t db = *dstcolor;
                uint8_t dg = *dstcolor >> 8;
                uint8_t dr = *dstcolor >> 16;
                uint8_t da = *dstcolor >> 24;
                uint8_t isa = 0xff - sa;
                uint8_t b = alpha_wrap(db * isa / 0xff + sb);
                uint8_t g = alpha_wrap(dg * isa / 0xff + sg);
                uint8_t r = alpha_wrap(dr * isa / 0xff + sr);
                uint8_t a = alpha_wrap(da * isa / 0xff + sa);
                *dstcolor = b | (g << 8) | (r << 16) | (a << 24);
            }
        } else {
            uint32_t beta = ch->beta;
            uint8_t bb = beta;
            uint8_t bg = beta >> 8;
            uint8_t br = beta >> 16;
            uint8_t iba = 0xff - (beta >> 24);
            uint8_t sb = *srccolor & 0x1f;
            uint8_t sg = (*srccolor >> 5) & 0x3f;
            uint8_t sr = (*srccolor >> 11) & 0x1f;
            uint8_t db = *dstcolor & 0x1f;
            uint8_t dg = (*dstcolor >> 5) & 0x3f;
            uint8_t dr = (*dstcolor >> 11) & 0x1f;
            uint8_t b = (db * iba + sb * bb) / 0xff;
            uint8_t g = (dg * iba + sg * bg) / 0xff;
            uint8_t r = (dr * iba + sr * br) / 0xff;
            *dstcolor = b | (g << 5) | (r << 11);
        }
    } else {
        *dstcolor = *srccolor;
    }
}

/* ======================================================================== */
/* Swizzle address generator (for swizzled surfaces)                        */
/* ======================================================================== */

static uint32_t swizzle(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
{
    bool xleft = true;
    bool yleft = height != 1;
    uint32_t xbit = 1, ybit = 1, rbit = 1, r = 0;
    do {
        if (xleft) {
            if ((x & xbit) != 0) {
                r |= rbit;
            }
            rbit <<= 1;
            xbit <<= 1;
            xleft = xbit < width;
        }
        if (yleft) {
            if ((y & ybit) != 0) {
                r |= rbit;
            }
            rbit <<= 1;
            ybit <<= 1;
            yleft = ybit < height;
        }
    } while (xleft || yleft);
    return r;
}

/* ======================================================================== */
/* 2D primitives (ported from Bochs)                                        */
/* ======================================================================== */

static void gdi_fillrect(NVVGAState *s, NvChannel *ch, bool clipped)
{
    int16_t clipx0 = 0, clipy0 = 0, clipx1 = 0, clipy1 = 0;
    if (clipped) {
        clipx0 = ch->gdi_clip_yx0 & 0xffff;
        clipy0 = ch->gdi_clip_yx0 >> 16;
        clipx1 = ch->gdi_clip_yx1 & 0xffff;
        clipy1 = ch->gdi_clip_yx1 >> 16;
    }
    int16_t dx, dy;
    if (clipped) {
        dx = ch->gdi_rect_yx0 & 0xffff;
        dy = ch->gdi_rect_yx0 >> 16;
        clipx0 -= dx;
        clipy0 -= dy;
        clipx1 -= dx;
        clipy1 -= dy;
    } else {
        dx = ch->gdi_rect_xy >> 16;
        dy = ch->gdi_rect_xy & 0xffff;
    }
    uint16_t width, height;
    if (clipped) {
        width = (ch->gdi_rect_yx1 & 0xffff) - dx;
        height = (ch->gdi_rect_yx1 >> 16) - dy;
    } else {
        width = ch->gdi_rect_wh >> 16;
        height = ch->gdi_rect_wh & 0xffff;
    }
    uint32_t pitch = ch->s2d_pitch_dst;
    uint32_t srccolor = ch->gdi_rect_color;
    uint32_t draw_offset = ch->s2d_ofs_dst + dy * pitch + dx * ch->s2d_color_bytes;
    uint32_t redraw_offset = dma_lin_lookup(s, ch->s2d_img_dst, draw_offset) -
                             s->disp_offset;
    for (uint16_t y = 0; y < height; y++) {
        for (uint16_t x = 0; x < width; x++) {
            if (!clipped || (x >= clipx0 && x < clipx1 &&
                             y >= clipy0 && y < clipy1)) {
                uint32_t dstcolor = get_pixel(s, ch->s2d_img_dst, draw_offset,
                                              x, ch->s2d_color_bytes);
                pixel_operation(s, ch, ch->gdi_operation, &dstcolor, &srccolor,
                                ch->s2d_color_bytes, dx + x, dy + y);
                put_pixel(s, ch, draw_offset, x, dstcolor);
            }
        }
        draw_offset += pitch;
    }
    nv_redraw_nd(s, redraw_offset, width, height);
}

static void gdi_blit(NVVGAState *s, NvChannel *ch, uint32_t type)
{
    int16_t dx = ch->gdi_image_xy & 0xffff;
    int16_t dy = ch->gdi_image_xy >> 16;
    int16_t clipx0 = (ch->gdi_clip_yx0 & 0xffff) - dx;
    int16_t clipy0 = (ch->gdi_clip_yx0 >> 16) - dy;
    int16_t clipx1 = (ch->gdi_clip_yx1 & 0xffff) - dx;
    int16_t clipy1 = (ch->gdi_clip_yx1 >> 16) - dy;
    uint32_t swidth = ch->gdi_image_swh & 0xffff;
    uint32_t dwidth = type ? ch->gdi_image_dwh & 0xffff : swidth;
    uint32_t height = ch->gdi_image_swh >> 16;
    uint32_t pitch = ch->s2d_pitch_dst;
    uint32_t bg_color = ch->gdi_bg_color;
    uint32_t fg_color = ch->gdi_fg_color;
    if (ch->s2d_color_bytes == 4 && ch->gdi_color_fmt != 3) {
        bg_color = color_565_to_888(bg_color);
        fg_color = color_565_to_888(fg_color);
    }
    uint32_t draw_offset = ch->s2d_ofs_dst + dy * pitch + dx * ch->s2d_color_bytes;
    uint32_t redraw_offset = dma_lin_lookup(s, ch->s2d_img_dst, draw_offset) -
                             s->disp_offset;
    uint32_t bit_index = 0;
    for (uint16_t y = 0; y < height; y++) {
        for (uint16_t x = 0; x < dwidth; x++) {
            if (x >= clipx0 && x < clipx1 && y >= clipy0 && y < clipy1) {
                uint32_t word_offset = bit_index / 32;
                uint32_t bit_offset = bit_index % 32;
                if (ch->gdi_mono_fmt == 1) {
                    bit_offset ^= 7;
                }
                bool pixel = (ch->gdi_words[word_offset] >> bit_offset) & 1;
                if (type || pixel) {
                    uint32_t dstcolor = get_pixel(s, ch->s2d_img_dst, draw_offset,
                                                  x, ch->s2d_color_bytes);
                    uint32_t srccolor = pixel ? fg_color : bg_color;
                    pixel_operation(s, ch, ch->gdi_operation, &dstcolor, &srccolor,
                                    ch->s2d_color_bytes, dx + x, dy + y);
                    put_pixel(s, ch, draw_offset, x, dstcolor);
                }
            }
            bit_index++;
        }
        bit_index += swidth - dwidth;
        draw_offset += pitch;
    }
    nv_redraw_nd(s, redraw_offset, dwidth, height);
}

static void nv_rect(NVVGAState *s, NvChannel *ch)
{
    int16_t dx = ch->rect_yx & 0xffff;
    int16_t dy = ch->rect_yx >> 16;
    uint16_t width = ch->rect_hw & 0xffff;
    uint16_t height = ch->rect_hw >> 16;
    uint32_t pitch = ch->s2d_pitch_dst;
    uint32_t srccolor = ch->rect_color;
    uint32_t draw_offset = ch->s2d_ofs_dst + dy * pitch + dx * ch->s2d_color_bytes;
    uint32_t redraw_offset = dma_lin_lookup(s, ch->s2d_img_dst, draw_offset) -
                             s->disp_offset;
    for (uint16_t y = 0; y < height; y++) {
        for (uint16_t x = 0; x < width; x++) {
            uint32_t dstcolor = get_pixel(s, ch->s2d_img_dst, draw_offset,
                                          x, ch->s2d_color_bytes);
            pixel_operation(s, ch, ch->rect_operation, &dstcolor, &srccolor,
                            ch->s2d_color_bytes, dx + x, dy + y);
            put_pixel(s, ch, draw_offset, x, dstcolor);
        }
        draw_offset += pitch;
    }
    nv_redraw_nd(s, redraw_offset, width, height);
}

static void nv_ifc(NVVGAState *s, NvChannel *ch, uint32_t word)
{
    uint32_t chromacolor = 0;
    bool chroma_enabled = false;
    if (ch->ifc_color_key_enable) {
        if (ch->ifc_color_bytes == 4) {
            chromacolor = ch->chroma_color & 0x00ffffff;
            chroma_enabled = ch->chroma_color & 0xff000000;
        } else if (ch->ifc_color_bytes == 2) {
            chromacolor = ch->chroma_color & 0x0000ffff;
            chroma_enabled = ch->chroma_color & 0xffff0000;
        } else {
            chromacolor = ch->chroma_color & 0x000000ff;
            chroma_enabled = ch->chroma_color & 0xffffff00;
        }
    }
    for (uint32_t i = 0; i < ch->ifc_pixels_per_word; i++) {
        if (ch->ifc_x >= ch->ifc_clip_x0 && ch->ifc_x < ch->ifc_clip_x1 &&
            ch->ifc_y >= ch->ifc_clip_y0 && ch->ifc_y < ch->ifc_clip_y1) {
            uint32_t srccolor;
            if (ch->ifc_color_bytes == 4) {
                srccolor = word;
            } else if (ch->ifc_color_bytes == 2) {
                srccolor = i == 0 ? word & 0xffff : word >> 16;
            } else {
                srccolor = (word >> (i * 8)) & 0xff;
            }
            if (!chroma_enabled || srccolor != chromacolor) {
                uint32_t dstcolor = get_pixel(s, ch->s2d_img_dst,
                                              ch->ifc_draw_offset, ch->ifc_x,
                                              ch->s2d_color_bytes);
                if (ch->ifc_color_bytes == 4 && ch->s2d_color_bytes == 2) {
                    dstcolor = color_565_to_888(dstcolor);
                }
                pixel_operation(s, ch, ch->ifc_operation, &dstcolor, &srccolor,
                                ch->ifc_color_bytes,
                                ch->ifc_ofs_x + ch->ifc_x,
                                ch->ifc_ofs_y + ch->ifc_y);
                if (ch->ifc_color_bytes == 4 && ch->s2d_color_bytes == 2) {
                    dstcolor = color_888_to_565(dstcolor);
                }
                put_pixel(s, ch, ch->ifc_draw_offset, ch->ifc_x, dstcolor);
            }
        }
        ch->ifc_x++;
        if (ch->ifc_x >= ch->ifc_src_width) {
            nv_redraw_nd(s, ch->ifc_redraw_offset, ch->ifc_dst_width, 1);
            ch->ifc_draw_offset += ch->s2d_pitch_dst;
            ch->ifc_redraw_offset += ch->s2d_pitch_dst;
            ch->ifc_x = 0;
            ch->ifc_y++;
        }
    }
}

static void nv_iifc(NVVGAState *s, NvChannel *ch)
{
    int16_t dx = ch->iifc_yx & 0xffff;
    int16_t dy = ch->iifc_yx >> 16;
    int16_t clipx0 = ch->clip_x - dx;
    int16_t clipy0 = ch->clip_y - dy;
    int16_t clipx1 = clipx0 + ch->clip_width;
    int16_t clipy1 = clipy0 + ch->clip_height;
    uint32_t swidth = ch->iifc_shw & 0xffff;
    uint32_t dwidth = ch->iifc_dhw & 0xffff;
    uint32_t height = ch->iifc_dhw >> 16;
    uint32_t pitch = ch->s2d_pitch_dst;
    uint32_t draw_offset = ch->s2d_ofs_dst + dy * pitch + dx * ch->s2d_color_bytes;
    uint32_t redraw_offset = dma_lin_lookup(s, ch->s2d_img_dst, draw_offset) -
                             s->disp_offset;
    uint32_t symbol_index = 0;
    for (uint16_t y = 0; y < height; y++) {
        for (uint16_t x = 0; x < dwidth; x++) {
            if (x >= clipx0 && x < clipx1 && y >= clipy0 && y < clipy1) {
                uint8_t symbol;
                if (ch->iifc_bpp4) {
                    uint32_t word_offset = symbol_index / 8;
                    uint32_t symbol_offset = (symbol_index % 8 ^ 1) * 4;
                    symbol = ch->iifc_words[word_offset] >> symbol_offset & 0xf;
                } else {
                    uint32_t word_offset = symbol_index / 4;
                    uint32_t symbol_offset = symbol_index % 4 * 8;
                    symbol = ch->iifc_words[word_offset] >> symbol_offset & 0xff;
                }
                uint32_t dstcolor = get_pixel(s, ch->s2d_img_dst, draw_offset,
                                              x, ch->s2d_color_bytes);
                if (ch->iifc_color_bytes == 4) {
                    uint32_t srccolor = dma_read32(s, ch->iifc_palette,
                                                   ch->iifc_palette_ofs + symbol * 4);
                    if (ch->s2d_color_bytes == 2) {
                        dstcolor = color_565_to_888(dstcolor);
                    }
                    pixel_operation(s, ch, ch->iifc_operation, &dstcolor,
                                    &srccolor, 4, dx + x, dy + y);
                    if (ch->s2d_color_bytes == 2) {
                        dstcolor = color_888_to_565(dstcolor);
                    }
                } else if (ch->iifc_color_bytes == 2) {
                    uint32_t srccolor = dma_read16(s, ch->iifc_palette,
                                                   ch->iifc_palette_ofs + symbol * 2);
                    pixel_operation(s, ch, ch->iifc_operation, &dstcolor,
                                    &srccolor, 2, dx + x, dy + y);
                }
                put_pixel(s, ch, draw_offset, x, dstcolor);
            }
            symbol_index++;
        }
        symbol_index += swidth - dwidth;
        draw_offset += pitch;
    }
    nv_redraw_nd(s, redraw_offset, dwidth, height);
}

static void nv_sifc(NVVGAState *s, NvChannel *ch)
{
    uint16_t dx = ch->sifc_clip_yx & 0xffff;
    uint16_t dy = ch->sifc_clip_yx >> 16;
    uint32_t dsdx = (uint32_t)(1099511627776ULL / ch->sifc_dxds);
    uint32_t dtdy = (uint32_t)(1099511627776ULL / ch->sifc_dydt);
    uint32_t swidth = ch->sifc_shw & 0xffff;
    uint32_t dwidth = ch->sifc_clip_hw & 0xffff;
    uint32_t height = ch->sifc_clip_hw >> 16;
    uint32_t pitch = ch->s2d_pitch_dst;
    uint32_t draw_offset = ch->s2d_ofs_dst + dy * pitch + dx * ch->s2d_color_bytes;
    uint32_t redraw_offset = dma_lin_lookup(s, ch->s2d_img_dst, draw_offset) -
                             s->disp_offset;
    int32_t sx0 = ((ch->sifc_syx & 0xffff) << 16) - (dx << 20) - 0x80000;
    int32_t sy = (ch->sifc_syx & 0xffff0000) - (dy << 20) - 0x80000;
    if (sx0 < 0) {
        sx0 = 0;
    }
    if (sy < 0) {
        sy = 0;
    }
    uint32_t symbol_offset_y = 0;
    for (uint16_t y = 0; y < height; y++) {
        uint32_t sx = sx0;
        for (uint16_t x = 0; x < dwidth; x++) {
            uint32_t dstcolor = get_pixel(s, ch->s2d_img_dst, draw_offset,
                                          x, ch->s2d_color_bytes);
            uint32_t srccolor;
            uint32_t symbol_offset = symbol_offset_y + (sx >> 20);
            if (ch->sifc_color_bytes == 4) {
                srccolor = ch->sifc_words[symbol_offset];
            } else if (ch->sifc_color_bytes == 2) {
                srccolor = ((uint16_t *)ch->sifc_words)[symbol_offset];
            } else {
                srccolor = ((uint8_t *)ch->sifc_words)[symbol_offset];
            }
            if (ch->sifc_color_bytes == 4 && ch->s2d_color_bytes == 2) {
                dstcolor = color_565_to_888(dstcolor);
            }
            pixel_operation(s, ch, ch->sifc_operation, &dstcolor, &srccolor,
                            ch->sifc_color_bytes, dx + x, dy + y);
            if (ch->sifc_color_bytes == 4 && ch->s2d_color_bytes == 2) {
                dstcolor = color_888_to_565(dstcolor);
            }
            put_pixel(s, ch, draw_offset, x, dstcolor);
            sx += dsdx;
        }
        sy += dtdy;
        symbol_offset_y = (sy >> 20) * swidth;
        draw_offset += pitch;
    }
    nv_redraw_nd(s, redraw_offset, dwidth, height);
}

static void nv_copyarea(NVVGAState *s, NvChannel *ch)
{
    uint16_t sx = ch->blit_syx & 0xffff;
    uint16_t sy = ch->blit_syx >> 16;
    uint16_t dx = ch->blit_dyx & 0xffff;
    uint16_t dy = ch->blit_dyx >> 16;
    uint16_t width = ch->blit_hw & 0xffff;
    uint16_t height = ch->blit_hw >> 16;
    uint32_t spitch = ch->s2d_pitch_src;
    uint32_t dpitch = ch->s2d_pitch_dst;
    uint32_t src_offset = ch->s2d_ofs_src;
    uint32_t draw_offset = ch->s2d_ofs_dst;
    bool xdir = dx > sx;
    bool ydir = dy > sy;
    src_offset += (sy + ydir * (height - 1)) * spitch + sx * ch->s2d_color_bytes;
    uint32_t redraw_offset = dma_lin_lookup(s, ch->s2d_img_dst, draw_offset) +
                             dy * dpitch + dx * ch->s2d_color_bytes - s->disp_offset;
    draw_offset += (dy + ydir * (height - 1)) * dpitch + dx * ch->s2d_color_bytes;
    uint32_t chromacolor = 0;
    bool chroma_enabled = false;
    if (ch->blit_color_key_enable) {
        if (ch->s2d_color_bytes == 4) {
            chromacolor = ch->chroma_color & 0x00ffffff;
            chroma_enabled = ch->chroma_color & 0xff000000;
        } else if (ch->s2d_color_bytes == 2) {
            chromacolor = ch->chroma_color & 0x0000ffff;
            chroma_enabled = ch->chroma_color & 0xffff0000;
        } else {
            chromacolor = ch->chroma_color & 0x000000ff;
            chroma_enabled = ch->chroma_color & 0xffffff00;
        }
    }
    for (uint16_t y = 0; y < height; y++) {
        for (uint16_t x = 0; x < width; x++) {
            uint16_t xa = xdir ? width - x - 1 : x;
            uint32_t srccolor = get_pixel(s, ch->s2d_img_src, src_offset, xa,
                                          ch->s2d_color_bytes);
            if (!chroma_enabled || srccolor != chromacolor) {
                uint32_t dstcolor = get_pixel(s, ch->s2d_img_dst, draw_offset,
                                              xa, ch->s2d_color_bytes);
                pixel_operation(s, ch, ch->blit_operation, &dstcolor, &srccolor,
                                ch->s2d_color_bytes, dx + x, dy + y);
                put_pixel(s, ch, draw_offset, xa, dstcolor);
            }
        }
        src_offset += spitch * (1 - 2 * ydir);
        draw_offset += dpitch * (1 - 2 * ydir);
    }
    nv_redraw_nd(s, redraw_offset, width, height);
}

static void nv_m2mf(NVVGAState *s, NvChannel *ch)
{
    uint32_t src_offset = ch->m2mf_src_offset;
    uint32_t dst_offset = ch->m2mf_dst_offset;
    for (uint16_t y = 0; y < ch->m2mf_line_count; y++) {
        dma_copy(s, ch->m2mf_dst, dst_offset, ch->m2mf_src, src_offset,
                 ch->m2mf_line_length);
        src_offset += ch->m2mf_src_pitch;
        dst_offset += ch->m2mf_dst_pitch;
    }
    uint32_t dma_target = nv_ramin_read32(s, ch->m2mf_dst) >> 12 & 0xff;
    if (dma_target == 0x03 || dma_target == 0x0b) {
        uint32_t redraw_offset = dma_lin_lookup(s, ch->m2mf_dst,
                                                ch->m2mf_dst_offset) -
                                 s->disp_offset;
        uint32_t bpp = nv_bytes_per_pixel(s);
        uint32_t width = ch->m2mf_line_length / bpp;
        nv_redraw_nd(s, redraw_offset, width, ch->m2mf_line_count);
    }
}

static void nv_tfc(NVVGAState *s, NvChannel *ch)
{
    uint16_t dx = ch->tfc_yx & 0xffff;
    uint16_t dy = ch->tfc_yx >> 16;
    int16_t clipx0 = (ch->tfc_clip_wx & 0xffff) - dx;
    int16_t clipy0 = (ch->tfc_clip_hy & 0xffff) - dy;
    int16_t clipx1 = clipx0 + (ch->tfc_clip_wx >> 16);
    int16_t clipy1 = clipy0 + (ch->tfc_clip_hy >> 16);
    uint32_t width = ch->tfc_hw & 0xffff;
    uint32_t height = ch->tfc_hw >> 16;
    uint32_t word_offset = 0;
    if (ch->tfc_swizzled) {
        for (uint16_t y = 0; y < height; y++) {
            for (uint16_t x = 0; x < width; x++) {
                if (x >= clipx0 && x < clipx1 && y >= clipy0 && y < clipy1) {
                    uint32_t srccolor;
                    if (ch->tfc_color_bytes == 4) {
                        srccolor = ch->tfc_words[word_offset];
                    } else if (ch->tfc_color_bytes == 2) {
                        srccolor = ((uint16_t *)ch->tfc_words)[word_offset];
                    } else {
                        srccolor = ((uint8_t *)ch->tfc_words)[word_offset];
                    }
                    put_pixel_swzs(s, ch, ch->swzs_ofs +
                        swizzle(x + dx, y + dy, ch->swzs_width, ch->swzs_height) *
                        ch->swzs_color_bytes, srccolor);
                }
                word_offset++;
            }
        }
    } else {
        uint32_t pitch = ch->s2d_pitch_dst;
        uint32_t draw_offset = ch->s2d_ofs_dst + dy * pitch +
                               dx * ch->s2d_color_bytes;
        for (uint16_t y = 0; y < height; y++) {
            for (uint16_t x = 0; x < width; x++) {
                if (x >= clipx0 && x < clipx1 && y >= clipy0 && y < clipy1) {
                    uint32_t srccolor;
                    if (ch->tfc_color_bytes == 4) {
                        srccolor = ch->tfc_words[word_offset];
                    } else if (ch->tfc_color_bytes == 2) {
                        srccolor = ((uint16_t *)ch->tfc_words)[word_offset];
                    } else {
                        srccolor = ((uint8_t *)ch->tfc_words)[word_offset];
                    }
                    put_pixel(s, ch, draw_offset, x, srccolor);
                }
                word_offset++;
            }
            draw_offset += pitch;
        }
    }
}

static void nv_sifm(NVVGAState *s, NvChannel *ch, bool swizzled)
{
    uint16_t dx = ch->sifm_dyx & 0xffff;
    uint16_t dy = ch->sifm_dyx >> 16;
    uint16_t dwidth = ch->sifm_dhw & 0xffff;
    uint16_t dheight = ch->sifm_dhw >> 16;
    uint32_t spitch = ch->sifm_sfmt & 0xffff;

    if (ch->sifm_dudx == 0x00100000 && ch->sifm_dvdy == 0x00100000) {
        uint16_t sx = (ch->sifm_syx & 0xffff) >> 4;
        uint16_t sy = (ch->sifm_syx >> 16) >> 4;
        uint32_t src_offset = ch->sifm_sofs + sy * spitch + sx * ch->sifm_color_bytes;
        if (swizzled) {
            for (uint16_t y = 0; y < dheight; y++) {
                for (uint16_t x = 0; x < dwidth; x++) {
                    uint32_t srccolor = get_pixel(s, ch->sifm_src, src_offset,
                                                  x, ch->sifm_color_bytes);
                    put_pixel_swzs(s, ch, ch->swzs_ofs +
                        swizzle(x + dx, y + dy, ch->swzs_width, ch->swzs_height) *
                        ch->swzs_color_bytes, srccolor);
                }
                src_offset += spitch;
            }
        } else {
            uint32_t dpitch = ch->s2d_pitch_dst;
            uint32_t draw_offset = ch->s2d_ofs_dst + dy * dpitch +
                                   dx * ch->s2d_color_bytes;
            uint32_t redraw_offset = dma_lin_lookup(s, ch->s2d_img_dst,
                                                    draw_offset) - s->disp_offset;
            for (uint16_t y = 0; y < dheight; y++) {
                for (uint16_t x = 0; x < dwidth; x++) {
                    uint32_t dstcolor = get_pixel(s, ch->s2d_img_dst, draw_offset,
                                                  x, ch->s2d_color_bytes);
                    uint32_t srccolor = get_pixel(s, ch->sifm_src, src_offset,
                                                  x, ch->sifm_color_bytes);
                    if (ch->sifm_color_fmt == 4) {
                        srccolor |= 0xff000000;
                    }
                    pixel_operation(s, ch, ch->sifm_operation, &dstcolor,
                                    &srccolor, ch->s2d_color_bytes, dx + x, dy + y);
                    put_pixel(s, ch, draw_offset, x, dstcolor);
                }
                src_offset += spitch;
                draw_offset += dpitch;
            }
            nv_redraw_nd(s, redraw_offset, dwidth, dheight);
        }
    } else {
        int32_t sx0 = ((ch->sifm_syx & 0xffff) << 16) - 0x80000;
        int32_t sy = (ch->sifm_syx & 0xffff0000) - 0x80000;
        if (sx0 < 0) {
            sx0 = 0;
        }
        if (sy < 0) {
            sy = 0;
        }
        if (swizzled) {
            for (uint16_t y = 0; y < dheight; y++) {
                uint32_t sx = sx0;
                uint32_t src_offset = ch->sifm_sofs + (sy >> 20) * spitch;
                for (uint16_t x = 0; x < dwidth; x++) {
                    uint32_t srccolor = get_pixel(s, ch->sifm_src, src_offset,
                                                  sx >> 20, ch->sifm_color_bytes);
                    put_pixel_swzs(s, ch, ch->swzs_ofs +
                        swizzle(x + dx, y + dy, ch->swzs_width, ch->swzs_height) *
                        ch->swzs_color_bytes, srccolor);
                    sx += ch->sifm_dudx;
                }
                sy += ch->sifm_dvdy;
            }
        } else {
            uint32_t dpitch = ch->s2d_pitch_dst;
            uint32_t draw_offset = ch->s2d_ofs_dst + dy * dpitch +
                                   dx * ch->s2d_color_bytes;
            uint32_t redraw_offset = dma_lin_lookup(s, ch->s2d_img_dst,
                                                    draw_offset) - s->disp_offset;
            for (uint16_t y = 0; y < dheight; y++) {
                uint32_t sx = sx0;
                uint32_t src_offset = ch->sifm_sofs + (sy >> 20) * spitch;
                for (uint16_t x = 0; x < dwidth; x++) {
                    uint32_t dstcolor = get_pixel(s, ch->s2d_img_dst, draw_offset,
                                                  x, ch->s2d_color_bytes);
                    uint32_t srccolor = get_pixel(s, ch->sifm_src, src_offset,
                                                  sx >> 20, ch->sifm_color_bytes);
                    if (ch->sifm_color_fmt == 4) {
                        srccolor |= 0xff000000;
                    }
                    pixel_operation(s, ch, ch->sifm_operation, &dstcolor,
                                    &srccolor, ch->s2d_color_bytes, dx + x, dy + y);
                    put_pixel(s, ch, draw_offset, x, dstcolor);
                    sx += ch->sifm_dudx;
                }
                sy += ch->sifm_dvdy;
                draw_offset += dpitch;
            }
            nv_redraw_nd(s, redraw_offset, dwidth, dheight);
        }
    }
}

/* ======================================================================== */
/* Colour-format → bytes-per-pixel resolution                               */
/* ======================================================================== */

static void update_color_bytes(uint32_t s2d_color_fmt, uint32_t color_fmt,
                               uint32_t *color_bytes)
{
    if (s2d_color_fmt == 1) {
        *color_bytes = 1; /* Y8 hack */
    } else if (color_fmt == 1 || color_fmt == 2 || color_fmt == 3) {
        *color_bytes = 2;
    } else if (color_fmt == 4 || color_fmt == 5) {
        *color_bytes = 4;
    } else {
        qemu_log_mask(LOG_GUEST_ERROR, "geforce: unknown color format 0x%02x\n",
                      color_fmt);
    }
}

static void update_color_bytes_s2d(NvChannel *ch)
{
    if (ch->s2d_color_fmt == 0x1) {
        ch->s2d_color_bytes = 1;
    } else if (ch->s2d_color_fmt == 0x2 || ch->s2d_color_fmt == 0x4 ||
               ch->s2d_color_fmt == 0x5) {
        ch->s2d_color_bytes = 2;
    } else if (ch->s2d_color_fmt == 0x6 || ch->s2d_color_fmt == 0x7 ||
               ch->s2d_color_fmt == 0xA || ch->s2d_color_fmt == 0xB) {
        ch->s2d_color_bytes = 4;
    } else {
        qemu_log_mask(LOG_GUEST_ERROR,
                      "geforce: unknown 2d surface color format 0x%02x\n",
                      ch->s2d_color_fmt);
    }
}

static void update_color_bytes_ifc(NvChannel *ch)
{
    update_color_bytes(ch->s2d_color_fmt, ch->ifc_color_fmt, &ch->ifc_color_bytes);
}
static void update_color_bytes_sifc(NvChannel *ch)
{
    update_color_bytes(ch->s2d_color_fmt, ch->sifc_color_fmt, &ch->sifc_color_bytes);
}
static void update_color_bytes_tfc(NvChannel *ch)
{
    update_color_bytes(ch->s2d_color_fmt, ch->tfc_color_fmt, &ch->tfc_color_bytes);
}
static void update_color_bytes_iifc(NvChannel *ch)
{
    update_color_bytes(0, ch->iifc_color_fmt, &ch->iifc_color_bytes);
}

/* ======================================================================== */
/* Method dispatchers (execute_*)                                           */
/* ======================================================================== */

static void execute_clip(NvChannel *ch, uint32_t method, uint32_t param)
{
    if (method == 0x0c0) {
        ch->clip_x = (uint16_t)param;
        ch->clip_y = param >> 16;
    } else if (method == 0x0c1) {
        ch->clip_width = (uint16_t)param;
        ch->clip_height = param >> 16;
    }
}

static void execute_m2mf(NVVGAState *s, NvChannel *ch, uint32_t subc,
                         uint32_t method, uint32_t param)
{
    if (method == 0x061) {
        ch->m2mf_src = param;
    } else if (method == 0x062) {
        ch->m2mf_dst = param;
    } else if (method == 0x0c3) {
        ch->m2mf_src_offset = param;
    } else if (method == 0x0c4) {
        ch->m2mf_dst_offset = param;
    } else if (method == 0x0c5) {
        ch->m2mf_src_pitch = param;
    } else if (method == 0x0c6) {
        ch->m2mf_dst_pitch = param;
    } else if (method == 0x0c7) {
        ch->m2mf_line_length = param;
    } else if (method == 0x0c8) {
        ch->m2mf_line_count = param;
    } else if (method == 0x0c9) {
        ch->m2mf_format = param;
    } else if (method == 0x0ca) {
        ch->m2mf_buffer_notify = param;
        nv_m2mf(s, ch);
        if ((nv_ramin_read32(s, ch->schs[subc].notifier) & 0xff) != 0x30) {
            dma_write64(s, ch->schs[subc].notifier, 0x10 + 0x0, nv_get_time(s));
            dma_write32(s, ch->schs[subc].notifier, 0x10 + 0x8, 0);
            dma_write32(s, ch->schs[subc].notifier, 0x10 + 0xC, 0);
        }
    }
}

static void execute_rop(NvChannel *ch, uint32_t method, uint32_t param)
{
    if (method == 0x0c0) {
        ch->rop = param;
    }
}

static void execute_patt(NvChannel *ch, uint32_t method, uint32_t param)
{
    if (method == 0x0c2) {
        ch->patt_shape = param;
    } else if (method == 0x0c3) {
        ch->patt_type_color = param == 2;
    } else if (method == 0x0c4) {
        ch->patt_bg_color = param;
    } else if (method == 0x0c5) {
        ch->patt_fg_color = param;
    } else if (method == 0x0c6 || method == 0x0c7) {
        for (uint32_t i = 0; i < 32; i++) {
            ch->patt_data_mono[i + (method & 1) * 32] = (1 << (i ^ 7)) & param;
        }
    } else if (method >= 0x100 && method < 0x110) {
        uint32_t i = (method - 0x100) * 4;
        ch->patt_data_color[i] = param & 0xff;
        ch->patt_data_color[i + 1] = (param >> 8) & 0xff;
        ch->patt_data_color[i + 2] = (param >> 16) & 0xff;
        ch->patt_data_color[i + 3] = param >> 24;
    } else if (method >= 0x140 && method < 0x160) {
        uint32_t i = (method - 0x140) * 2;
        ch->patt_data_color[i] = param & 0xffff;
        ch->patt_data_color[i + 1] = param >> 16;
    } else if (method >= 0x1c0 && method < 0x200) {
        ch->patt_data_color[method - 0x1c0] = param;
    }
}

static void execute_gdi(NVVGAState *s, NvChannel *ch, uint32_t cls,
                        uint32_t method, uint32_t param)
{
    if (method == 0x0bf) {
        ch->gdi_operation = param;
    } else if (method == 0x0c0) {
        ch->gdi_color_fmt = param;
    } else if (method == 0x0c1) {
        ch->gdi_mono_fmt = param;
    } else if (method == 0x0ff) {
        ch->gdi_rect_color = param;
    } else if (method >= 0x100 && method < 0x140) {
        if (method & 1) {
            ch->gdi_rect_wh = param;
            gdi_fillrect(s, ch, false);
        } else {
            ch->gdi_rect_xy = param;
        }
    } else if (method == 0x17d) {
        ch->gdi_clip_yx0 = param;
    } else if (method == 0x17e) {
        ch->gdi_clip_yx1 = param;
    } else if (method == 0x17f) {
        ch->gdi_rect_color = param;
    } else if (method >= 0x180 && method < 0x1c0) {
        if (method & 1) {
            ch->gdi_rect_yx1 = param;
            gdi_fillrect(s, ch, true);
        } else {
            ch->gdi_rect_yx0 = param;
        }
    } else if ((method == 0x1fb && cls == 0x004a) ||
               (method == 0x2fb && cls == 0x004b)) {
        ch->gdi_clip_yx0 = param;
    } else if ((method == 0x1fc && cls == 0x004a) ||
               (method == 0x2fc && cls == 0x004b)) {
        ch->gdi_clip_yx1 = param;
    } else if ((method == 0x1fd && cls == 0x004a) ||
               (method == 0x2fd && cls == 0x004b)) {
        ch->gdi_fg_color = param;
    } else if ((method == 0x1fe && cls == 0x004a) ||
               (method == 0x2fe && cls == 0x004b)) {
        ch->gdi_image_swh = param;
    } else if ((method == 0x1ff && cls == 0x004a) ||
               (method == 0x2ff && cls == 0x004b)) {
        ch->gdi_image_xy = param;
        uint32_t width = ch->gdi_image_swh & 0xffff;
        uint32_t height = ch->gdi_image_swh >> 16;
        uint32_t wc = NV_ALIGN(width * height, 32) >> 5;
        g_free(ch->gdi_words);
        ch->gdi_words_ptr = 0;
        ch->gdi_words_left = wc;
        ch->gdi_words = g_new(uint32_t, wc ? wc : 1);
    } else if ((method >= 0x200 && method < 0x280 && cls == 0x004a) ||
               (method >= 0x300 && method < 0x380 && cls == 0x004b)) {
        ch->gdi_words[ch->gdi_words_ptr++] = param;
        ch->gdi_words_left--;
        if (!ch->gdi_words_left) {
            gdi_blit(s, ch, 0);
            g_free(ch->gdi_words);
            ch->gdi_words = NULL;
        }
    } else if ((method == 0x2f9 && cls == 0x004a) ||
               (method == 0x4f9 && cls == 0x004b)) {
        ch->gdi_clip_yx0 = param;
    } else if ((method == 0x2fa && cls == 0x004a) ||
               (method == 0x4fa && cls == 0x004b)) {
        ch->gdi_clip_yx1 = param;
    } else if ((method == 0x2fb && cls == 0x004a) ||
               (method == 0x4fb && cls == 0x004b)) {
        ch->gdi_bg_color = param;
    } else if ((method == 0x2fc && cls == 0x004a) ||
               (method == 0x4fc && cls == 0x004b)) {
        ch->gdi_fg_color = param;
    } else if ((method == 0x2fd && cls == 0x004a) ||
               (method == 0x4fd && cls == 0x004b)) {
        ch->gdi_image_swh = param;
    } else if ((method == 0x2fe && cls == 0x004a) ||
               (method == 0x4fe && cls == 0x004b)) {
        ch->gdi_image_dwh = param;
    } else if ((method == 0x2ff && cls == 0x004a) ||
               (method == 0x4ff && cls == 0x004b)) {
        ch->gdi_image_xy = param;
        uint32_t width = ch->gdi_image_swh & 0xffff;
        uint32_t height = ch->gdi_image_swh >> 16;
        uint32_t wc = NV_ALIGN(width * height, 32) >> 5;
        g_free(ch->gdi_words);
        ch->gdi_words_ptr = 0;
        ch->gdi_words_left = wc;
        ch->gdi_words = g_new(uint32_t, wc ? wc : 1);
    } else if ((method >= 0x300 && method < 0x380 && cls == 0x004a) ||
               (method >= 0x500 && method < 0x580 && cls == 0x004b)) {
        ch->gdi_words[ch->gdi_words_ptr++] = param;
        ch->gdi_words_left--;
        if (!ch->gdi_words_left) {
            gdi_blit(s, ch, 1);
            g_free(ch->gdi_words);
            ch->gdi_words = NULL;
        }
    } else if (method == 0x3fd) {
        ch->gdi_clip_yx0 = param;
    } else if (method == 0x3fe) {
        ch->gdi_clip_yx1 = param;
    } else if (method == 0x3ff) {
        ch->gdi_fg_color = param;
    }
}

static void execute_swzsurf(NVVGAState *s, NvChannel *ch, uint32_t method,
                            uint32_t param)
{
    if (method == 0x061) {
        ch->swzs_img_obj = param;
    } else if (method == 0x0c0) {
        ch->swzs_fmt = param;
        ch->swzs_width = 1 << ((param >> 16) & 0xff);
        ch->swzs_height = 1 << (param >> 24);
        uint32_t color_fmt = param & 0xffff;
        if (color_fmt == 1) {
            ch->swzs_color_bytes = 1;
        } else if (color_fmt == 2 || color_fmt == 4) {
            ch->swzs_color_bytes = 2;
        } else if (color_fmt == 0x6 || color_fmt == 0xA || color_fmt == 0xB) {
            ch->swzs_color_bytes = 4;
        } else {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "geforce: unknown swizzled surface fmt 0x%02x\n",
                          color_fmt);
        }
    } else if (method == 0x0c1) {
        ch->swzs_ofs = param;
    }
}

static void execute_chroma(NvChannel *ch, uint32_t method, uint32_t param)
{
    if (method == 0x0c0) {
        ch->chroma_color_fmt = param;
    } else if (method == 0x0c1) {
        ch->chroma_color = param;
    }
}

static void execute_rect(NVVGAState *s, NvChannel *ch, uint32_t method,
                         uint32_t param)
{
    if (method == 0x0bf) {
        ch->rect_operation = param;
    } else if (method == 0x0c0) {
        ch->rect_color_fmt = param;
    } else if (method == 0x0c1) {
        ch->rect_color = param;
    } else if (method >= 0x100 && method < 0x120) {
        if (method & 1) {
            ch->rect_hw = param;
            nv_rect(s, ch);
        } else {
            ch->rect_yx = param;
        }
    }
}

static void execute_imageblit(NVVGAState *s, NvChannel *ch, uint32_t method,
                              uint32_t param)
{
    if (method == 0x061) {
        ch->blit_color_key_enable = (nv_ramin_read32(s, param) & 0xff) != 0x30;
    } else if (method == 0x0bf) {
        ch->blit_operation = param;
    } else if (method == 0x0c0) {
        ch->blit_syx = param;
    } else if (method == 0x0c1) {
        ch->blit_dyx = param;
    } else if (method == 0x0c2) {
        ch->blit_hw = param;
        nv_copyarea(s, ch);
    }
}

static void execute_ifc(NVVGAState *s, NvChannel *ch, uint32_t method,
                        uint32_t param)
{
    if (method == 0x061) {
        ch->ifc_color_key_enable = (nv_ramin_read32(s, param) & 0xff) != 0x30;
    } else if (method == 0x062) {
        ch->ifc_clip_enable = (nv_ramin_read32(s, param) & 0xff) != 0x30;
    } else if (method == 0x0bf) {
        ch->ifc_operation = param;
    } else if (method == 0x0c0) {
        ch->ifc_color_fmt = param;
        update_color_bytes_ifc(ch);
        ch->ifc_pixels_per_word = 4 / ch->ifc_color_bytes;
    } else if (method == 0x0c1) {
        ch->ifc_x = 0;
        ch->ifc_y = 0;
        ch->ifc_ofs_x = param & 0xffff;
        ch->ifc_ofs_y = param >> 16;
        ch->ifc_draw_offset = ch->s2d_ofs_dst +
            ch->ifc_ofs_y * ch->s2d_pitch_dst + ch->ifc_ofs_x * ch->s2d_color_bytes;
        ch->ifc_redraw_offset = dma_lin_lookup(s, ch->s2d_img_dst,
                                               ch->ifc_draw_offset) - s->disp_offset;
    } else if (method == 0x0c2) {
        ch->ifc_dst_width = param & 0xffff;
        ch->ifc_dst_height = param >> 16;
        ch->ifc_clip_x0 = 0;
        ch->ifc_clip_y0 = 0;
        ch->ifc_clip_x1 = ch->ifc_dst_width;
        ch->ifc_clip_y1 = ch->ifc_dst_height;
        if (ch->ifc_clip_enable) {
            int32_t clipx0 = ch->clip_x - ch->ifc_ofs_x;
            int32_t clipy0 = ch->clip_y - ch->ifc_ofs_y;
            int32_t clipx1 = clipx0 + ch->clip_width;
            int32_t clipy1 = clipy0 + ch->clip_height;
            ch->ifc_clip_x0 = MAX((int32_t)ch->ifc_clip_x0, clipx0);
            ch->ifc_clip_y0 = MAX((int32_t)ch->ifc_clip_y0, clipy0);
            ch->ifc_clip_x1 = MIN((int32_t)ch->ifc_clip_x1, clipx1);
            ch->ifc_clip_y1 = MIN((int32_t)ch->ifc_clip_y1, clipy1);
        }
    } else if (method == 0x0c3) {
        ch->ifc_src_width = param & 0xffff;
        ch->ifc_src_height = param >> 16;
    } else if (method >= 0x100 && method < 0x800) {
        nv_ifc(s, ch, param);
    }
}

static void execute_surf2d(NvChannel *ch, uint32_t method, uint32_t param)
{
    ch->s2d_locked = true;
    if (method == 0x061) {
        ch->s2d_img_src = param;
    } else if (method == 0x062) {
        ch->s2d_img_dst = param;
    } else if (method == 0x0c0) {
        ch->s2d_color_fmt = param;
        uint32_t prev = ch->s2d_color_bytes;
        update_color_bytes_s2d(ch);
        if (ch->s2d_color_bytes != prev &&
            (ch->s2d_color_bytes == 1 || prev == 1)) {
            update_color_bytes_ifc(ch);
            update_color_bytes_sifc(ch);
            update_color_bytes_tfc(ch);
        }
    } else if (method == 0x0c1) {
        ch->s2d_pitch_src = param & 0xffff;
        ch->s2d_pitch_dst = param >> 16;
    } else if (method == 0x0c2) {
        ch->s2d_ofs_src = param;
    } else if (method == 0x0c3) {
        ch->s2d_ofs_dst = param;
    }
}

static void execute_iifc(NVVGAState *s, NvChannel *ch, uint32_t method,
                         uint32_t param)
{
    if (method == 0x061) {
        ch->iifc_palette = param;
    } else if (method == 0x0f9) {
        ch->iifc_operation = param;
    } else if (method == 0x0fa) {
        ch->iifc_color_fmt = param;
        update_color_bytes_iifc(ch);
    } else if (method == 0x0fb) {
        ch->iifc_bpp4 = param;
    } else if (method == 0x0fc) {
        ch->iifc_palette_ofs = param;
    } else if (method == 0x0fd) {
        ch->iifc_yx = param;
    } else if (method == 0x0fe) {
        ch->iifc_dhw = param;
    } else if (method == 0x0ff) {
        ch->iifc_shw = param;
        uint32_t width = ch->iifc_shw & 0xffff;
        uint32_t height = ch->iifc_shw >> 16;
        uint32_t wc = NV_ALIGN(width * height * (ch->iifc_bpp4 ? 4 : 8), 32) >> 5;
        g_free(ch->iifc_words);
        ch->iifc_words_ptr = 0;
        ch->iifc_words_left = wc;
        ch->iifc_words = g_new(uint32_t, wc ? wc : 1);
    } else if (method >= 0x100 && method < 0x800) {
        ch->iifc_words[ch->iifc_words_ptr++] = param;
        ch->iifc_words_left--;
        if (!ch->iifc_words_left) {
            nv_iifc(s, ch);
            g_free(ch->iifc_words);
            ch->iifc_words = NULL;
        }
    }
}

static void execute_sifc(NVVGAState *s, NvChannel *ch, uint32_t method,
                         uint32_t param)
{
    if (method == 0x0bf) {
        ch->sifc_operation = param;
    } else if (method == 0x0c0) {
        ch->sifc_color_fmt = param;
        update_color_bytes_sifc(ch);
    } else if (method == 0x0c1) {
        ch->sifc_shw = param;
    } else if (method == 0x0c2) {
        ch->sifc_dxds = param;
    } else if (method == 0x0c3) {
        ch->sifc_dydt = param;
    } else if (method == 0x0c4) {
        ch->sifc_clip_yx = param;
    } else if (method == 0x0c5) {
        ch->sifc_clip_hw = param;
    } else if (method == 0x0c6) {
        ch->sifc_syx = param;
        uint32_t width = ch->sifc_shw & 0xffff;
        uint32_t height = ch->sifc_shw >> 16;
        uint32_t wc = NV_ALIGN(width * height * ch->sifc_color_bytes, 4) >> 2;
        g_free(ch->sifc_words);
        ch->sifc_words_ptr = 0;
        ch->sifc_words_left = wc;
        ch->sifc_words = g_new(uint32_t, wc ? wc : 1);
    } else if (method >= 0x100 && method < 0x800) {
        ch->sifc_words[ch->sifc_words_ptr++] = param;
        ch->sifc_words_left--;
        if (!ch->sifc_words_left) {
            nv_sifc(s, ch);
            g_free(ch->sifc_words);
            ch->sifc_words = NULL;
        }
    }
}

static void execute_beta(NvChannel *ch, uint32_t method, uint32_t param)
{
    if (method == 0x0c0) {
        ch->beta = param;
    }
}

static void execute_tfc(NVVGAState *s, NvChannel *ch, uint32_t method,
                        uint32_t param)
{
    if (method == 0x061) {
        uint8_t cls8 = nv_ramin_read32(s, param);
        ch->tfc_swizzled = cls8 == 0x52 || cls8 == 0x9e;
    } else if (method == 0x0c0) {
        ch->tfc_color_fmt = param;
        update_color_bytes_tfc(ch);
    } else if (method == 0x0c1) {
        ch->tfc_yx = param;
    } else if (method == 0x0c2) {
        ch->tfc_hw = param;
        ch->tfc_upload = param == 0x01000100 && ch->tfc_yx == 0 &&
            ch->tfc_color_fmt == 4 && ch->s2d_color_fmt == 0xA &&
            ch->s2d_pitch_src == 0x0400 && ch->s2d_pitch_dst == 0x0400;
        if (ch->tfc_upload) {
            ch->tfc_upload_offset = ch->s2d_ofs_dst;
        } else {
            uint32_t width = ch->tfc_hw & 0xffff;
            uint32_t height = ch->tfc_hw >> 16;
            uint32_t wc = NV_ALIGN(width * height * ch->tfc_color_bytes, 4) >> 2;
            g_free(ch->tfc_words);
            ch->tfc_words_ptr = 0;
            ch->tfc_words_left = wc;
            ch->tfc_words = g_new(uint32_t, wc ? wc : 1);
        }
    } else if (method == 0x0c3) {
        ch->tfc_clip_wx = param;
    } else if (method == 0x0c4) {
        ch->tfc_clip_hy = param;
    } else if (method >= 0x100 && method < 0x800) {
        if (ch->tfc_upload) {
            dma_write32(s, ch->s2d_img_dst, ch->tfc_upload_offset, param);
            ch->tfc_upload_offset += 4;
        } else if (ch->tfc_words != NULL) {
            ch->tfc_words[ch->tfc_words_ptr++] = param;
            ch->tfc_words_left--;
            if (!ch->tfc_words_left) {
                nv_tfc(s, ch);
                g_free(ch->tfc_words);
                ch->tfc_words = NULL;
            }
        }
    }
}

static void execute_sifm(NVVGAState *s, NvChannel *ch, uint32_t cls,
                         uint32_t method, uint32_t param)
{
    if (method == 0x061) {
        ch->sifm_src = param;
    } else if (method == 0x066) {
        uint8_t surf_cls8 = nv_ramin_read32(s, param);
        bool swizzled = surf_cls8 == 0x52 || surf_cls8 == 0x9e;
        if (cls == 0x0389) {
            ch->sifm_swizzled_0389 = swizzled;
        } else {
            ch->sifm_swizzled = swizzled;
        }
    } else if (method == 0x0c0) {
        ch->sifm_color_fmt = param;
        if (ch->sifm_color_fmt == 8) {
            ch->sifm_color_bytes = 1;
        } else if (ch->sifm_color_fmt == 1 || ch->sifm_color_fmt == 2 ||
                   ch->sifm_color_fmt == 7) {
            ch->sifm_color_bytes = 2;
        } else if (ch->sifm_color_fmt == 3 || ch->sifm_color_fmt == 4) {
            ch->sifm_color_bytes = 4;
        } else {
            qemu_log_mask(LOG_GUEST_ERROR,
                          "geforce: unknown sifm color fmt 0x%02x\n",
                          ch->sifm_color_fmt);
        }
    } else if (method == 0x0c1) {
        ch->sifm_operation = param;
    } else if (method == 0x0c4) {
        ch->sifm_dyx = param;
    } else if (method == 0x0c5) {
        ch->sifm_dhw = param;
    } else if (method == 0x0c6) {
        ch->sifm_dudx = param;
    } else if (method == 0x0c7) {
        ch->sifm_dvdy = param;
    } else if (method == 0x100) {
        ch->sifm_shw = param;
    } else if (method == 0x101) {
        ch->sifm_sfmt = param;
    } else if (method == 0x102) {
        ch->sifm_sofs = param;
    } else if (method == 0x103) {
        ch->sifm_syx = param;
        nv_sifm(s, ch, cls == 0x0389 ? ch->sifm_swizzled_0389 : ch->sifm_swizzled);
    }
}

/* D3D dispatcher lives in geforce_d3d.c; forward through the public entry. */
static void execute_d3d(NVVGAState *s, NvChannel *ch, uint32_t cls,
                        uint32_t method, uint32_t param)
{
    nv_execute_d3d(s, ch, cls, method, param);
}

/* ---- public wrappers used by the D3D translation unit ------------------- */
uint32_t nv_ramin_read32_pub(NVVGAState *s, uint32_t o) { return nv_ramin_read32(s, o); }
uint32_t nv_dma_lin_lookup(NVVGAState *s, uint32_t obj, uint32_t a) { return dma_lin_lookup(s, obj, a); }
uint8_t  nv_dma_read8(NVVGAState *s, uint32_t obj, uint32_t a) { return dma_read8(s, obj, a); }
uint16_t nv_dma_read16(NVVGAState *s, uint32_t obj, uint32_t a) { return dma_read16(s, obj, a); }
uint32_t nv_dma_read32(NVVGAState *s, uint32_t obj, uint32_t a) { return dma_read32(s, obj, a); }
uint64_t nv_dma_read64(NVVGAState *s, uint32_t obj, uint32_t a) { return dma_read64(s, obj, a); }
void nv_dma_write8(NVVGAState *s, uint32_t obj, uint32_t a, uint8_t v) { dma_write8(s, obj, a, v); }
void nv_dma_write16(NVVGAState *s, uint32_t obj, uint32_t a, uint16_t v) { dma_write16(s, obj, a, v); }
void nv_dma_write32(NVVGAState *s, uint32_t obj, uint32_t a, uint32_t v) { dma_write32(s, obj, a, v); }
void nv_dma_write64(NVVGAState *s, uint32_t obj, uint32_t a, uint64_t v) { dma_write64(s, obj, a, v); }
uint32_t nv_swizzle(uint32_t x, uint32_t y, uint32_t w, uint32_t h) { return swizzle(x, y, w, h); }
uint64_t nv_get_time_pub(NVVGAState *s) { return nv_get_time(s); }
void nv_update_irq_pub(NVVGAState *s) { update_irq_level(s); }
void nv_redraw_nd_pub(NVVGAState *s, uint32_t o, uint32_t w, uint32_t h) { nv_redraw_nd(s, o, w, h); }

/* ======================================================================== */
/* FIFO command execution                                                   */
/* ======================================================================== */

static void update_fifo_wait(NVVGAState *s)
{
    s->fifo_wait = s->fifo_wait_soft | s->fifo_wait_notify |
                   s->fifo_wait_flip | s->fifo_wait_acquire;
}

static int execute_command(NVVGAState *s, uint32_t chid, uint32_t subc,
                           uint32_t method, uint32_t param)
{
    int result = 0;
    bool software_method = false;
    NvChannel *ch = &s->chs[chid];

    if (method == 0x000) {
        if (ch->schs[subc].engine == 0x01) {
            uint32_t word1 = nv_ramin_read32(s, ch->schs[subc].object + 0x4);
            word1 = (word1 & 0x0000ffff) | (ch->schs[subc].notifier >> 4 << 16);
            uint32_t word0 = nv_ramin_read32(s, ch->schs[subc].object);
            uint8_t cls8 = word0;
            if (cls8 == 0x4a || cls8 == 0x4b) {
                word0 = (word0 & 0xfffc7fff) | (ch->gdi_operation << 15);
                word1 = (word1 & 0xfffffffc) | ch->gdi_mono_fmt;
                nv_ramin_write32(s, ch->schs[subc].object, word0);
            } else if (cls8 == 0x62) {
                nv_ramin_write32(s, ch->schs[subc].object + 0x8,
                                 (ch->s2d_img_src >> 4) |
                                 (ch->s2d_img_dst >> 4 << 16));
            } else if (cls8 == 0x64) {
                nv_ramin_write32(s, ch->schs[subc].object + 0x8,
                                 ch->iifc_palette >> 4);
                word0 = (word0 & 0xfffc7fff) | (ch->iifc_operation << 15);
                nv_ramin_write32(s, ch->schs[subc].object, word0);
                word1 = (word1 & 0xffff00ff) | ((ch->iifc_color_fmt + 9) << 8);
            }
            nv_ramin_write32(s, ch->schs[subc].object + 0x4, word1);
        }
        ramht_lookup(s, param, chid, &ch->schs[subc].object,
                     &ch->schs[subc].engine);
        if (ch->schs[subc].engine == 0x01) {
            uint32_t word1 = nv_ramin_read32(s, ch->schs[subc].object + 0x4);
            ch->schs[subc].notifier = word1 >> 16 << 4;
            uint32_t word0 = nv_ramin_read32(s, ch->schs[subc].object);
            uint8_t cls8 = word0;
            if (cls8 == 0x48) {
                if (!ch->s2d_locked) {
                    uint32_t srcdst = nv_ramin_read32(s, ch->schs[subc].object + 0x8);
                    ch->s2d_img_src = (srcdst & 0xffff) << 4;
                    ch->s2d_img_dst = srcdst >> 16 << 4;
                    ch->s2d_color_fmt = s->graph_bpixel & 0xf;
                    update_color_bytes_s2d(ch);
                    ch->s2d_pitch_src = s->graph_pitch0 & 0xffff;
                    ch->s2d_pitch_dst = ch->s2d_pitch_src;
                    ch->s2d_ofs_src = s->graph_offset0;
                    ch->s2d_ofs_dst = s->graph_offset0;
                }
            } else if (cls8 == 0x4a || cls8 == 0x4b) {
                ch->gdi_operation = (word0 >> 15) & 7;
                ch->gdi_mono_fmt = word1 & 3;
            } else if (cls8 == 0x62) {
                uint32_t srcdst = nv_ramin_read32(s, ch->schs[subc].object + 0x8);
                ch->s2d_img_src = (srcdst & 0xffff) << 4;
                ch->s2d_img_dst = srcdst >> 16 << 4;
            } else if (cls8 == 0x64) {
                ch->iifc_palette = nv_ramin_read32(s, ch->schs[subc].object + 0x8) << 4;
                ch->iifc_operation = (word0 >> 15) & 7;
                ch->iifc_color_fmt = (word1 >> 8 & 0xff) - 9;
                update_color_bytes_iifc(ch);
            } else if (cls8 == 0x96 || cls8 == 0x97) {
                execute_d3d(s, ch, word0 & s->class_mask, 0, 0);
            }
        } else if (ch->schs[subc].engine == 0x00) {
            software_method = true;
        }
    } else if (method == 0x014) {
        s->fifo_cache1_ref_cnt = param;
    } else if (method == 0x018) {
        uint32_t semaphore_obj;
        ramht_lookup(s, param, chid, &semaphore_obj, NULL);
        s->fifo_cache1_semaphore = semaphore_obj >> 4;
    } else if (method == 0x019) {
        s->fifo_cache1_semaphore &= 0x000fffff;
        s->fifo_cache1_semaphore |= param << 20;
    } else if (method == 0x01a || method == 0x01b) {
        uint32_t semaphore_obj = (s->fifo_cache1_semaphore & 0x000fffff) << 4;
        uint32_t semaphore_offset = s->fifo_cache1_semaphore >> 20;
        if (method == 0x01a) {
            if (dma_read32(s, semaphore_obj, semaphore_offset) != param) {
                s->fifo_wait_acquire = true;
                s->fifo_wait = true;
                result = 2;
            }
        } else {
            dma_write32(s, semaphore_obj, semaphore_offset, param);
        }
    } else if (method >= 0x040) {
        if (ch->schs[subc].engine == 0x01) {
            if (method >= 0x060 && method < 0x080) {
                ramht_lookup(s, param, chid, &param, NULL);
            }
            uint32_t cls = nv_ramin_read32(s, ch->schs[subc].object) & s->class_mask;
            uint8_t cls8 = cls;
            switch (cls8) {
            case 0x19: execute_clip(ch, method, param); break;
            case 0x39: execute_m2mf(s, ch, subc, method, param); break;
            case 0x43: execute_rop(ch, method, param); break;
            case 0x44:
            case 0x18: execute_patt(ch, method, param); break;
            case 0x4a:
            case 0x4b: execute_gdi(s, ch, cls, method, param); break;
            case 0x52:
            case 0x9e: execute_swzsurf(s, ch, method, param); break;
            case 0x57: execute_chroma(ch, method, param); break;
            case 0x5e: execute_rect(s, ch, method, param); break;
            case 0x5f:
            case 0x9f: execute_imageblit(s, ch, method, param); break;
            case 0x61:
            case 0x65:
            case 0x8a:
            case 0x21: execute_ifc(s, ch, method, param); break;
            case 0x62: execute_surf2d(ch, method, param); break;
            case 0x64: execute_iifc(s, ch, method, param); break;
            case 0x66:
            case 0x76: execute_sifc(s, ch, method, param); break;
            case 0x72: execute_beta(ch, method, param); break;
            case 0x7b: execute_tfc(s, ch, method, param); break;
            case 0x89: execute_sifm(s, ch, cls, method, param); break;
            case 0x96:
            case 0x97:
                execute_d3d(s, ch, cls, method, param);
                if (s->fifo_wait_flip) {
                    result = 1;
                }
                break;
            default:
                break;
            }
            if (ch->notify_pending) {
                ch->notify_pending = false;
                if ((nv_ramin_read32(s, ch->schs[subc].notifier) & 0xff) != 0x30) {
                    dma_write64(s, ch->schs[subc].notifier, 0x0, nv_get_time(s));
                    dma_write32(s, ch->schs[subc].notifier, 0x8, 0);
                    dma_write32(s, ch->schs[subc].notifier, 0xC, 0);
                }
                if (ch->notify_type) {
                    s->graph_intr |= 0x00000001;
                    update_irq_level(s);
                    s->graph_nsource |= 0x00000001;
                    s->graph_notify = 0x00110000;
                    uint32_t notifier = ch->schs[subc].notifier >> 4;
                    s->graph_ctx_switch2 = notifier << 16;
                    s->graph_ctx_switch4 = ch->schs[subc].object >> 4;
                    s->graph_trapped_addr = (method << 2) | (subc << 16) | (chid << 20);
                    s->graph_trapped_data = param;
                    s->fifo_wait_notify = true;
                    s->fifo_wait = true;
                }
            }
            if (method == 0x041) {
                ch->notify_pending = true;
                ch->notify_type = param;
            } else if (method == 0x060) {
                ch->schs[subc].notifier = param;
            }
        } else if (ch->schs[subc].engine == 0x00) {
            software_method = true;
        }
    }

    if (software_method) {
        s->fifo_wait_soft = true;
        s->fifo_wait = true;
        s->fifo_intr |= 0x00000001;
        update_irq_level(s);
        s->fifo_cache1_pull0 |= 0x00000100;
        s->fifo_cache1_method[s->fifo_cache1_put / 4] = (method << 2) | (subc << 13);
        s->fifo_cache1_data[s->fifo_cache1_put / 4] = param;
        s->fifo_cache1_put += 4;
        if (s->fifo_cache1_put == NV_CACHE1_SIZE * 4) {
            s->fifo_cache1_put = 0;
        }
        result = 1;
    }
    return result;
}

static void fifo_process_chid(NVVGAState *s, uint32_t chid)
{
    if (s->fifo_wait) {
        return;
    }
    if ((s->fifo_mode & (1u << chid)) == 0) {
        return;
    }
    if ((s->fifo_cache1_push0 & 1u) == 0) {
        return;
    }
    if ((s->fifo_cache1_pull0 & 1u) == 0) {
        return;
    }

    uint32_t oldchid = s->fifo_cache1_push1 & 0x1f;
    if (oldchid == chid) {
        if (s->fifo_cache1_dma_put == s->fifo_cache1_dma_get) {
            return;
        }
    } else {
        if (ramfc_read32(s, chid, 0x0) == ramfc_read32(s, chid, 0x4)) {
            return;
        }
    }

    if (oldchid != chid) {
        ramfc_write32(s, oldchid, 0x0, s->fifo_cache1_dma_put);
        ramfc_write32(s, oldchid, 0x4, s->fifo_cache1_dma_get);
        ramfc_write32(s, oldchid, 0x8, s->fifo_cache1_ref_cnt);
        ramfc_write32(s, oldchid, 0xC, s->fifo_cache1_dma_instance);
        if (s->card_type >= 0x20) {
            ramfc_write32(s, oldchid, 0x2C, s->fifo_cache1_semaphore);
        }
        s->fifo_cache1_dma_put = ramfc_read32(s, chid, 0x0);
        s->fifo_cache1_dma_get = ramfc_read32(s, chid, 0x4);
        s->fifo_cache1_ref_cnt = ramfc_read32(s, chid, 0x8);
        s->fifo_cache1_dma_instance = ramfc_read32(s, chid, 0xC);
        if (s->card_type >= 0x20) {
            s->fifo_cache1_semaphore = ramfc_read32(s, chid, 0x2C);
        }
        s->fifo_cache1_push1 = (s->fifo_cache1_push1 & ~0x1fu) | chid;
    }

    s->fifo_cache1_dma_push |= 0x100;
    if (s->fifo_cache1_dma_instance == 0) {
        return;
    }

    NvChannel *ch = &s->chs[chid];
    while (s->fifo_cache1_dma_get != s->fifo_cache1_dma_put) {
        uint32_t word = dma_read32(s, s->fifo_cache1_dma_instance << 4,
                                   s->fifo_cache1_dma_get);
        s->fifo_cache1_dma_get += 4;

        if (ch->dma_state.mcnt) {
            int cmd_result = execute_command(s, chid, ch->dma_state.subc,
                                             ch->dma_state.mthd, word);
            if (cmd_result <= 1) {
                if (!ch->dma_state.ni) {
                    ch->dma_state.mthd++;
                }
                ch->dma_state.mcnt--;
            } else {
                s->fifo_cache1_dma_get -= 4;
            }
            if (cmd_result != 0) {
                break;
            }
        } else {
            if ((word & 0xe0000003u) == 0x20000000u) {
                s->fifo_cache1_dma_get = word & 0x1fffffffu;
            } else if ((word & 3u) == 1u) {
                s->fifo_cache1_dma_get = word & 0xfffffffcu;
            } else if ((word & 3u) == 2u) {
                ch->subr_return = s->fifo_cache1_dma_get;
                ch->subr_active = true;
                s->fifo_cache1_dma_get = word & 0xfffffffcu;
            } else if (word == 0x00020000u) {
                s->fifo_cache1_dma_get = ch->subr_return;
                ch->subr_active = false;
            } else if ((word & 0xa0030003u) == 0u) {
                ch->dma_state.mthd = (word >> 2) & 0x7ffu;
                ch->dma_state.subc = (word >> 13) & 7u;
                ch->dma_state.mcnt = (word >> 18) & 0x7ffu;
                ch->dma_state.ni = word & 0x40000000u;
            } else {
                s->fifo_cache1_dma_get = s->fifo_cache1_dma_put;
                break;
            }
        }
    }
}

static void fifo_process(NVVGAState *s)
{
    uint32_t offset = (s->fifo_cache1_push1 & 0x1f) + 1;
    for (uint32_t i = 0; i < NV_CHANNEL_COUNT; i++) {
        fifo_process_chid(s, (i + offset) & 0x1f);
    }
}

/* ======================================================================== */
/* VGA I/O relay                                                            */
/* ======================================================================== */

static uint32_t nv_vga_ioport_read(NVVGAState *s, uint32_t addr)
{
    return vga_ioport_read(&s->vga, addr);
}
static void nv_vga_ioport_write(NVVGAState *s, uint32_t addr, uint32_t val)
{
    vga_ioport_write(&s->vga, addr, val);
}

/* ======================================================================== */
/* Display mode switch (NV CRTC extended regs → QEMU VBE)                   */
/* ======================================================================== */

static void nv_switch_mode(NVVGAState *s)
{
    uint8_t depth = s->crtc_reg[0x28] & 0x7f;

    if (depth == 0) {
        vbe_ioport_write_index(&s->vga, 0, VBE_DISPI_INDEX_ENABLE);
        vbe_ioport_write_data(&s->vga, 0, VBE_DISPI_DISABLED);
        return;
    }

    uint32_t h = ((uint32_t)s->crtc_reg[0x01]
               + ((uint32_t)((s->crtc_reg[0x2d] & 0x02) >> 1) << 8)
               + 1u) * 8u;
    uint32_t v = (uint32_t)s->crtc_reg[0x12]
               | ((uint32_t)((s->crtc_reg[0x07] & 0x02) >> 1) << 8)
               | ((uint32_t)((s->crtc_reg[0x07] & 0x40) >> 6) << 9)
               | ((uint32_t)((s->crtc_reg[0x25] & 0x02) >> 1) << 10)
               | ((uint32_t)((s->crtc_reg[0x41] & 0x04) >> 2) << 10);
    v += 1;

    uint32_t pitch = ((uint32_t)s->crtc_reg[0x13]
                   | ((uint32_t)(s->crtc_reg[0x19] >> 5) << 8)
                   | ((uint32_t)((s->crtc_reg[0x42] >> 6) & 1u) << 11u)) << 3;

    uint32_t fb_off = (((uint32_t)s->crtc_reg[0x0d]
                      | ((uint32_t)s->crtc_reg[0x0c] << 8)
                      | ((uint32_t)(s->crtc_reg[0x19] & 0x1f) << 16u)) << 2)
                    + s->crtc_start;

    int bpp;
    switch (depth) {
    case 1: bpp = 8;  break;
    case 2: bpp = 16; break;
    case 3: bpp = 32; break;
    default:
        qemu_log_mask(LOG_UNIMP, "geforce: unsupported depth 0x%02x\n", depth);
        return;
    }

    if (!h || !v) {
        return;
    }

    s->disp_offset = fb_off;

    vbe_ioport_write_index(&s->vga, 0, VBE_DISPI_INDEX_ENABLE);
    vbe_ioport_write_data(&s->vga, 0, VBE_DISPI_DISABLED);

    s->vga.vbe_regs[VBE_DISPI_INDEX_XRES] = h;
    s->vga.vbe_regs[VBE_DISPI_INDEX_YRES] = v;
    s->vga.vbe_regs[VBE_DISPI_INDEX_BPP]  = bpp;

    vbe_ioport_write_index(&s->vga, 0, VBE_DISPI_INDEX_ENABLE);
    vbe_ioport_write_data(&s->vga, 0,
                          VBE_DISPI_ENABLED | VBE_DISPI_LFB_ENABLED |
                          VBE_DISPI_NOCLEARMEM);

    if (pitch) {
        vbe_ioport_write_index(&s->vga, 0, VBE_DISPI_INDEX_VIRT_WIDTH);
        vbe_ioport_write_data(&s->vga, 0, pitch / (bpp / 8));
    }

    s->vga.vbe_start_addr = fb_off / 4;
}

/* ======================================================================== */
/* NV CRTC extended register write (geometry/depth/cursor/DDC)              */
/* ======================================================================== */

static void nv_crtc_ext_write(NVVGAState *s, unsigned idx, uint8_t val)
{
    if (idx > NV_CRTC_MAX) {
        return;
    }
    s->crtc_reg[idx] = val;

    switch (idx) {
    case 0x1d:
    case 0x1e:
        s->bank_base[idx - 0x1d] = (uint32_t)val * 0x8000u;
        break;
    case 0x3f: /* DDC SCL(bit5)/SDA(bit4) bit-bang */
        bitbang_i2c_set(&s->bbi2c, BITBANG_I2C_SCL, (val >> 5) & 1);
        s->crtc_reg[0x3e] = (bitbang_i2c_set(&s->bbi2c, BITBANG_I2C_SDA,
                                             (val >> 4) & 1) ? 0x08 : 0)
                          | (((val >> 5) & 1) ? 0x04 : 0);
        break;
    case 0x37: /* secondary I2C — echo, no device */
    case 0x51:
        s->crtc_reg[idx - 1] = (((val >> 4) & 1) ? 0x08 : 0)
                             | (((val >> 5) & 1) ? 0x04 : 0);
        break;
    default:
        break;
    }

    switch (idx) {
    case 0x01: case 0x07: case 0x0c: case 0x0d:
    case 0x12: case 0x13: case 0x19: case 0x25:
    case 0x28: case 0x2d: case 0x41: case 0x42:
        nv_switch_mode(s);
        break;
    default:
        break;
    }
}

static uint8_t nv_crtc_ext_read(NVVGAState *s, unsigned idx)
{
    if (idx == 0x3e) {
        /* DDC status: SCL_in bit2, SDA_in bit3 */
        return (bitbang_i2c_set(&s->bbi2c, BITBANG_I2C_SDA, 1) ? 0x08 : 0) | 0x04;
    }
    if (idx <= NV_CRTC_MAX) {
        return s->crtc_reg[idx];
    }
    return 0xff;
}

/* ======================================================================== */
/* Byte-wide register access (VGA mirrors + palette DAC + RAMIN)            */
/* ======================================================================== */

static uint32_t register_read32(NVVGAState *s, uint32_t address);
static void register_write32(NVVGAState *s, uint32_t address, uint32_t value);

static uint8_t register_read8(NVVGAState *s, uint32_t address)
{
    /* PRMVIO — VGA sequencer / misc at 0x0C0000 */
    if (address >= 0x0c0000u && address < 0x0c0400u) {
        return nv_vga_ioport_read(s, address & 0xfff);
    }
    /* PRMCIO — VGA CRTC / attribute at 0x601000 */
    if (address >= 0x601000u && address < 0x602000u) {
        uint32_t port = address - 0x601000u;
        if (port == 0x3d4) {
            return s->crtc_index;
        }
        if (port == 0x3d5) {
            if (s->crtc_index > NV_VGA_CRTC_LAST) {
                return nv_crtc_ext_read(s, s->crtc_index);
            }
            return nv_vga_ioport_read(s, 0x3d5);
        }
        return nv_vga_ioport_read(s, port);
    }
    /* PRAMDAC palette DAC at 0x6813xx / 0x6833xx (ports 0x3C6-0x3C9) */
    if ((address >= 0x681300u && address < 0x681400u) ||
        (address >= 0x683300u && address < 0x683400u)) {
        uint32_t port = address & 0xfffu;
        if (port >= 0x3c6 && port <= 0x3c9) {
            return nv_vga_ioport_read(s, port);
        }
        return 0;
    }
    /* RAMIN byte access */
    if (address >= 0x700000u && address < 0x800000u) {
        return nv_ramin_read8(s, address - 0x700000u);
    }
    return (uint8_t)register_read32(s, address & ~3u);
}

static void register_write8(NVVGAState *s, uint32_t address, uint8_t value)
{
    if (address >= 0x0c0000u && address < 0x0c0400u) {
        nv_vga_ioport_write(s, address & 0xfff, value);
        return;
    }
    if (address >= 0x601000u && address < 0x602000u) {
        uint32_t port = address - 0x601000u;
        if (port == 0x3d4) {
            s->crtc_index = value;
            return;
        }
        if (port == 0x3d5) {
            if (s->crtc_index > NV_VGA_CRTC_LAST) {
                nv_crtc_ext_write(s, s->crtc_index, value);
                return;
            }
            nv_vga_ioport_write(s, 0x3d5, value);
            return;
        }
        nv_vga_ioport_write(s, port, value);
        return;
    }
    if ((address >= 0x681300u && address < 0x681400u) ||
        (address >= 0x683300u && address < 0x683400u)) {
        uint32_t port = address & 0xfffu;
        if (port >= 0x3c6 && port <= 0x3c9) {
            nv_vga_ioport_write(s, port, value);
        }
        return;
    }
    if (address >= 0x700000u && address < 0x800000u) {
        nv_ramin_write8(s, address - 0x700000u, value);
        return;
    }
    {
        uint32_t aligned = address & ~3u;
        uint32_t cur = register_read32(s, aligned);
        int shift = (address & 3u) * 8;
        cur = (cur & ~(0xffu << shift)) | ((uint32_t)value << shift);
        register_write32(s, aligned, cur);
    }
}

/* ======================================================================== */
/* 32-bit NV register map                                                   */
/* ======================================================================== */


static uint32_t register_read32(NVVGAState *s, uint32_t address)
{
    uint32_t value = 0;

    if (address == 0x000000) {
        value = (s->card_type == NV_CARD_NV20) ? 0x020200a5u
                                               : ((uint32_t)s->card_type << 20);
    } else if (address == 0x000004) {
        value = s->big_endian_mode ? 0x00000001u : 0x00000000u;
    } else if (address == 0x000100) {
        value = get_mc_intr(s) | (s->mc_soft_intr ? 0x80000000u : 0u);
    } else if (address == 0x000140) {
        value = s->mc_intr_en;
    } else if (address == 0x000200) {
        value = s->mc_enable;

    } else if (address == 0x001100) {
        value = s->bus_intr;
    } else if (address == 0x001140) {
        value = s->bus_intr_en;
    } else if (address >= 0x001800 && address < 0x001900) {
        value = pci_default_read_config(&s->dev, address - 0x001800, 4);

    } else if (address == 0x002080) {
        value = 0x00000001u;
    } else if (address == 0x002100) {
        value = s->fifo_intr;
    } else if (address == 0x002140) {
        value = s->fifo_intr_en;
    } else if (address == 0x002210) {
        value = s->fifo_ramht;
    } else if (address == 0x002214) {
        value = s->fifo_ramfc;
    } else if (address == 0x002218) {
        value = s->fifo_ramro;
    } else if (address == 0x002400) {
        value = (s->fifo_cache1_get != s->fifo_cache1_put) ? 0u : 0x10u;
    } else if (address == 0x002504) {
        value = s->fifo_mode;
    } else if (address == 0x003200) {
        value = s->fifo_cache1_push0;
    } else if (address == 0x003204) {
        value = s->fifo_cache1_push1;
    } else if (address == 0x003210) {
        value = s->fifo_cache1_put;
    } else if (address == 0x003214) {
        value = (s->fifo_cache1_get != s->fifo_cache1_put) ? 0u : 0x10u;
    } else if (address == 0x003220) {
        value = s->fifo_cache1_dma_push;
    } else if (address == 0x00322c) {
        value = s->fifo_cache1_dma_instance;
    } else if (address == 0x003230) {
        value = 0x80000000u;
    } else if (address == 0x003240) {
        value = s->fifo_cache1_dma_put;
    } else if (address == 0x003244) {
        value = s->fifo_cache1_dma_get;
    } else if (address == 0x003248) {
        value = s->fifo_cache1_ref_cnt;
    } else if (address == 0x003250) {
        if (s->fifo_cache1_get != s->fifo_cache1_put) {
            s->fifo_cache1_pull0 |= 0x100u;
        }
        value = s->fifo_cache1_pull0;
    } else if (address == 0x003270) {
        value = s->fifo_cache1_get;
    } else if (address == 0x0032e0) {
        value = s->fifo_grctx_instance;
    } else if (address == 0x003304) {
        value = 1u;
    } else if (address >= 0x003800 && address < 0x004000) {
        uint32_t off = address - 0x3800u;
        uint32_t idx = off / 8u;
        value = (off % 8u == 0u) ? s->fifo_cache1_method[idx]
                                 : s->fifo_cache1_data[idx];

    } else if (address == 0x009100) {
        value = s->timer_intr;
    } else if (address == 0x009140) {
        value = s->timer_intr_en;
    } else if (address == 0x009200) {
        value = s->timer_num;
    } else if (address == 0x009210) {
        value = s->timer_den;
    } else if (address == 0x009400) {
        value = (uint32_t)nv_get_time(s);
    } else if (address == 0x009410) {
        value = (uint32_t)(nv_get_time(s) >> 32);
    } else if (address == 0x009420) {
        value = s->timer_alarm;

    } else if (address == 0x101000) {
        value = s->straps0_primary;
    } else if (address == 0x10020c) {
        value = s->vga.vram_size;
    } else if (address == 0x100320) {
        value = (s->card_type == NV_CARD_NV20) ? 0x00007fffu : 0x0002e3ffu;

    } else if (address >= 0x300000u && address < 0x310000u) {
        uint32_t off = address - 0x300000u;
        uint64_t rom_size = memory_region_size(&s->dev.rom);
        if (rom_size && off + 3 < rom_size) {
            value = ldl_le_p(memory_region_get_ram_ptr(&s->dev.rom) + off);
        }

    } else if (address == 0x400100u) {
        value = s->graph_intr;
    } else if (address == 0x400108u) {
        value = s->graph_nsource;
    } else if (address == 0x400140u) {
        value = s->graph_intr_en;
    } else if (address == 0x40014cu) {
        value = s->graph_ctx_switch1;
    } else if (address == 0x400700u) {
        value = 0;   /* PGRAPH_STATUS: idle */
    } else if (address == 0x400704u) {
        value = s->graph_trapped_addr;
    } else if (address == 0x400708u) {
        value = s->graph_trapped_data;
    } else if (address == 0x400720u) {
        value = 0;
    } else if (address == 0x400724u) {
        value = s->graph_bpixel;

    } else if (address == 0x600100u) {
        value = s->crtc_intr;
    } else if (address == 0x600140u) {
        value = s->crtc_intr_en;
    } else if (address == 0x600800u) {
        value = s->crtc_start;
    } else if (address == 0x600804u) {
        value = s->crtc_config;
    } else if (address == 0x600808u) {
        s->crtc_raster_pos ^= 1u;
        value = s->crtc_raster_pos;
    } else if (address == 0x60080cu) {
        value = s->crtc_cursor_offset;
    } else if (address == 0x600810u) {
        value = s->crtc_cursor_config;
    } else if (address == 0x60081cu) {
        value = s->crtc_gpio_ext;

    } else if (address == 0x680300u) {
        value = s->ramdac_cu_start_pos;
    } else if (address == 0x680404u) {
        value = 0;
    } else if (address == 0x680500u) {
        value = s->ramdac_nvpll;
    } else if (address == 0x680504u) {
        value = s->ramdac_mpll;
    } else if (address == 0x680508u) {
        value = s->ramdac_vpll;
    } else if (address == 0x68050cu) {
        value = s->ramdac_pll_select;
    } else if (address == 0x680578u) {
        value = s->ramdac_vpll_b;
    } else if (address == 0x680600u) {
        value = s->ramdac_general_control;
    } else if (address == 0x680828u) {
        value = 0;

    } else if ((address >= 0x0c0300u && address < 0x0c0400u) ||
               (address >= 0x601300u && address < 0x601400u) ||
               (address >= 0x603300u && address < 0x603400u) ||
               (address >= 0x681300u && address < 0x681400u) ||
               (address >= 0x683300u && address < 0x683400u)) {
        value = register_read8(s, address);

    } else if (address == 0x6013d4u || address == 0x6033d4u) {
        uint8_t idx = s->crtc_index;
        uint8_t dat = (idx > NV_VGA_CRTC_LAST) ? nv_crtc_ext_read(s, idx)
                                               : (uint8_t)nv_vga_ioport_read(s, 0x3d5);
        value = (uint32_t)idx | ((uint32_t)dat << 8);

    } else if (address >= 0x700000u && address < 0x800000u) {
        value = nv_ramin_read32(s, address - 0x700000u);

    } else if ((address >= 0x800000 && address < 0xa00000) ||
               (address >= 0xc00000 && address < 0xe00000)) {
        uint32_t chid, offset;
        if (address < 0xa00000) {
            chid = (address >> 16) & 0x1f;
            offset = address & 0x1fff;
        } else {
            chid = (address >> 12) & 0x1ff;
            if (chid >= NV_CHANNEL_COUNT) {
                chid = 0;
            }
            offset = address & 0xfff;
        }
        uint32_t curchid = s->fifo_cache1_push1 & 0x1f;
        if (offset == 0x40) {
            value = (curchid == chid) ? s->fifo_cache1_dma_put
                                      : ramfc_read32(s, chid, 0x0);
        } else if (offset == 0x44) {
            value = (curchid == chid) ? s->fifo_cache1_dma_get
                                      : ramfc_read32(s, chid, 0x4);
        } else if (offset == 0x48) {
            value = s->fifo_cache1_ref_cnt;
        } else if (offset == 0x10) {
            value = 0xffff;
        }

    } else {
        if ((address >> 2) < (4u * 1024u * 1024u)) {
            value = s->unk_regs[address >> 2];
        }
    }

    return value;
}

static void register_write32(NVVGAState *s, uint32_t address, uint32_t value)
{
    if (address == 0x000004) {
        s->big_endian_mode = (value & 1u) != 0;
        nv_switch_mode(s);
    } else if (address == 0x000100) {
        if (s->mc_soft_intr && (value & 0x80000000u)) {
            s->mc_soft_intr = false;
        }
        update_irq_level(s);
    } else if (address == 0x000140) {
        s->mc_intr_en = value;
        update_irq_level(s);
    } else if (address == 0x000200) {
        s->mc_enable = value;

    } else if (address == 0x001100) {
        s->bus_intr &= ~value;
        update_irq_level(s);
    } else if (address == 0x001140) {
        s->bus_intr_en = value;
        update_irq_level(s);
    } else if (address >= 0x001800u && address < 0x001900u) {
        pci_default_write_config(&s->dev, address - 0x001800u, value, 4);

    } else if (address == 0x002100) {
        s->fifo_intr &= ~value;
        if (value & 0x00000001u) {
            s->fifo_wait_soft = false;
            update_fifo_wait(s);
        }
        update_irq_level(s);
    } else if (address == 0x002140) {
        s->fifo_intr_en = value;
        update_irq_level(s);
    } else if (address == 0x002210) {
        s->fifo_ramht = value;
    } else if (address == 0x002214) {
        s->fifo_ramfc = value;
    } else if (address == 0x002218) {
        s->fifo_ramro = value;
    } else if (address == 0x002504) {
        bool process = (s->fifo_mode | value) != s->fifo_mode;
        s->fifo_mode = value;
        if (process) {
            fifo_process(s);
        }
    } else if (address == 0x003200) {
        s->fifo_cache1_push0 = value;
    } else if (address == 0x003204) {
        s->fifo_cache1_push1 = value;
    } else if (address == 0x003210) {
        s->fifo_cache1_put = value;
    } else if (address == 0x003220) {
        s->fifo_cache1_dma_push = value;
    } else if (address == 0x00322c) {
        s->fifo_cache1_dma_instance = value;
    } else if (address == 0x003240) {
        s->fifo_cache1_dma_put = value;
    } else if (address == 0x003244) {
        s->fifo_cache1_dma_get = value;
    } else if (address == 0x003248) {
        s->fifo_cache1_ref_cnt = value;
    } else if (address == 0x003250) {
        s->fifo_cache1_pull0 = value;
        if (value & 1u) {
            fifo_process(s);
        }
    } else if (address == 0x003270) {
        s->fifo_cache1_get = value;
    } else if (address == 0x0032e0) {
        s->fifo_grctx_instance = value;
    } else if (address >= 0x003800 && address < 0x004000) {
        uint32_t off = address - 0x003800;
        uint32_t idx = off / 8;
        if (off % 8 == 0) {
            s->fifo_cache1_method[idx] = value;
        } else {
            s->fifo_cache1_data[idx] = value;
        }

    } else if (address == 0x009100) {
        s->timer_intr &= ~value;
        update_irq_level(s);
    } else if (address == 0x009140) {
        s->timer_intr_en = value;
        update_irq_level(s);
    } else if (address == 0x009200) {
        s->timer_num = value;
    } else if (address == 0x009210) {
        s->timer_den = value;
    } else if (address == 0x009420) {
        s->timer_alarm = value;

    } else if (address == 0x101000) {
        s->straps0_primary = value;

    } else if (address == 0x400100) {
        s->graph_intr &= ~value;
        if (value & 0x00000001u) {
            s->fifo_wait_notify = false;
            update_fifo_wait(s);
        }
        if (value & 0x00000100u) {
            s->fifo_wait_flip = false;
            update_fifo_wait(s);
        }
        update_irq_level(s);
    } else if (address == 0x400140u) {
        s->graph_intr_en = value;
        update_irq_level(s);
    } else if (address == 0x40014cu) {
        s->graph_ctx_switch1 = value;
    } else if (address == 0x400724u) {
        s->graph_bpixel = value;

    } else if (address == 0x600100) {
        s->crtc_intr &= ~value;
        update_irq_level(s);
    } else if (address == 0x600140) {
        s->crtc_intr_en = value;
        update_irq_level(s);
    } else if (address == 0x600800) {
        s->crtc_start = value;
        nv_switch_mode(s);
    } else if (address == 0x600804) {
        s->crtc_config = value;
    } else if (address == 0x60080c) {
        s->crtc_cursor_offset = value;
        s->hw_cursor.offset = value;
    } else if (address == 0x600810) {
        s->crtc_cursor_config = value;
        s->hw_cursor.enabled = (value & 0x1u) != 0;
        s->hw_cursor.bpp32 = (value & 0x1000u) != 0;
        s->hw_cursor.size = (value & 0x10000u) ? 64 : 32;
        s->hw_cursor.vram = (value & 0x100u) != 0;
    } else if (address == 0x60081c) {
        s->crtc_gpio_ext = value;

    } else if (address == 0x680300) {
        s->ramdac_cu_start_pos = value;
        s->hw_cursor.x = (int16_t)(value & 0xffffu);
        s->hw_cursor.y = (int16_t)((value >> 16) & 0xffffu);
    } else if (address == 0x680500) {
        s->ramdac_nvpll = value;
    } else if (address == 0x680504) {
        s->ramdac_mpll = value;
    } else if (address == 0x680508) {
        s->ramdac_vpll = value;
    } else if (address == 0x68050c) {
        s->ramdac_pll_select = value;
    } else if (address == 0x680578) {
        s->ramdac_vpll_b = value;
    } else if (address == 0x680600) {
        s->ramdac_general_control = value;

    } else if ((address >= 0x0c0300u && address < 0x0c0400u) ||
               (address >= 0x601300u && address < 0x601400u) ||
               (address >= 0x603300u && address < 0x603400u) ||
               (address >= 0x681300u && address < 0x681400u) ||
               (address >= 0x683300u && address < 0x683400u)) {
        register_write8(s, address, (uint8_t)value);

    } else if (address == 0x6013d4u || address == 0x6033d4u) {
        register_write8(s, 0x601000u + 0x3d4, value & 0xff);
        register_write8(s, 0x601000u + 0x3d5, (value >> 8) & 0xff);

    } else if (address >= 0x700000 && address < 0x800000) {
        nv_ramin_write32(s, address - 0x700000u, value);

    } else if (address >= 0x800000 && address < 0xa00000) {
        uint32_t chid = (address >> 16) & 0x1f;
        uint32_t offset = address & 0x1fff;
        s->saw_user_dma_put = true;
        if ((s->fifo_mode & (1u << chid)) != 0 &&
            s->fifo_cache1_dma_instance != 0 && offset == 0x40) {
            uint32_t curchid = s->fifo_cache1_push1 & 0x1f;
            if (curchid == chid) {
                s->fifo_cache1_dma_put = value;
            } else {
                ramfc_write32(s, chid, 0x0, value);
            }
            fifo_process_chid(s, chid);
        }

    } else if (address >= 0xc00000 && address < 0xe00000) {
        uint32_t chid = (address >> 12) & 0x1ff;
        if (chid < NV_CHANNEL_COUNT) {
            uint32_t offset = address & 0xfff;
            if ((s->fifo_mode & (1u << chid)) != 0 &&
                s->fifo_cache1_dma_instance != 0 && offset == 0x40) {
                uint32_t curchid = s->fifo_cache1_push1 & 0x1f;
                if (curchid == chid) {
                    s->fifo_cache1_dma_put = value;
                } else {
                    ramfc_write32(s, chid, 0x0, value);
                }
                fifo_process_chid(s, chid);
            }
        }

    } else {
        if ((address >> 2) < (4u * 1024u * 1024u)) {
            s->unk_regs[address >> 2] = value;
        }
    }
}

/* ======================================================================== */
/* MMIO BAR0 register window                                                */
/* ======================================================================== */

static uint64_t nv_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    NVVGAState *s = opaque;
    uint32_t a = (uint32_t)addr;
    uint32_t val;

    if (size == 1) {
        return register_read8(s, a);
    }
    val = register_read32(s, a & ~3u);
    if (size == 2) {
        return (val >> ((a & 2u) * 8)) & 0xffffu;
    }
    return val;
}

static void nv_mmio_write(void *opaque, hwaddr addr, uint64_t data,
                          unsigned size)
{
    NVVGAState *s = opaque;
    uint32_t a = (uint32_t)addr;

    if (size == 4) {
        register_write32(s, a, (uint32_t)data);
    } else if (size == 1) {
        register_write8(s, a, (uint8_t)data);
    } else { /* size == 2 */
        uint32_t cur = register_read32(s, a & ~3u);
        int shift = (a & 2u) * 8;
        cur = (cur & ~(0xffffu << shift)) | (((uint32_t)data & 0xffff) << shift);
        register_write32(s, a & ~3u, cur);
    }
}

static const MemoryRegionOps nv_mmio_ops = {
    .read = nv_mmio_read,
    .write = nv_mmio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = { .min_access_size = 1, .max_access_size = 4 },
};

/* ======================================================================== */
/* VBlank timer                                                             */
/* ======================================================================== */

static void nv_vblank_irq(void *opaque)
{
    NVVGAState *s = opaque;
    timer_mod(&s->vblank_timer,
              qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +
              NANOSECONDS_PER_SECOND / 60);
    s->crtc_intr |= 0x00000001u;
    update_irq_level(s);

    if (s->fifo_wait_acquire) {
        s->fifo_wait_acquire = false;
        update_fifo_wait(s);
        fifo_process(s);
    }
}

/* ======================================================================== */
/* Realize / reset / exit                                                   */
/* ======================================================================== */

static void nv_vga_realize(PCIDevice *dev, Error **errp)
{
    NVVGAState *s = NV_VGA(dev);
    VGACommonState *vga = &s->vga;
    I2CBus *i2cbus;
    int i;

    if (s->model) {
        for (i = 0; i < (int)ARRAY_SIZE(nv_model_tab); i++) {
            if (!strcmp(s->model, nv_model_tab[i].name)) {
                s->dev_id = nv_model_tab[i].dev_id;
                s->card_type = nv_model_tab[i].card_type;
                vga->vram_size_mb = nv_model_tab[i].vram_mb;
                break;
            }
        }
        if (i >= (int)ARRAY_SIZE(nv_model_tab)) {
            warn_report("geforce: unknown model '%s', using geforce3", s->model);
        }
    }
    if (!s->dev_id) {
        s->dev_id = PCI_DEVICE_ID_NV_GEFORCE3;
    }
    if (!s->card_type) {
        s->card_type = NV_CARD_NV20;
    }
    if (!vga->vram_size_mb) {
        vga->vram_size_mb = 64;
    }

    pci_set_word(dev->config + PCI_DEVICE_ID, s->dev_id);

    s->ramin_flip = (vga->vram_size_mb * MiB) - 0x80000u;
    s->memsize_mask = (vga->vram_size_mb * MiB) - 1u;
    s->class_mask = 0x00000fffu;
    s->straps0_primary = 0x7ff86c6bu | 0x00000180u;
    s->straps0_primary_original = s->straps0_primary;

    s->unk_regs = g_malloc0(4u * 1024u * 1024u * sizeof(uint32_t));

    for (i = 0; i < NV_CHANNEL_COUNT; i++) {
        s->chs[i].s2d_color_bytes = 1;
        s->chs[i].swzs_color_bytes = 1;
        s->chs[i].d3d_color_bytes = 1;
        s->chs[i].d3d_depth_bytes = 1;
    }

    if (!vga_common_init(vga, OBJECT(s), errp)) {
        return;
    }
    vga_init(vga, OBJECT(s), pci_address_space(dev),
             pci_address_space_io(dev), true);
    vga->con = graphic_console_init(DEVICE(s), 0, s->vga.hw_ops, vga);

    i2cbus = i2c_init_bus(DEVICE(s), "nv-vga.ddc");
    bitbang_i2c_init(&s->bbi2c, i2cbus);
    i2c_slave_set_address(I2C_SLAVE(&s->i2cddc), 0x50);
    qdev_realize(DEVICE(&s->i2cddc), BUS(i2cbus), &error_abort);

    memory_region_init_io(&s->mmio, OBJECT(s), &nv_mmio_ops, s,
                          "nv.mmio", NV_PNPMMIO_SIZE);
    pci_std_vga_mmio_region_init(&s->vga, OBJECT(s), &s->mmio, s->mmio_std_vga,
                                 true, false);
    pci_register_bar(dev, 0, PCI_BASE_ADDRESS_MEM_PREFETCH, &s->mmio);

    memory_region_init(&s->vram_aper, OBJECT(dev), "nv.vram",
                       (uint64_t)vga->vram_size_mb * MiB);
    memory_region_add_subregion(&s->vram_aper, 0, &vga->vram);
    pci_register_bar(dev, 1, PCI_BASE_ADDRESS_MEM_PREFETCH, &s->vram_aper);

    dev->config[PCI_INTERRUPT_PIN] = 1;
    timer_init_ns(&s->vblank_timer, QEMU_CLOCK_VIRTUAL, nv_vblank_irq, s);
    timer_mod(&s->vblank_timer,
              qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +
              NANOSECONDS_PER_SECOND / 60);

    s->crtc_reg[0x28] = 0x00;
    s->dac_mask = 0xff;
}

static void nv_vga_reset(DeviceState *dev)
{
    NVVGAState *s = NV_VGA(dev);
    int i;

    update_irq_level(s);
    vga_common_reset(&s->vga);

    s->mc_soft_intr = false;
    s->mc_intr_en = s->mc_enable = 0;
    s->bus_intr = s->bus_intr_en = 0;
    s->fifo_intr = s->fifo_intr_en = 0;
    s->graph_intr = s->graph_intr_en = 0;
    s->crtc_intr = s->crtc_intr_en = 0;
    s->timer_intr = s->timer_intr_en = 0;
    s->big_endian_mode = false;

    s->fifo_wait = s->fifo_wait_soft = s->fifo_wait_notify = false;
    s->fifo_wait_flip = s->fifo_wait_acquire = false;

    memset(s->crtc_reg, 0, sizeof(s->crtc_reg));
    s->crtc_index = 0;
    s->bank_base[0] = s->bank_base[1] = 0;
    s->disp_offset = 0;

    /* free any in-flight engine word buffers */
    for (i = 0; i < NV_CHANNEL_COUNT; i++) {
        g_free(s->chs[i].gdi_words);
        g_free(s->chs[i].iifc_words);
        g_free(s->chs[i].sifc_words);
        g_free(s->chs[i].tfc_words);
        s->chs[i].gdi_words = NULL;
        s->chs[i].iifc_words = NULL;
        s->chs[i].sifc_words = NULL;
        s->chs[i].tfc_words = NULL;
    }
}

static void nv_vga_exit(PCIDevice *dev)
{
    NVVGAState *s = NV_VGA(dev);
    int i;

    timer_del(&s->vblank_timer);
    graphic_console_close(s->vga.con);
    for (i = 0; i < NV_CHANNEL_COUNT; i++) {
        g_free(s->chs[i].gdi_words);
        g_free(s->chs[i].iifc_words);
        g_free(s->chs[i].sifc_words);
        g_free(s->chs[i].tfc_words);
    }
    g_free(s->unk_regs);
}

/* ======================================================================== */
/* Properties / QOM registration                                            */
/* ======================================================================== */

static const Property nv_vga_properties[] = {
    DEFINE_PROP_UINT32("vgamem_mb", NVVGAState, vga.vram_size_mb, 64),
    DEFINE_PROP_STRING("model", NVVGAState, model),
    DEFINE_PROP_UINT16("x-device-id", NVVGAState, dev_id,
                       PCI_DEVICE_ID_NV_GEFORCE3),
    DEFINE_EDID_PROPERTIES(NVVGAState, i2cddc.edid_info),
};

static void nv_vga_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    device_class_set_legacy_reset(dc, nv_vga_reset);
    device_class_set_props(dc, nv_vga_properties);
    dc->hotpluggable = false;
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);

    k->class_id = PCI_CLASS_DISPLAY_VGA;
    k->vendor_id = PCI_VENDOR_ID_NVIDIA;
    k->device_id = PCI_DEVICE_ID_NV_GEFORCE3;
    k->romfile = "vgabios-nvidia.bin";
    k->realize = nv_vga_realize;
    k->exit = nv_vga_exit;
}

static void nv_vga_inst_init(Object *obj)
{
    NVVGAState *s = NV_VGA(obj);
    object_initialize_child(obj, "i2cddc", &s->i2cddc, TYPE_I2CDDC);
}

static const TypeInfo nv_vga_info = {
    .name = TYPE_NV_VGA,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(NVVGAState),
    .instance_init = nv_vga_inst_init,
    .class_init = nv_vga_class_init,
    .interfaces = (InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void nv_vga_register_types(void)
{
    type_register_static(&nv_vga_info);
}

type_init(nv_vga_register_types)
