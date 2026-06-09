/*
 * QEMU NVIDIA GeForce VGA emulation
 *
 * Ported from the Bochs geforce plugin (geforce.cc) to the QEMU PCI device
 * model, following the same patterns used by the ATI VGA emulation (ati.c).
 *
 * Supported models (selectable via -device geforce-vga,model=<name>):
 *   geforce2   — NV15 (10DE:0150), GeForce2 Pro,      64 MiB VRAM
 *   geforce3   — NV20 (10DE:0200), GeForce3,           64 MiB VRAM
 *   geforcefx  — NV35 (10DE:0338), GeForce FX 5900,  128 MiB VRAM
 *   geforce6   — NV40 (10DE:0040), GeForce 6800 GT,  256 MiB VRAM
 *
 * BAR layout (matches real hardware):
 *   BAR0  — 16 MiB  prefetchable MMIO register space (NV_PNPMMIO)
 *   BAR1  — VRAM linear aperture (prefetchable, 64/128/256 MiB)
 *   BAR2  — RAMIN / instance memory (non-prefetchable)
 *
 * The display path reuses VGA common infrastructure (vga_common_init /
 * vga_init) exactly as ati.c does.  Extended modes are activated via
 * NV CRTC register 0x28 (depth byte): 0=VGA, 1=8bpp, 2=16bpp, 3=32bpp.
 *
 * The inline Metal renderer (geforce_render.m, darwin only) accepts NV10
 * push-buffer packets forwarded from nv_3d_flush() and writes rendered
 * pixels back into guest VRAM so the normal display path picks them up.
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
#include "trace.h"

/* ---- model name table --------------------------------------------------- */

static const struct {
    const char *name;
    uint16_t    dev_id;
    uint32_t    card_type;
    uint32_t    vram_mb;
    uint32_t    bar2_size;  /* 0 = no BAR2 */
} nv_model_tab[] = {
    { "geforce2",  PCI_DEVICE_ID_NV_GEFORCE2_PRO,   NV_CARD_NV15, 64,  0 },
    /* NV20 (GeForce3): no BAR2 — RAMIN accessed via BAR0 0x700000 window.
     * Matches Apple FCode ROM expectation on PPC Power Mac G4 AGP. */
    { "geforce3",  PCI_DEVICE_ID_NV_GEFORCE3,        NV_CARD_NV20, 64,  0 },
    { "geforcefx", PCI_DEVICE_ID_NV_GEFORCEFX_5900,  NV_CARD_NV35, 128, 0 },
    { "geforce6",  PCI_DEVICE_ID_NV_GEFORCE6800_GT,  NV_CARD_NV40, 256,
                   NV_RAMIN_SIZE_NV40 },
};

/* ---- helpers ------------------------------------------------------------ */

static inline uint32_t nv_vram_read32(NVVGAState *s, uint32_t offset)
{
    return ldl_le_p(s->vga.vram_ptr + offset);
}

static inline void nv_vram_write32(NVVGAState *s, uint32_t offset, uint32_t v)
{
    stl_le_p(s->vga.vram_ptr + offset, v);
}

static inline uint32_t nv_ramin_read32(NVVGAState *s, uint32_t offset)
{
    return nv_vram_read32(s, offset ^ s->ramin_flip);
}

static inline void nv_ramin_write32(NVVGAState *s, uint32_t offset, uint32_t v)
{
    nv_vram_write32(s, offset ^ s->ramin_flip, v);
}

/* ---- VGA I/O port relay ------------------------------------------------- */

static uint32_t nv_vga_ioport_read(NVVGAState *s, uint32_t addr)
{
    return vga_ioport_read(&s->vga, addr);
}

static void nv_vga_ioport_write(NVVGAState *s, uint32_t addr, uint32_t val)
{
    vga_ioport_write(&s->vga, addr, val);
}

/* ---- IRQ update --------------------------------------------------------- */

static void nv_update_irq(NVVGAState *s)
{
    /* Aggregate interrupt sources */
    uint32_t intr = 0;
    if ((s->bus_intr   & s->bus_intr_en)   != 0) intr |= 0x01000000;
    if ((s->fifo_intr  & s->fifo_intr_en)  != 0) intr |= 0x00000100;
    if ((s->graph_intr & s->graph_intr_en) != 0) intr |= 0x00001000;
    if ((s->crtc_intr  & s->crtc_intr_en) != 0) intr |= 0x02000000;
    if ((s->timer_intr & s->timer_intr_en) != 0) intr |= 0x00100000;
    if (s->mc_soft_intr) intr |= 0x80000000;
    bool level = (intr & s->mc_intr_en) != 0;
    pci_set_irq(&s->dev, level ? 1 : 0);
}

/* ---- VBlank timer ------------------------------------------------------- */

static void nv_vga_vblank_irq(void *opaque)
{
    NVVGAState *s = opaque;
    timer_mod(&s->vblank_timer,
              qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) +
              NANOSECONDS_PER_SECOND / 60);
    s->crtc_intr |= 0x00000001;
    nv_update_irq(s);
}

/* ---- Extended mode switch ----------------------------------------------- */

static void nv_vga_switch_mode(NVVGAState *s)
{
    uint8_t depth = s->crtc_reg[0x28] & 0x7f;

    if (depth == 0) {
        /* plain VGA mode */
        vbe_ioport_write_index(&s->vga, 0, VBE_DISPI_INDEX_ENABLE);
        vbe_ioport_write_data(&s->vga, 0, VBE_DISPI_DISABLED);
        return;
    }

    /* Reconstruct resolution and pitch from NV CRTC registers.
     *
     * Horizontal display end (pixels):
     *   CRTC[1] + 1 → number of characters = hend_chars
     *   pixels = hend_chars * 8
     *
     * Vertical display end (lines):
     *   CRTC[0x12] | (CRTC[7] bit1 << 8) | (CRTC[7] bit6 << 9) | ...
     *
     * Pitch (bytes per row, shifted left 3):
     *   CRTC[0x13] | (CRTC[0x19] >> 5) << 8 | ...  then << 3
     *
     * Color depth from CRTC[0x28]:
     *   0x01 → 8 bpp
     *   0x02 → 16 bpp (565)
     *   0x03 → 32 bpp (ARGB)
     */
    uint32_t h = ((uint32_t)s->crtc_reg[0x01] + 1u) * 8u;
    uint32_t v = (uint32_t)s->crtc_reg[0x12]
               | ((uint32_t)(s->crtc_reg[0x07] & 0x02) << 7)
               | ((uint32_t)(s->crtc_reg[0x07] & 0x40) << 3)
               | ((uint32_t)(s->crtc_reg[0x25] & 0x02) << 9)
               | ((uint32_t)(s->crtc_reg[0x41] & 0x20) << 6);
    v += 1;

    uint32_t pitch = ((uint32_t)s->crtc_reg[0x13]
                   | ((uint32_t)(s->crtc_reg[0x19] >> 5) << 8)
                   | ((uint32_t)(s->crtc_reg[0x42] >> 6 & 1u) << 11u))
                   << 3;

    uint32_t fb_off = (((uint32_t)s->crtc_reg[0x0d]
                      | ((uint32_t)s->crtc_reg[0x0c] << 8)
                      | ((uint32_t)(s->crtc_reg[0x19] & 0x1f) << 16u)) << 2)
                    + s->crtc_start;

    int bpp;
    switch (depth) {
    case 0x01: bpp = 8;  break;
    case 0x02: bpp = 16; break;
    case 0x03: bpp = 32; break;
    default:
        qemu_log_mask(LOG_UNIMP, "geforce: unsupported depth 0x%02x\n", depth);
        return;
    }

    if (!h || !v) return;

    /* Programme the VBE registers for the new mode */
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

    /* Set framebuffer start address (VBE uses 32-bit word units) */
    s->vga.vbe_start_addr = fb_off / 4;

    /* Update Metal renderer if available */
    if (s->render) {
        gf_metal_set_fb(s->render, h, v, pitch, bpp, fb_off);
    }
}

/* ---- NV CRTC extended register read/write ------------------------------- */

static uint8_t nv_crtc_read(NVVGAState *s, unsigned idx)
{
    if (idx <= NV_CRTC_MAX) {
        return s->crtc_reg[idx];
    }
    qemu_log_mask(LOG_UNIMP, "geforce: CRTC read idx=0x%02x\n", idx);
    return 0xff;
}

static void nv_crtc_write(NVVGAState *s, unsigned idx, uint8_t val)
{
    if (idx > NV_CRTC_MAX) {
        qemu_log_mask(LOG_UNIMP, "geforce: CRTC write idx=0x%02x val=0x%02x\n",
                      idx, val);
        return;
    }

    uint8_t old = s->crtc_reg[idx];
    s->crtc_reg[idx] = val;

    switch (idx) {
    case 0x1c:
        if (!(old & 0x80) && (val & 0x80)) {
            s->crtc_intr_en = 0;
            nv_update_irq(s);
        }
        break;
    case 0x1d:
    case 0x1e:
        s->bank_base[idx - 0x1d] = (uint32_t)val * 0x8000u;
        break;
    case 0x28:
        /* depth byte changed: re-programme display mode */
        if (old != val) {
            nv_vga_switch_mode(s);
        }
        break;
    case 0x37: /* DDC SCL/SDA write (head 0) */
    case 0x3f:
    {
        bool scl = (val & 0x20) != 0;
        bool sda = (val & 0x10) != 0;
        if (idx == 0x3f) {
            bitbang_i2c_set(&s->bbi2c, BITBANG_I2C_SCL, scl);
            bitbang_i2c_set(&s->bbi2c, BITBANG_I2C_SDA, sda);
        } else {
            s->crtc_reg[idx - 1] = (sda ? 0x08 : 0) | (scl ? 0x04 : 0);
        }
        break;
    }
    case 0x2f: case 0x30: case 0x31:
    {
        /* cursor base address */
        s->hw_cursor.enabled =
            (s->crtc_reg[0x31] & 0x01) ||
            (s->crtc_cursor_config & 0x00000001);
        s->hw_cursor.vram =
            (s->crtc_reg[0x30] & 0x80) ||
            (s->crtc_cursor_config & 0x00000100) ||
            (s->card_type >= NV_CARD_NV40);
        s->hw_cursor.offset =
            ((uint32_t)(s->crtc_reg[0x31] >> 2) << 11) |
            ((uint32_t)(s->crtc_reg[0x30] & 0x7f) << 17) |
            ((uint32_t)s->crtc_reg[0x2f] << 24);
        s->hw_cursor.offset += s->crtc_cursor_offset;
        break;
    }
    default:
        break;
    }
}

/* ---- Legacy VGA I/O port handlers --------------------------------------- */

static uint64_t nv_vga_ioport_read_cb(void *opaque, hwaddr addr, unsigned size)
{
    NVVGAState *s = opaque;
    uint32_t port = (uint32_t)addr + 0x3b0;

    if (port == 0x3b4 || port == 0x3d4) {
        return s->crtc_index;
    }
    if (port == 0x3b5 || port == 0x3d5) {
        if (s->crtc_index > 0x18) {
            return nv_crtc_read(s, s->crtc_index);
        }
    }
    return nv_vga_ioport_read(s, port);
}

static void nv_vga_ioport_write_cb(void *opaque, hwaddr addr, uint64_t data,
                                   unsigned size)
{
    NVVGAState *s = opaque;
    uint32_t port = (uint32_t)addr + 0x3b0;

    if (port == 0x3b4 || port == 0x3d4) {
        s->crtc_index = (uint8_t)data;
        return;
    }
    if (port == 0x3b5 || port == 0x3d5) {
        if (s->crtc_index > 0x18) {
            nv_crtc_write(s, s->crtc_index, (uint8_t)data);
            /* mode-sensitive indices that also update VBE state */
            if (s->crtc_index == 0x01 || s->crtc_index == 0x07 ||
                s->crtc_index == 0x0c || s->crtc_index == 0x0d ||
                s->crtc_index == 0x12 || s->crtc_index == 0x13 ||
                s->crtc_index == 0x19 || s->crtc_index == 0x28 ||
                s->crtc_index == 0x42) {
                nv_vga_switch_mode(s);
            }
            return;
        }
        s->crtc_reg[s->crtc_index] = (uint8_t)data;
    }
    nv_vga_ioport_write(s, port, (uint32_t)data);
}

static const MemoryRegionOps nv_vga_ioport_ops = {
    .read  = nv_vga_ioport_read_cb,
    .write = nv_vga_ioport_write_cb,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = { .min_access_size = 1, .max_access_size = 1 },
};

/* ---- MMIO register read ------------------------------------------------- */

static uint64_t nv_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    NVVGAState *s = opaque;
    uint32_t a = (uint32_t)addr;
    uint32_t val = 0;

    /* PMC */
    if (a == 0x000000) {
        /* NV_PMC_BOOT_0: chip ID */
        if (s->card_type == NV_CARD_NV20)
            val = 0x020200A5u;
        else
            val = (uint32_t)s->card_type << 20;
    } else if (a == 0x000100) {
        /* NV_PMC_INTR_0: aggregate interrupt status */
        val = 0;
        if (s->bus_intr   & s->bus_intr_en)   val |= 0x01000000u;
        if (s->fifo_intr  & s->fifo_intr_en)  val |= 0x00000100u;
        if (s->graph_intr & s->graph_intr_en) val |= 0x00001000u;
        if (s->crtc_intr  & s->crtc_intr_en)  val |= 0x02000000u;
        if (s->timer_intr & s->timer_intr_en) val |= 0x00100000u;
        if (s->mc_soft_intr) val |= 0x80000000u;
    } else if (a == 0x000140) {
        val = s->mc_intr_en;
    } else if (a == 0x000200) {
        val = s->mc_enable;

    /* PBUS */
    } else if (a == 0x001100) {
        val = s->bus_intr;
    } else if (a == 0x001140) {
        val = s->bus_intr_en;

    /* PCI config space mirror at 0x1800 */
    } else if (a >= 0x001800 && a < 0x001900) {
        val = pci_default_read_config(&s->dev, a - 0x001800, size);

    /* PFIFO */
    } else if (a == 0x002100) {
        val = s->fifo_intr;
    } else if (a == 0x002140) {
        val = s->fifo_intr_en;
    } else if (a == 0x002210) {
        val = s->fifo_ramht;
    } else if (a == 0x002214 && s->card_type < NV_CARD_NV40) {
        val = s->fifo_ramfc;
    } else if (a == 0x002218) {
        val = s->fifo_ramro;
    } else if (a == 0x002220 && s->card_type >= NV_CARD_NV40) {
        val = s->fifo_ramfc;
    } else if (a == 0x002400) {
        /* PFIFO_RUNOUT_STATUS: idle when get==put */
        val = (s->fifo_cache1_get == s->fifo_cache1_put) ? 0x10u : 0x00u;
    } else if (a == 0x002504) {
        val = s->fifo_mode;
    } else if (a == 0x003200) {
        val = s->fifo_cache1_push0;
    } else if (a == 0x003204) {
        val = s->fifo_cache1_push1;
    } else if (a == 0x003210) {
        val = s->fifo_cache1_put;
    } else if (a == 0x003214) {
        val = (s->fifo_cache1_get == s->fifo_cache1_put) ? 0x10u : 0x00u;
    } else if (a == 0x003220) {
        val = s->fifo_cache1_dma_push;
    } else if (a == 0x00322c) {
        val = s->fifo_cache1_dma_instance;
    } else if (a == 0x003230) {
        val = 0x80000000u; /* DMA_CTL: idle */
    } else if (a == 0x003240) {
        val = s->fifo_cache1_dma_put;
    } else if (a == 0x003244) {
        val = s->fifo_cache1_dma_get;
    } else if (a == 0x003248) {
        val = s->fifo_cache1_ref_cnt;
    } else if (a == 0x003250) {
        val = s->fifo_cache1_pull0;
    } else if (a == 0x003270) {
        val = s->fifo_cache1_get;
    } else if (a == 0x0032e0) {
        val = s->fifo_grctx_instance;
    } else if (a == 0x003304) {
        val = 0x00000001u; /* PFIFO_DMA_TIMESLICE: always enabled */

    /* PTIMER */
    } else if (a == 0x009100) {
        val = s->timer_intr;
    } else if (a == 0x009140) {
        val = s->timer_intr_en;
    } else if (a == 0x009200) {
        val = s->timer_num;
    } else if (a == 0x009210) {
        val = s->timer_den;
    } else if (a == 0x009400) {
        val = (uint32_t)qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    } else if (a == 0x009410) {
        val = (uint32_t)(qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) >> 32);
    } else if (a == 0x009420) {
        val = s->timer_alarm;

    /* PFB */
    } else if (a == 0x10020c) {
        val = s->vga.vram_size;

    /* PSTRAPS */
    } else if (a == 0x101000) {
        val = s->straps0_primary;

    /* ROM shadow at 0x300000–0x310000 */
    } else if (a >= 0x300000 && a < 0x310000) {
        /* Option ROM is provided by PCI infrastructure; return zeroes here */
        val = 0;

    /* PGRAPH */
    } else if (a == 0x400100) {
        val = s->graph_intr;
    } else if (a == 0x400108) {
        val = s->graph_nsource;
    } else if ((a == 0x40013c && s->card_type >= NV_CARD_NV40) ||
               (a == 0x400140 && s->card_type <  NV_CARD_NV40)) {
        val = s->graph_intr_en;
    } else if (a == 0x40014c) {
        val = s->graph_ctx_switch1;
    } else if (a == 0x400150) {
        val = s->graph_ctx_switch2;
    } else if (a == 0x400158) {
        val = s->graph_ctx_switch4;
    } else if (a == 0x40032c) {
        val = s->graph_ctxctl_cur;
    } else if (a == 0x400700) {
        val = s->graph_status;
    } else if (a == 0x400704) {
        val = s->graph_trapped_addr;
    } else if (a == 0x400708) {
        val = s->graph_trapped_data;
    } else if (a == 0x400718) {
        val = s->graph_notify;
    } else if (a == 0x400720) {
        val = s->graph_fifo;
    } else if (a == 0x400724) {
        val = s->graph_bpixel;
    } else if (a == 0x400780) {
        val = s->graph_channel_ctx_table;

    /* VGA register aliases inside PRMVIO / PRMDIO (byte-granular) */
    } else if ((a >= 0x0c0300 && a < 0x0c0400) ||
               (a >= 0x0c2300 && a < 0x0c2400)) {
        uint32_t port = a & 0xfff;
        if (port == 0x3c2 || port == 0x3c3 || port == 0x3c4 ||
            port == 0x3c5 || port == 0x3cc || port == 0x3cf) {
            val = nv_vga_ioport_read(s, port);
        }
    } else if ((a >= 0x601300 && a < 0x601400) ||
               (a >= 0x603300 && a < 0x603400)) {
        uint32_t port = a & 0xfff;
        val = nv_vga_ioport_read(s, port);

    /* PCRTC */
    } else if (a == 0x600100) {
        val = s->crtc_intr;
    } else if (a == 0x600140) {
        val = s->crtc_intr_en;
    } else if (a == 0x600800) {
        val = s->crtc_start;
    } else if (a == 0x600804) {
        val = s->crtc_config;
    } else if (a == 0x600808) {
        /* Raster position — fake toggling for drivers that poll it */
        s->crtc_raster_pos ^= 1u;
        val = s->crtc_raster_pos;
    } else if (a == 0x60080c) {
        val = s->crtc_cursor_offset;
    } else if (a == 0x600810) {
        val = s->crtc_cursor_config;
    } else if (a == 0x60081c) {
        val = s->crtc_gpio_ext;
        if (s->card_type == NV_CARD_NV35)
            val |= 0x04000000u;

    /* PRAMDAC */
    } else if (a == 0x680300) {
        val = s->ramdac_cu_start_pos;
    } else if (a == 0x680404) {
        val = 0; /* RAMDAC_NV10_CURSYNC */
    } else if (a == 0x680508) {
        val = s->ramdac_vpll;
    } else if (a == 0x68050c) {
        val = s->ramdac_pll_select;
    } else if (a == 0x680578) {
        val = s->ramdac_vpll_b;
    } else if (a == 0x680600) {
        val = s->ramdac_general_control;
    } else if (a == 0x680828) {
        val = 0; /* PRAMDAC_FP_HCRTC: second monitor disconnected */

    /* DAC I/O aliases */
    } else if ((a >= 0x681300 && a < 0x681400) ||
               (a >= 0x683300 && a < 0x683400)) {
        uint32_t port = a & 0xfff;
        if (port >= 0x3c6 && port <= 0x3c9)
            val = nv_vga_ioport_read(s, port);

    /* RAMIN window at 0x700000–0x800000 (< NV40) */
    } else if (a >= 0x700000 && a < 0x800000) {
        val = nv_ramin_read32(s, a & 0x0fffffu);

    /* FIFO user channels 0x800000–0xA00000 */
    } else if (a >= 0x800000 && a < 0xa00000) {
        uint32_t chid   = (a >> 16) & 0x1f;
        uint32_t offset = a & 0x1fff;
        uint32_t cur_chid = s->fifo_cache1_push1 & 0x1f;
        if (offset == 0x10) {
            val = 0xffffu;
        } else if (offset == 0x40 && cur_chid == chid) {
            val = s->fifo_cache1_dma_put;
        } else if (offset == 0x44 && cur_chid == chid) {
            val = s->fifo_cache1_dma_get;
        } else if (offset == 0x48 && cur_chid == chid) {
            val = s->fifo_cache1_ref_cnt;
        }

    } else {
        /* Unknown register: return 0 quietly */
        qemu_log_mask(LOG_UNIMP,
                      "geforce: mmio_read: unhandled addr=0x%08x size=%u\n",
                      a, size);
        val = 0;
    }

    if (size == 1) {
        return val & 0xffu;
    } else if (size == 2) {
        return val & 0xffffu;
    }
    return val;
}

/* ---- 3D push-buffer flush ------------------------------------------------ */

static void nv_3d_flush(NVVGAState *s)
{
    uint32_t rptr   = s->push.rptr;
    uint32_t wptr   = s->push.wptr;
    uint32_t buf_dw = s->push.buf_size;

    if (!buf_dw || !s->push.buf_addr || !s->render) {
        s->push.rptr = wptr;
        return;
    }

    uint32_t fb_w  = s->vga.vbe_regs[VBE_DISPI_INDEX_XRES];
    uint32_t fb_h  = s->vga.vbe_regs[VBE_DISPI_INDEX_YRES];
    uint32_t bpp   = s->vga.vbe_regs[VBE_DISPI_INDEX_BPP];
    if (!bpp) bpp = 32;
    uint32_t bypp  = (bpp <= 8) ? 1u : (bpp <= 16) ? 2u : 4u;
    uint32_t fb_stride = fb_w * bypp;
    uint32_t fb_off    = s->vga.vbe_start_addr * 4u;

    if (!fb_w || !fb_h) { s->push.rptr = wptr; return; }

    gf_metal_set_fb(s->render, fb_w, fb_h, fb_stride, bpp, fb_off);

    if (bpp <= 8) {
        uint8_t pal8[768];
        for (uint32_t i = 0; i < 256u; i++) {
            pal8[i * 3u + 0u] = s->vga.palette[i * 3u + 0u];
            pal8[i * 3u + 1u] = s->vga.palette[i * 3u + 1u];
            pal8[i * 3u + 2u] = s->vga.palette[i * 3u + 2u];
        }
        gf_metal_set_palette(s->render, pal8, 256u);
    }

#define PB_MAX_BATCH 16384u
    while (rptr != wptr) {
        uint32_t avail = (wptr > rptr) ? (wptr - rptr) : (buf_dw - rptr);
        if (!avail) break;
        uint32_t batch = (avail > PB_MAX_BATCH) ? PB_MAX_BATCH : avail;

        uint32_t *tmp = g_malloc(batch * sizeof(uint32_t));
        for (uint32_t i = 0; i < batch; i++) {
            uint32_t gpa = (uint32_t)(s->push.buf_addr + (uint64_t)(rptr + i) * 4);
            if (gpa + 4 <= s->vga.vram_size) {
                tmp[i] = ldl_le_p(s->vga.vram_ptr + gpa);
            } else {
                tmp[i] = 0;
            }
        }
        gf_metal_submit(s->render, tmp, batch);
        g_free(tmp);

        rptr = (rptr + batch) % buf_dw;
    }
    s->push.rptr = wptr;

    /* Dirty the whole framebuffer so QEMU display picks it up */
    memory_region_set_dirty(&s->vga.vram, fb_off, (uint64_t)fb_stride * fb_h);
    graphic_hw_update(s->vga.con);
}

/* ---- MMIO register write ------------------------------------------------ */

static void nv_mmio_write(void *opaque, hwaddr addr, uint64_t data,
                          unsigned size)
{
    NVVGAState *s = opaque;
    uint32_t a = (uint32_t)addr;
    uint32_t v = (uint32_t)data;

    /* PMC */
    if (a == 0x000100) {
        s->mc_soft_intr = (v >> 31) != 0;
        nv_update_irq(s);
    } else if (a == 0x000140) {
        s->mc_intr_en = v;
        nv_update_irq(s);
    } else if (a == 0x000200) {
        s->mc_enable = v;

    /* PBUS */
    } else if (a == 0x001100) {
        s->bus_intr &= ~v;
        nv_update_irq(s);
    } else if (a == 0x001140) {
        s->bus_intr_en = v;
        nv_update_irq(s);

    /* PCI config space mirror */
    } else if (a >= 0x001800 && a < 0x001900) {
        pci_default_write_config(&s->dev, a - 0x001800, v, size);

    /* PFIFO */
    } else if (a == 0x002100) {
        s->fifo_intr &= ~v;
        nv_update_irq(s);
    } else if (a == 0x002140) {
        s->fifo_intr_en = v;
        nv_update_irq(s);
    } else if (a == 0x002210) {
        s->fifo_ramht = v;
    } else if (a == 0x002214 && s->card_type < NV_CARD_NV40) {
        s->fifo_ramfc = v;
    } else if (a == 0x002218) {
        s->fifo_ramro = v;
    } else if (a == 0x002220 && s->card_type >= NV_CARD_NV40) {
        s->fifo_ramfc = v;
    } else if (a == 0x002504) {
        s->fifo_mode = v;
    } else if (a == 0x003200) {
        s->fifo_cache1_push0 = v;
    } else if (a == 0x003204) {
        s->fifo_cache1_push1 = v;
    } else if (a == 0x003210) {
        s->fifo_cache1_put = v;
    } else if (a == 0x003220) {
        s->fifo_cache1_dma_push = v;
    } else if (a == 0x00322c) {
        s->fifo_cache1_dma_instance = v;
    } else if (a == 0x003240) {
        s->fifo_cache1_dma_put = v;
        /* Guest has advanced the write pointer: flush push buffer */
        s->push.wptr = v / 4;
        nv_3d_flush(s);
    } else if (a == 0x003244) {
        s->fifo_cache1_dma_get = v;
        s->push.rptr = v / 4;
    } else if (a == 0x003248) {
        s->fifo_cache1_ref_cnt = v;
    } else if (a == 0x003250) {
        s->fifo_cache1_pull0 = v;
    } else if (a == 0x003270) {
        s->fifo_cache1_get = v;
    } else if (a == 0x0032e0) {
        s->fifo_grctx_instance = v;

    /* PTIMER */
    } else if (a == 0x009100) {
        s->timer_intr &= ~v;
        nv_update_irq(s);
    } else if (a == 0x009140) {
        s->timer_intr_en = v;
        nv_update_irq(s);
    } else if (a == 0x009200) {
        s->timer_num = v;
    } else if (a == 0x009210) {
        s->timer_den = v;
    } else if (a == 0x009420) {
        s->timer_alarm = v;

    /* PGRAPH */
    } else if (a == 0x400100) {
        s->graph_intr &= ~v;
        nv_update_irq(s);
    } else if ((a == 0x40013c && s->card_type >= NV_CARD_NV40) ||
               (a == 0x400140 && s->card_type <  NV_CARD_NV40)) {
        s->graph_intr_en = v;
        nv_update_irq(s);
    } else if (a == 0x40014c) {
        s->graph_ctx_switch1 = v;
    } else if (a == 0x400150) {
        s->graph_ctx_switch2 = v;
    } else if (a == 0x400158) {
        s->graph_ctx_switch4 = v;
    } else if (a == 0x40032c) {
        s->graph_ctxctl_cur = v;
    } else if (a == 0x400718) {
        s->graph_notify = v;
    } else if (a == 0x400720) {
        s->graph_fifo = v;
    } else if (a == 0x400724) {
        s->graph_bpixel = v;
    } else if (a == 0x400780) {
        s->graph_channel_ctx_table = v;

    /* VGA register aliases (byte-granular) */
    } else if ((a >= 0x0c0300 && a < 0x0c0400) ||
               (a >= 0x0c2300 && a < 0x0c2400)) {
        uint32_t port = a & 0xfff;
        nv_vga_ioport_write(s, port, v & 0xff);
    } else if ((a >= 0x601300 && a < 0x601400) ||
               (a >= 0x603300 && a < 0x603400)) {
        uint32_t port = a & 0xfff;
        nv_vga_ioport_write(s, port, v & 0xff);

    /* PCRTC */
    } else if (a == 0x600100) {
        s->crtc_intr &= ~v;
        nv_update_irq(s);
    } else if (a == 0x600140) {
        s->crtc_intr_en = v;
        nv_update_irq(s);
    } else if (a == 0x600800) {
        s->crtc_start = v;
        nv_vga_switch_mode(s);
    } else if (a == 0x600804) {
        s->crtc_config = v;
    } else if (a == 0x60080c) {
        s->crtc_cursor_offset = v;
    } else if (a == 0x600810) {
        s->crtc_cursor_config = v;
        s->hw_cursor.enabled = (v & 0x1) != 0;
        s->hw_cursor.vram    = (v & 0x100) != 0 || s->card_type >= NV_CARD_NV40;
        s->hw_cursor.bpp32   = true;
        s->hw_cursor.size    = 64;
    } else if (a == 0x60081c) {
        s->crtc_gpio_ext = v;

    /* PRAMDAC */
    } else if (a == 0x680300) {
        s->ramdac_cu_start_pos = v;
    } else if (a == 0x680508) {
        s->ramdac_vpll = v;
    } else if (a == 0x68050c) {
        s->ramdac_pll_select = v;
    } else if (a == 0x680578) {
        s->ramdac_vpll_b = v;
    } else if (a == 0x680600) {
        s->ramdac_general_control = v;

    /* Palette (RAMDAC register aliases) */
    } else if ((a >= 0x681300 && a < 0x681400) ||
               (a >= 0x683300 && a < 0x683400)) {
        uint32_t port = a & 0xfff;
        if (port >= 0x3c6 && port <= 0x3c9)
            nv_vga_ioport_write(s, port, v & 0xff);

    /* RAMIN window at 0x700000 */
    } else if (a >= 0x700000 && a < 0x800000) {
        nv_ramin_write32(s, a & 0x0fffffu, v);

    /* FIFO user channel push-buffer DMA_PUT register
     * 0x800000–0xA00000: chid=(a>>16)&0x1f, offset=a&0x1fff */
    } else if (a >= 0x800000 && a < 0xa00000) {
        uint32_t chid   = (a >> 16) & 0x1f;
        uint32_t offset = a & 0x1fff;
        (void)chid;
        if (offset == 0x40) {
            s->fifo_cache1_dma_put = v;
            s->push.wptr = v / 4;
            nv_3d_flush(s);
        } else if (offset == 0x44) {
            s->fifo_cache1_dma_get = v;
            s->push.rptr = v / 4;
        }

    } else {
        qemu_log_mask(LOG_UNIMP,
                      "geforce: mmio_write: unhandled addr=0x%08x "
                      "val=0x%08x size=%u\n", a, v, size);
    }
}

static const MemoryRegionOps nv_mmio_ops = {
    .read  = nv_mmio_read,
    .write = nv_mmio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 1,
        .max_access_size = 4,
    },
};

/* ---- Device realize ----------------------------------------------------- */

static void nv_vga_realize(PCIDevice *dev, Error **errp)
{
    NVVGAState    *s   = NV_VGA(dev);
    VGACommonState *vga = &s->vga;
    I2CBus *i2cbus;
    int i;

    s->render = NULL;

    /* Resolve model name → device/card parameters */
    if (s->model) {
        for (i = 0; i < (int)ARRAY_SIZE(nv_model_tab); i++) {
            if (!strcmp(s->model, nv_model_tab[i].name)) {
                s->dev_id    = nv_model_tab[i].dev_id;
                s->card_type = nv_model_tab[i].card_type;
                vga->vram_size_mb = nv_model_tab[i].vram_mb;
                s->bar2_size = nv_model_tab[i].bar2_size;
                break;
            }
        }
        if (i >= (int)ARRAY_SIZE(nv_model_tab)) {
            warn_report("geforce: unknown model '%s', using geforce3", s->model);
        }
    }

    /* Fallbacks if nothing set */
    if (!s->dev_id)    s->dev_id    = PCI_DEVICE_ID_NV_GEFORCE3;
    if (!s->card_type) s->card_type = NV_CARD_NV20;
    if (!vga->vram_size_mb) vga->vram_size_mb = 64;
    /* No bar2_size fallback: NV15/NV20/NV35 have no BAR2 (RAMIN is via BAR0
     * 0x700000). Only NV40 sets bar2_size, done from nv_model_tab above. */

    pci_set_word(dev->config + PCI_DEVICE_ID, s->dev_id);

    /* RAMIN is at the top of VRAM XOR-flipped */
    s->ramin_flip = (vga->vram_size_mb * MiB) - 64u;
    s->memsize_mask = (vga->vram_size_mb * MiB) - 1u;
    s->class_mask   = (s->card_type < NV_CARD_NV40) ? 0x00000fffu : 0x0000ffffu;

    /* straps: crystal = 14.318 MHz, CRT connected */
    s->straps0_primary = 0x7FF86C6Bu | 0x00000180u;

    /* ---- VGA init -------------------------------------------------------- */
    if (!vga_common_init(vga, OBJECT(s), errp)) {
        return;
    }

    /* Initialise Metal renderer (no-op on non-darwin) */
    s->render = gf_metal_init(vga->vram_ptr,
                              (uint32_t)vga->vram_size_mb << 20);

    vga_init(vga, OBJECT(s), pci_address_space(dev),
             pci_address_space_io(dev), true);
    vga->con = graphic_console_init(DEVICE(s), 0, s->vga.hw_ops, vga);

    /* ---- i2c / EDID ------------------------------------------------------ */
    i2cbus = i2c_init_bus(DEVICE(s), "nv-vga.ddc");
    bitbang_i2c_init(&s->bbi2c, i2cbus);
    i2c_slave_set_address(I2C_SLAVE(&s->i2cddc), 0x50);
    qdev_realize(DEVICE(&s->i2cddc), BUS(i2cbus), &error_abort);

    /* ---- BAR0: 16 MiB MMIO register window ------------------------------ */
    memory_region_init_io(&s->mmio, OBJECT(s), &nv_mmio_ops, s,
                          "nv.mmio", NV_PNPMMIO_SIZE);
    /* qemu_vga.ndrv talks to Bochs DISPI via BAR0+0x500 (same as pci-vga/ati). */
    pci_std_vga_mmio_region_init(&s->vga, OBJECT(s), &s->mmio, s->mmio_std_vga,
                                 true, false);
    pci_register_bar(dev, 0, PCI_BASE_ADDRESS_MEM_PREFETCH, &s->mmio);

    /* ---- BAR1: VRAM linear aperture (prefetchable) ----------------------- */
    memory_region_init(&s->vram_aper, OBJECT(dev), "nv.vram",
                       (uint64_t)vga->vram_size_mb * MiB);
    memory_region_add_subregion(&s->vram_aper, 0, &vga->vram);
    pci_register_bar(dev, 1, PCI_BASE_ADDRESS_MEM_PREFETCH, &s->vram_aper);

    /* ---- BAR2: RAMIN window (NV40+ only) --------------------------------- */
    if (s->bar2_size) {
        /* NV15/NV20/NV35 have no BAR2 — RAMIN is reached through BAR0
         * 0x700000–0x800000 (already handled in nv_mmio_read/write).
         * NV40 adds a dedicated BAR2 RAMIN aperture. */
        memory_region_init_alias(&s->ramin_mr, OBJECT(s), "nv.ramin",
                                 &vga->vram,
                                 (uint64_t)vga->vram_size_mb * MiB -
                                     s->bar2_size,
                                 s->bar2_size);
        pci_register_bar(dev, 2, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->ramin_mr);
    }

    /* ---- VBlank interrupt timer ----------------------------------------- */
    dev->config[PCI_INTERRUPT_PIN] = 1;
    timer_init_ns(&s->vblank_timer, QEMU_CLOCK_VIRTUAL,
                  nv_vga_vblank_irq, s);

    /* ---- Seed initial CRTC register defaults ----------------------------- */
    s->crtc_reg[0x28] = 0x00; /* VGA mode at boot */
}

/* ---- Device reset ------------------------------------------------------- */

static void nv_vga_reset(DeviceState *dev)
{
    NVVGAState *s = NV_VGA(dev);

    timer_del(&s->vblank_timer);
    nv_update_irq(s);
    vga_common_reset(&s->vga);

    s->mc_soft_intr  = false;
    s->mc_intr_en    = 0;
    s->mc_enable     = 0;
    s->bus_intr      = 0;
    s->bus_intr_en   = 0;
    s->fifo_intr     = 0;
    s->fifo_intr_en  = 0;
    s->graph_intr    = 0;
    s->graph_intr_en = 0;
    s->crtc_intr     = 0;
    s->crtc_intr_en  = 0;
    s->timer_intr    = 0;
    s->timer_intr_en = 0;

    memset(s->crtc_reg, 0, sizeof(s->crtc_reg));
    s->crtc_index = 0;
    s->bank_base[0] = s->bank_base[1] = 0;

    s->push.rptr = s->push.wptr = 0;
    s->push.buf_addr = 0;
    s->push.buf_size = 0;
    s->push.rptr_addr = 0;
}

/* ---- Device exit -------------------------------------------------------- */

static void nv_vga_exit(PCIDevice *dev)
{
    NVVGAState *s = NV_VGA(dev);

    timer_del(&s->vblank_timer);
    graphic_console_close(s->vga.con);
    gf_metal_destroy(s->render);
    s->render = NULL;
}

/* ---- Properties --------------------------------------------------------- */

static const Property nv_vga_properties[] = {
    DEFINE_PROP_UINT32("vgamem_mb", NVVGAState, vga.vram_size_mb, 64),
    DEFINE_PROP_STRING("model", NVVGAState, model),
    DEFINE_PROP_UINT16("x-device-id", NVVGAState, dev_id,
                       PCI_DEVICE_ID_NV_GEFORCE3),
    DEFINE_EDID_PROPERTIES(NVVGAState, i2cddc.edid_info),
};

/* ---- Class init --------------------------------------------------------- */

static void nv_vga_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass   *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    device_class_set_legacy_reset(dc, nv_vga_reset);
    device_class_set_props(dc, nv_vga_properties);
    dc->hotpluggable = false;
    set_bit(DEVICE_CATEGORY_DISPLAY, dc->categories);

    k->class_id  = PCI_CLASS_DISPLAY_VGA;
    k->vendor_id = PCI_VENDOR_ID_NVIDIA;
    k->device_id = PCI_DEVICE_ID_NV_GEFORCE3;
    k->romfile   = "vgabios-nvidia.bin";
    k->realize   = nv_vga_realize;
    k->exit      = nv_vga_exit;
}

static void nv_vga_inst_init(Object *obj)
{
    NVVGAState *s = NV_VGA(obj);
    object_initialize_child(obj, "i2cddc", &s->i2cddc, TYPE_I2CDDC);
}

static const TypeInfo nv_vga_info = {
    .name          = TYPE_NV_VGA,
    .parent        = TYPE_PCI_DEVICE,
    .instance_size = sizeof(NVVGAState),
    .instance_init = nv_vga_inst_init,
    .class_init    = nv_vga_class_init,
    .interfaces    = (InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void nv_vga_register_types(void)
{
    type_register_static(&nv_vga_info);
}

type_init(nv_vga_register_types)
