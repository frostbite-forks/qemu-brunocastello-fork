/*
 * QEMU NVIDIA GeForce VGA emulation
 *
 * Based on the Bochs geforce plugin (geforce.cc / geforce.h) by
 * The Bochs Project, ported to the QEMU PCI device model.
 * Follows the ATI VGA emulation (ati_int.h / ati.c) for QEMU conventions.
 *
 * PCI device layout:
 *   BAR0  — 16 MiB MMIO register space  (NV_PNPMMIO)
 *   BAR1  — VRAM linear aperture (64/128/256 MiB, prefetchable)
 *   BAR2  — RAMIN / instance memory (small MMIO region, < NV40)
 *
 * Supported models:
 *   geforce2   — NV15 (10DE:0150), GeForce2 Pro,  64 MiB
 *   geforce3   — NV20 (10DE:0200), GeForce3,       64 MiB
 *   geforcefx  — NV35 (10DE:0338), GeForce FX 5900, 128 MiB
 *   geforce6   — NV40 (10DE:0040), GeForce 6800 GT, 256 MiB
 */

#ifndef GEFORCE_INT_H
#define GEFORCE_INT_H

#include "qemu/timer.h"
#include "qemu/units.h"
#include "hw/pci/pci_device.h"
#include "hw/i2c/bitbang_i2c.h"
#include "hw/display/i2c-ddc.h"
#include "vga_int.h"
#include "qom/object.h"
#include "geforce_render.h"

/* ---- vendor / device IDs ------------------------------------------------ */

#define PCI_VENDOR_ID_NVIDIA              0x10de

/* NV15 — GeForce2 Pro */
#define PCI_DEVICE_ID_NV_GEFORCE2_PRO     0x0150
/* NV20 — GeForce3 (Ti 500 is 0x0202; we expose GeForce3 = 0x0200) */
#define PCI_DEVICE_ID_NV_GEFORCE3         0x0200
/* NV35 — GeForce FX 5900 Ultra */
#define PCI_DEVICE_ID_NV_GEFORCEFX_5900   0x0338
/* NV40 — GeForce 6800 GT */
#define PCI_DEVICE_ID_NV_GEFORCE6800_GT   0x0040

/* ---- model tokens ------------------------------------------------------- */

#define NV_CARD_NV15  0x15  /* GeForce2 Pro */
#define NV_CARD_NV20  0x20  /* GeForce3     */
#define NV_CARD_NV35  0x35  /* GeForce FX 5900 */
#define NV_CARD_NV40  0x40  /* GeForce 6800 GT */

/* ---- MMIO sizes --------------------------------------------------------- */

/* BAR0: 16 MiB NV register window */
#define NV_PNPMMIO_SIZE         (16u * MiB)
/*
 * BAR2 (RAMIN window) only on NV40+.
 * NV15/NV20/NV35 expose no BAR2 — RAMIN is accessed through BAR0 0x700000.
 * This matches Apple FCode ROM expectations on PPC Power Mac G4/G5 AGP.
 */
#define NV_RAMIN_SIZE_NV40      (16u * MiB)

/* ---- CRTC extended register count --------------------------------------- */
#define NV_CRTC_MAX             0xF0

/* ---- QOM type ----------------------------------------------------------- */
#define TYPE_NV_VGA "geforce-vga"
OBJECT_DECLARE_SIMPLE_TYPE(NVVGAState, NV_VGA)

/* ---- NV push-buffer ring state ------------------------------------------ */
typedef struct NVPushState {
    uint64_t buf_addr;   /* guest phys addr of push-buffer ring */
    uint32_t buf_size;   /* ring size in dwords */
    uint32_t rptr;       /* host read pointer (dwords) */
    uint32_t wptr;       /* guest write pointer (dwords) */
    uint64_t rptr_addr;  /* guest phys addr to echo rptr */
} NVPushState;

/* ---- main device state -------------------------------------------------- */
struct NVVGAState {
    PCIDevice dev;
    VGACommonState vga;

    char    *model;
    uint16_t dev_id;

    /* ---- NV card generation -------------------------------------------- */
    uint32_t card_type;     /* NV_CARD_NV15 / NV20 / NV35 / NV40 */
    uint32_t memsize_mask;
    uint32_t bar2_size;
    uint32_t ramin_flip;    /* XOR mask to locate RAMIN at top of VRAM */
    uint32_t class_mask;    /* 0x00000FFF for < NV40, 0x0000FFFF for NV40+ */

    /* ---- SVGA/CRTC state ----------------------------------------------- */
    uint8_t  crtc_index;
    uint8_t  crtc_reg[NV_CRTC_MAX + 1];

    uint32_t svga_xres, svga_yres, svga_pitch, svga_bpp;
    uint32_t bank_base[2];
    uint32_t disp_offset, disp_end_offset;

    bool     svga_unlock_special;
    bool     svga_needs_update_tile;
    bool     svga_needs_update_mode;

    /* ---- hardware cursor ----------------------------------------------- */
    struct {
        bool     enabled;
        bool     vram;
        bool     bpp32;
        int16_t  x, y;
        uint8_t  size;       /* 32 or 64 pixels */
        uint32_t offset;     /* byte offset in VRAM/RAMIN */
    } hw_cursor;

    /* ---- MC / BUS / FIFO registers ------------------------------------- */
    bool     mc_soft_intr;
    uint32_t mc_intr_en;
    uint32_t mc_enable;
    uint32_t bus_intr;
    uint32_t bus_intr_en;
    uint32_t fifo_intr;
    uint32_t fifo_intr_en;
    uint32_t fifo_ramht;
    uint32_t fifo_ramfc;
    uint32_t fifo_ramro;
    uint32_t fifo_mode;
    uint32_t fifo_cache1_push0;
    uint32_t fifo_cache1_push1;
    uint32_t fifo_cache1_put;
    uint32_t fifo_cache1_dma_push;
    uint32_t fifo_cache1_dma_instance;
    uint32_t fifo_cache1_dma_put;
    uint32_t fifo_cache1_dma_get;
    uint32_t fifo_cache1_ref_cnt;
    uint32_t fifo_cache1_pull0;
    uint32_t fifo_cache1_get;
    uint32_t fifo_grctx_instance;

    /* ---- timer ---------------------------------------------------------- */
    uint32_t timer_intr;
    uint32_t timer_intr_en;
    uint32_t timer_num;
    uint32_t timer_den;
    uint32_t timer_alarm;

    /* ---- PGRAPH --------------------------------------------------------- */
    uint32_t graph_intr;
    uint32_t graph_nsource;
    uint32_t graph_intr_en;
    uint32_t graph_ctx_switch1;
    uint32_t graph_ctx_switch2;
    uint32_t graph_ctx_switch4;
    uint32_t graph_ctxctl_cur;
    uint32_t graph_status;
    uint32_t graph_trapped_addr;
    uint32_t graph_trapped_data;
    uint32_t graph_flip_read;
    uint32_t graph_flip_write;
    uint32_t graph_flip_modulo;
    uint32_t graph_notify;
    uint32_t graph_fifo;
    uint32_t graph_bpixel;
    uint32_t graph_channel_ctx_table;
    uint32_t graph_offset0;
    uint32_t graph_pitch0;

    /* ---- PCRTC / PRAMDAC ----------------------------------------------- */
    uint32_t crtc_intr;
    uint32_t crtc_intr_en;
    uint32_t crtc_start;
    uint32_t crtc_config;
    uint32_t crtc_raster_pos;
    uint32_t crtc_cursor_offset;
    uint32_t crtc_cursor_config;
    uint32_t crtc_gpio_ext;
    uint32_t ramdac_cu_start_pos;
    uint32_t ramdac_vpll;
    uint32_t ramdac_vpll_b;
    uint32_t ramdac_pll_select;
    uint32_t ramdac_general_control;

    /* ---- straps --------------------------------------------------------- */
    uint32_t straps0_primary;

    /* ---- RMA (register access via I/O port 0x3d0) ---------------------- */
    uint32_t rma_addr;

    /* ---- memory regions ------------------------------------------------- */
    MemoryRegion mmio;       /* BAR0: 16 MiB NV register window */
    MemoryRegion mmio_std_vga[4]; /* BAR0+0x400/0x500 Bochs VBE for qemu_vga.ndrv */
    MemoryRegion vram_aper;  /* BAR1: linear VRAM aperture (prefetchable) */
    MemoryRegion ramin_mr;   /* BAR2: RAMIN/instance memory window */
    MemoryRegion io;         /* VGA legacy I/O port alias */

    /* ---- i2c / DDC ------------------------------------------------------ */
    bitbang_i2c_interface bbi2c;
    I2CDDCState           i2cddc;

    /* ---- vblank timer --------------------------------------------------- */
    QEMUTimer vblank_timer;

    /* ---- 3D push-buffer ring -------------------------------------------- */
    NVPushState push;

    /* ---- Metal renderer (darwin only) ----------------------------------- */
    GFRenderState *render;
};

#endif /* GEFORCE_INT_H */
