/*
 * QEMU NVIDIA GeForce VGA emulation — internal header
 *
 * Fresh port: DingusPPC device structure + Bochs bx_geforce_c engine,
 * adapted to the QEMU PCI device model (follows ati_int.h conventions).
 * Software rendering only — no Metal.
 *
 * Supported models (-device geforce-vga,model=<name>):
 *   geforce2mx — NV11 (10DE:0110), GeForce2 MX,  64 MiB
 *   geforce3   — NV20 (10DE:0200), GeForce3,      64 MiB
 *
 * BAR layout:
 *   BAR0  — 16 MiB MMIO register window (NV_PNPMMIO)
 *   BAR1  — VRAM linear aperture (prefetchable)
 *   (no BAR2: NV11/NV20 reach RAMIN through BAR0 0x700000)
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

/* ---- vendor / device IDs ------------------------------------------------ */

#define PCI_VENDOR_ID_NVIDIA              0x10de
#define PCI_DEVICE_ID_NV_GEFORCE2_MX      0x0110   /* NV11 — GeForce2 MX */
#define PCI_DEVICE_ID_NV_GEFORCE3         0x0200   /* NV20 — GeForce3     */

/* ---- card generation tokens (match Bochs/DingusPPC NV_TYPE_*) ----------- */

#define NV_CARD_NV11  0x11   /* GeForce2 MX */
#define NV_CARD_NV15  0x15   /* GeForce2 Pro/GTS (engine-compatible w/ NV11) */
#define NV_CARD_NV20  0x20   /* GeForce3 */

/* ---- sizes -------------------------------------------------------------- */

#define NV_PNPMMIO_SIZE         (16u * MiB)
#define NV_CRTC_MAX             0xF0

#define NV_CHANNEL_COUNT        32
#define NV_SUBCHANNEL_COUNT     8
#define NV_CACHE1_SIZE          64

/* ---- QOM type ----------------------------------------------------------- */
#define TYPE_NV_VGA "geforce-vga"
OBJECT_DECLARE_SIMPLE_TYPE(NVVGAState, NV_VGA)

/* ---- texture descriptor (D3D, Layer 2) ---------------------------------- */
typedef struct NvTexture {
    uint32_t offset, dma_obj, format;
    bool cubemap, linear, unnormalized, compressed;
    bool dxt_alpha_data, dxt_alpha_explicit;
    uint32_t color_bytes, levels;
    uint32_t base_size[3], size[3];
    uint32_t face_bytes, wrap[3];
    uint32_t control0;
    bool enabled;
    uint32_t control1;
    bool signed_any, signed_comp[4];
    uint32_t image_rect, pal_dma_obj, pal_ofs, control3, key_color;
    float offset_matrix[4];
} NvTexture;

/* ---- D3D light (Layer 2) ------------------------------------------------ */
typedef struct NvLight {
    float ambient_color[3];
    float diffuse_color[3];
    float specular_color[3];
    float inf_half_vector[3];
    float inf_direction[3];
    float spot_direction[4];
    float local_position[3];
    float local_attenuation[3];
} NvLight;

/* ---- per-FIFO-channel engine state (ported from Bochs gf_channel) ------- */
typedef struct NvChannel {
    uint32_t subr_return;
    bool     subr_active;
    struct { uint32_t mthd, subc, mcnt; bool ni; } dma_state;
    struct { uint32_t object; uint8_t engine; uint32_t notifier; }
             schs[NV_SUBCHANNEL_COUNT];
    bool     notify_pending;
    uint32_t notify_type;

    /* surface2d / blit */
    bool     s2d_locked;
    uint32_t s2d_img_src, s2d_img_dst, s2d_color_fmt, s2d_color_bytes;
    uint32_t s2d_pitch_src, s2d_pitch_dst, s2d_ofs_src, s2d_ofs_dst;

    /* swizzled surface */
    uint32_t swzs_img_obj, swzs_fmt, swzs_color_bytes;
    uint32_t swzs_width, swzs_height, swzs_ofs;

    /* image-from-cpu */
    bool     ifc_color_key_enable, ifc_clip_enable;
    uint32_t ifc_operation, ifc_color_fmt, ifc_color_bytes, ifc_pixels_per_word;
    uint32_t ifc_x, ifc_y, ifc_ofs_x, ifc_ofs_y;
    uint32_t ifc_draw_offset, ifc_redraw_offset;
    uint32_t ifc_dst_width, ifc_dst_height, ifc_src_width, ifc_src_height;
    uint32_t ifc_clip_x0, ifc_clip_y0, ifc_clip_x1, ifc_clip_y1;

    /* indexed image-from-cpu */
    uint32_t iifc_palette, iifc_palette_ofs, iifc_operation;
    uint32_t iifc_color_fmt, iifc_color_bytes, iifc_bpp4;
    uint32_t iifc_yx, iifc_dhw, iifc_shw;
    uint32_t iifc_words_ptr, iifc_words_left;
    uint32_t *iifc_words;

    /* scaled image-from-cpu */
    uint32_t sifc_operation, sifc_color_fmt, sifc_color_bytes;
    uint32_t sifc_shw, sifc_dxds, sifc_dydt, sifc_clip_yx, sifc_clip_hw, sifc_syx;
    uint32_t sifc_words_ptr, sifc_words_left;
    uint32_t *sifc_words;

    /* blit */
    bool     blit_color_key_enable;
    uint32_t blit_operation, blit_syx, blit_dyx, blit_hw;

    /* texture-from-cpu */
    bool     tfc_swizzled;
    uint32_t tfc_color_fmt, tfc_color_bytes, tfc_yx, tfc_hw;
    uint32_t tfc_clip_wx, tfc_clip_hy;
    uint32_t tfc_words_ptr, tfc_words_left;
    uint32_t *tfc_words;
    bool     tfc_upload;
    uint32_t tfc_upload_offset;

    /* scaled image from memory */
    uint32_t sifm_src;
    bool     sifm_swizzled, sifm_swizzled_0389;
    uint32_t sifm_operation, sifm_color_fmt, sifm_color_bytes;
    uint32_t sifm_syx, sifm_dyx, sifm_shw, sifm_dhw;
    int32_t  sifm_dudx, sifm_dvdy;
    uint32_t sifm_sfmt, sifm_sofs;

    /* memory-to-memory format */
    uint32_t m2mf_src, m2mf_dst, m2mf_src_offset, m2mf_dst_offset;
    uint32_t m2mf_src_pitch, m2mf_dst_pitch, m2mf_line_length, m2mf_line_count;
    uint32_t m2mf_format, m2mf_buffer_notify;

    /* ---- D3D (Layer 2) ---- */
    uint32_t d3d_a_obj, d3d_b_obj, d3d_color_obj, d3d_zeta_obj;
    uint32_t d3d_vertex_a_obj, d3d_vertex_b_obj, d3d_report_obj;
    uint32_t d3d_clip_horizontal, d3d_clip_vertical;
    uint32_t d3d_surface_format, d3d_color_bytes, d3d_depth_bytes;
    uint32_t d3d_surface_pitch_a, d3d_surface_pitch_z;
    bool     d3d_local_viewer;
    uint32_t d3d_color_material_emission, d3d_color_material_ambient;
    uint32_t d3d_color_material_diffuse, d3d_color_material_specular;
    uint32_t d3d_fog_mode, d3d_fog_gen_mode;
    float    d3d_fog_params[3];
    uint32_t d3d_fog_enable;
    float    d3d_fog_color[4];
    int16_t  d3d_window_offset_x, d3d_window_offset_y;
    uint32_t d3d_window_clip_x1[8], d3d_window_clip_x2[8];
    uint32_t d3d_window_clip_y1[8], d3d_window_clip_y2[8];
    uint32_t d3d_surface_color_offset, d3d_surface_zeta_offset;
    uint32_t d3d_combiner_alpha_icw[8], d3d_combiner_final[2];
    uint32_t d3d_alpha_test_enable, d3d_alpha_func, d3d_alpha_ref;
    uint32_t d3d_blend_enable;
    uint16_t d3d_blend_sfactor_rgb, d3d_blend_sfactor_alpha;
    uint16_t d3d_blend_dfactor_rgb, d3d_blend_dfactor_alpha;
    uint16_t d3d_blend_equation_rgb, d3d_blend_equation_alpha;
    float    d3d_blend_color[4];
    uint32_t d3d_cull_face_enable, d3d_depth_test_enable, d3d_depth_write_enable;
    uint32_t d3d_stencil_mask, d3d_stencil_func, d3d_stencil_func_ref;
    uint32_t d3d_stencil_func_mask, d3d_stencil_op_sfail;
    uint32_t d3d_stencil_op_dpfail, d3d_stencil_op_dppass;
    uint32_t d3d_lighting_enable, d3d_stencil_test_enable, d3d_depth_func;
    uint32_t d3d_color_mask, d3d_shade_mode;
    float    d3d_clip_min, d3d_clip_max;
    uint32_t d3d_cull_face, d3d_front_face, d3d_normalize_enable;
    float    d3d_material_factor[4];
    uint32_t d3d_separate_specular, d3d_light_enable_mask;
    uint32_t d3d_texgen[8][4], d3d_texture_matrix_enable[16];
    uint32_t d3d_view_matrix_enable;
    float    d3d_model_view_matrix[2][16];
    float    d3d_inverse_model_view_matrix[12];
    float    d3d_composite_matrix[16];
    float    d3d_texture_matrix[8][16];
    float    d3d_texgen_plane[8][4][4];
    uint32_t d3d_scissor_x, d3d_scissor_width, d3d_scissor_y, d3d_scissor_height;
    uint32_t d3d_shader_program, d3d_shader_obj, d3d_shader_offset;
    float    d3d_specular_params[6], d3d_specular_power;
    float    d3d_scene_ambient_color[4];
    uint32_t d3d_viewport_x, d3d_viewport_width, d3d_viewport_y, d3d_viewport_height;
    float    d3d_viewport_offset[4];
    float    d3d_eye_position[4];
    float    d3d_combiner_const_color[8][2][4];
    uint32_t d3d_combiner_alpha_ocw[8], d3d_combiner_color_icw[8];
    float    d3d_viewport_scale[4];
    uint32_t d3d_transform_program[544][4];
    float    d3d_transform_constant[512][4];
    NvLight  d3d_light[8];
    uint32_t d3d_attrib_count, d3d_vertex_data_base_index;
    uint32_t d3d_vertex_data_array_offset[16];
    uint32_t d3d_vertex_data_array_format_type[16];
    uint32_t d3d_vertex_data_array_format_size[16];
    uint32_t d3d_vertex_data_array_format_stride[16];
    bool     d3d_vertex_data_array_format_dx[16];
    bool     d3d_vertex_data_array_format_homogeneous[16];
    uint32_t d3d_begin_end;
    bool     d3d_primitive_done, d3d_triangle_flip;
    uint32_t d3d_vertex_index, d3d_attrib_index, d3d_comp_index;
    float    d3d_vertex_data[4][16][4];
    float    d3d_vertex_data_imm[16][4];
    uint32_t d3d_index_array_offset, d3d_index_array_dma;
    NvTexture d3d_texture[16];
    uint32_t d3d_shader_control, d3d_semaphore_obj, d3d_semaphore_offset;
    uint32_t d3d_zstencil_clear_value, d3d_color_clear_value, d3d_clear_surface;
    uint32_t d3d_combiner_color_ocw[8];
    uint32_t d3d_combiner_control, d3d_combiner_control_num_stages;
    uint32_t d3d_tex_shader_op[4], d3d_tex_shader_previous[4];
    uint32_t d3d_transform_execution_mode, d3d_transform_program_load;
    uint32_t d3d_transform_program_start, d3d_transform_constant_load;
    uint32_t d3d_attrib_in_normal, d3d_attrib_in_color[2], d3d_attrib_out_color[2];
    uint32_t d3d_attrib_out_fogc;
    uint32_t d3d_attrib_in_tex_coord[16], d3d_attrib_out_tex_coord[16];
    bool     d3d_attrib_out_enable[32];
    uint32_t d3d_vs_temp_regs_count, d3d_tex_coord_count;

    /* 2D object state */
    uint8_t  rop;
    uint32_t beta;
    uint16_t clip_x, clip_y, clip_width, clip_height;
    uint32_t chroma_color_fmt, chroma_color;
    uint32_t patt_shape;
    bool     patt_type_color;
    uint32_t patt_bg_color, patt_fg_color;
    bool     patt_data_mono[64];
    uint32_t patt_data_color[64];

    uint32_t gdi_operation, gdi_color_fmt, gdi_mono_fmt;
    uint32_t gdi_clip_yx0, gdi_clip_yx1, gdi_rect_color, gdi_rect_xy;
    uint32_t gdi_rect_yx0, gdi_rect_yx1, gdi_rect_wh;
    uint32_t gdi_bg_color, gdi_fg_color;
    uint32_t gdi_image_swh, gdi_image_dwh, gdi_image_xy;
    uint32_t gdi_words_ptr, gdi_words_left;
    uint32_t *gdi_words;

    uint32_t rect_operation, rect_color_fmt, rect_color, rect_yx, rect_hw;
} NvChannel;

/* ---- main device state -------------------------------------------------- */
struct NVVGAState {
    PCIDevice dev;
    VGACommonState vga;

    char    *model;
    uint16_t dev_id;

    /* card generation */
    uint32_t card_type;     /* NV_CARD_NV11 / NV15 / NV20 */
    uint32_t memsize_mask;
    uint32_t bar2_size;     /* 0 on NV11/NV20 */
    uint32_t ramin_flip;    /* XOR/offset to locate RAMIN at top of VRAM */
    uint32_t class_mask;    /* 0x00000FFF for < NV40 */

    bool     big_endian_mode;   /* NV_PMC_BOOT_1 bit 0 */

    /* CRTC */
    uint8_t  crtc_index;
    uint8_t  crtc_reg[NV_CRTC_MAX + 1];
    uint32_t bank_base[2];

    /* VGA register mirrors (PRMVIO / PRMCIO) */
    uint8_t  vga_misc_output, vga_enable;
    uint8_t  vga_seq_index, vga_seq_data[8];
    uint8_t  vga_attr_index, vga_attr_data[0x20];
    bool     vga_attr_flip;
    uint8_t  vga_gfx_index, vga_gfx_data[16];

    /* DAC palette */
    uint8_t  dac_wr_index, dac_rd_index, dac_state, dac_comp, dac_mask;
    uint8_t  dac_rgb[256][3];

    /* hardware cursor */
    struct {
        bool     enabled, vram, bpp32;
        int16_t  x, y;
        uint8_t  size;
        uint32_t offset;
    } hw_cursor;

    /* MC / BUS / FIFO */
    bool     mc_soft_intr;
    uint32_t mc_intr_en, mc_enable;
    uint32_t bus_intr, bus_intr_en;

    bool     fifo_wait, fifo_wait_soft, fifo_wait_notify;
    bool     fifo_wait_flip, fifo_wait_acquire;
    uint32_t fifo_intr, fifo_intr_en;
    uint32_t fifo_ramht, fifo_ramfc, fifo_ramro, fifo_mode;
    uint32_t fifo_cache1_push0, fifo_cache1_push1, fifo_cache1_put;
    uint32_t fifo_cache1_dma_push, fifo_cache1_dma_instance;
    uint32_t fifo_cache1_dma_put, fifo_cache1_dma_get;
    uint32_t fifo_cache1_ref_cnt, fifo_cache1_pull0;
    uint32_t fifo_cache1_semaphore, fifo_cache1_get;
    uint32_t fifo_grctx_instance;
    uint32_t fifo_cache1_method[NV_CACHE1_SIZE];
    uint32_t fifo_cache1_data[NV_CACHE1_SIZE];

    /* timer */
    uint32_t timer_intr, timer_intr_en;
    uint32_t timer_num, timer_den, timer_alarm;
    uint64_t timer_inittime1, timer_inittime2;

    /* PGRAPH */
    uint32_t graph_intr, graph_nsource, graph_intr_en;
    uint32_t graph_ctx_switch1, graph_ctx_switch2, graph_ctx_switch4;
    uint32_t graph_ctxctl_cur, graph_status;
    uint32_t graph_trapped_addr, graph_trapped_data;
    uint32_t graph_flip_read, graph_flip_write, graph_flip_modulo;
    uint32_t graph_notify, graph_fifo, graph_bpixel;
    uint32_t graph_channel_ctx_table, graph_offset0, graph_pitch0;

    /* PCRTC / PRAMDAC */
    uint32_t crtc_intr, crtc_intr_en, crtc_start, crtc_config, crtc_raster_pos;
    uint32_t crtc_cursor_offset, crtc_cursor_config, crtc_gpio_ext;
    uint32_t ramdac_cu_start_pos, ramdac_nvpll, ramdac_mpll;
    uint32_t ramdac_vpll, ramdac_vpll_b, ramdac_pll_select, ramdac_general_control;

    /* straps */
    uint32_t straps0_primary, straps0_primary_original;

    /* RMA */
    uint32_t rma_addr;
    bool     saw_user_dma_put;

    /* display mode tracking */
    uint32_t disp_offset;
    bool     mode_needs_update;

    /* engine channels */
    NvChannel chs[NV_CHANNEL_COUNT];

    /* catch-all register backing (mirrors Bochs/DingusPPC unk_regs) */
    uint32_t *unk_regs;     /* 4M dwords, heap-allocated */

    /* memory regions */
    MemoryRegion mmio;            /* BAR0: NV register window */
    MemoryRegion mmio_std_vga[4]; /* BAR0+0x400/0x500 Bochs VBE for qemu_vga.ndrv */
    MemoryRegion vram_aper;       /* BAR1: linear VRAM aperture */

    /* i2c / DDC */
    bitbang_i2c_interface bbi2c;
    I2CDDCState           i2cddc;

    /* vblank timer */
    QEMUTimer vblank_timer;
};

/* ---- shared engine helpers (implemented in geforce.c, used by D3D) ------ */
uint32_t nv_ramin_read32_pub(NVVGAState *s, uint32_t o);
uint32_t nv_dma_lin_lookup(NVVGAState *s, uint32_t object, uint32_t address);
uint8_t  nv_dma_read8(NVVGAState *s, uint32_t object, uint32_t address);
uint16_t nv_dma_read16(NVVGAState *s, uint32_t object, uint32_t address);
uint32_t nv_dma_read32(NVVGAState *s, uint32_t object, uint32_t address);
uint64_t nv_dma_read64(NVVGAState *s, uint32_t object, uint32_t address);
void     nv_dma_write8(NVVGAState *s, uint32_t object, uint32_t address, uint8_t v);
void     nv_dma_write16(NVVGAState *s, uint32_t object, uint32_t address, uint16_t v);
void     nv_dma_write32(NVVGAState *s, uint32_t object, uint32_t address, uint32_t v);
void     nv_dma_write64(NVVGAState *s, uint32_t object, uint32_t address, uint64_t v);
uint32_t nv_swizzle(uint32_t x, uint32_t y, uint32_t width, uint32_t height);
uint64_t nv_get_time_pub(NVVGAState *s);
void     nv_update_irq_pub(NVVGAState *s);
void     nv_redraw_nd_pub(NVVGAState *s, uint32_t offset, uint32_t w, uint32_t h);

/* D3D engine entry point (implemented in geforce_d3d.c) */
void nv_execute_d3d(NVVGAState *s, NvChannel *ch, uint32_t cls,
                    uint32_t method, uint32_t param);

#endif /* GEFORCE_INT_H */
