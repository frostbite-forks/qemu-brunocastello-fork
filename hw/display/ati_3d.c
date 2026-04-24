/*
 * QEMU ATI Rage 128 — 3D passthrough via Unix socket
 *
 * Intercepts PM4 command ring writes from the guest and forwards them
 * to a host-side Metal renderer daemon listening on ATI_3D_SOCK_PATH.
 *
 * Protocol (host renderer side):
 *   Each message is a 4-byte little-endian length (in bytes) followed by
 *   that many bytes of raw PM4 dwords (little-endian, as written by guest).
 *   The renderer processes the batch asynchronously; no reply is expected.
 *   RBBM_STATUS always reports ENGINE_IDLE to keep the guest moving.
 */

#include "qemu/osdep.h"
#include "ati_int.h"
#include "exec/cpu-common.h"
#include "qemu/log.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#define ATI_3D_SOCK_PATH  "/tmp/qemu-ati3d.sock"
#define ATI_3D_MAX_BATCH  (64 * 1024)   /* max bytes per flush */

/* Connect (or reconnect) to the host renderer daemon, non-blocking attempt. */
void ati_3d_connect(ATIVGAState *s)
{
    struct sockaddr_un addr;
    int fd;

    if (s->pm4.sock_fd >= 0) {
        return; /* already connected */
    }

    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, ATI_3D_SOCK_PATH, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return;
    }

    /* Set non-blocking after connect so send() doesn't stall the vCPU */
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);

    s->pm4.sock_fd = fd;
}

void ati_3d_disconnect(ATIVGAState *s)
{
    if (s->pm4.sock_fd >= 0) {
        close(s->pm4.sock_fd);
        s->pm4.sock_fd = -1;
    }
}

/*
 * Called when the guest advances PM4_BUFFER_DL_WPTR.
 * Reads new commands from the guest ring buffer and sends them to the
 * renderer daemon.  The ring buffer wraps at pm4.buf_size dwords.
 */
void ati_3d_flush(ATIVGAState *s)
{
    uint32_t rptr = s->pm4.rptr;
    uint32_t wptr = s->pm4.wptr;
    uint32_t buf_dwords = s->pm4.buf_size;

    if (!buf_dwords || s->pm4.buf_addr == 0) {
        s->pm4.rptr = wptr;
        return;
    }

    /* Reconnect if renderer restarted */
    if (s->pm4.sock_fd < 0) {
        ati_3d_connect(s);
    }

    while (rptr != wptr) {
        /* Number of dwords available without wrap */
        uint32_t avail = (wptr > rptr)
            ? (wptr - rptr)
            : (buf_dwords - rptr);

        if (avail == 0) {
            break;
        }

        uint32_t bytes = avail * sizeof(uint32_t);
        if (bytes > ATI_3D_MAX_BATCH) {
            bytes = ATI_3D_MAX_BATCH;
            avail = bytes / sizeof(uint32_t);
        }

        /* Read from guest physical ring buffer */
        uint8_t *tmp = g_malloc(bytes);
        cpu_physical_memory_read(s->pm4.buf_addr + rptr * sizeof(uint32_t),
                                 tmp, bytes);

        /* Send to renderer: 4-byte LE length header + data */
        if (s->pm4.sock_fd >= 0) {
            uint32_t hdr = cpu_to_le32(bytes);
            /* Best-effort send; renderer absence is non-fatal */
            if (send(s->pm4.sock_fd, &hdr, sizeof(hdr), MSG_NOSIGNAL) < 0 ||
                send(s->pm4.sock_fd, tmp, bytes, MSG_NOSIGNAL) < 0) {
                /* Renderer gone — disconnect and carry on */
                ati_3d_disconnect(s);
            }
        }

        g_free(tmp);

        rptr = (rptr + avail) % buf_dwords;
    }

    s->pm4.rptr = wptr;

    /* Write updated rptr back to guest memory if requested */
    if (s->pm4.rptr_addr) {
        uint32_t val = cpu_to_le32(s->pm4.rptr);
        cpu_physical_memory_write(s->pm4.rptr_addr, &val, sizeof(val));
    }
}
