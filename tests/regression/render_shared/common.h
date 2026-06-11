#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Default framebuffer size and the GPU/CPU split row (overridable via args).
#define IMG_W   64
#define IMG_H   48

typedef struct {
  uint64_t fb_va;   // device VA of the shared RGB framebuffer (uint32_t per pixel)
  uint32_t w;
  uint32_t h;
  uint32_t split;   // GPU renders rows [0, split); CPU renders [split, h)
  uint32_t _pad;
} kernel_arg_t;

// Per-pixel shader: a deterministic procedural pattern (radial rings + XOR
// texture over an RGB gradient), packed 0x00RRGGBB.
//
// IMPORTANT: integer-only and NON-ITERATIVE on purpose. The host (x86) and the
// device (RV32) must compute BIT-IDENTICAL pixels so the co-rendered shared
// framebuffer can be compared EXACTLY against a single-thread reference. We avoid
// floats (low-bit differences across ISAs) and avoid any feedback/iteration
// (which would amplify a tiny per-op difference into flipped pixels — as a
// fixed-point fractal does). Every op here (mul, div, xor, shift on
// non-negatives) is exact and identical on both compilers, so the seam between
// the GPU and CPU bands is provably invisible. The same function is shared
// verbatim by the kernel and the host.
static inline uint32_t shade(uint32_t px, uint32_t py, uint32_t w, uint32_t h) {
  // Radial term: integer squared distance from the image center.
  int32_t dx = (int32_t)px - (int32_t)(w / 2);
  int32_t dy = (int32_t)py - (int32_t)(h / 2);
  uint32_t d2 = (uint32_t)(dx * dx + dy * dy);   // small, non-negative

  uint32_t r = (px * 255u) / w;                  // horizontal gradient
  uint32_t g = (py * 255u) / h;                  // vertical gradient
  uint32_t b = ((px ^ py) ^ (d2 >> 3)) & 0xffu;  // XOR texture + rings
  return (r << 16) | (g << 8) | b;
}

#endif
