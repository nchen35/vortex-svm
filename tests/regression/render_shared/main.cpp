// render_shared — CPU+GPU split-frame heterogeneous rendering into ONE shared
// SVM framebuffer (single backing store, simultaneous access).
//
// A real graphics use case for fine-grained simultaneous-access SVM: the GPU
// renders the top band of the image while the host CPU renders the bottom band,
// concurrently, into the same buffer — no copies, no map/unmap, no separate
// framebuffers. The host work happens in the window between vx_start_g (which
// launches the kernel on a background thread and returns) and vx_ready_wait
// (which joins), so it overlaps the kernel in wall-clock time.
//
// Correctness: a single-threaded reference renders the WHOLE image with the same
// shader; the co-rendered shared framebuffer must equal it exactly. The image is
// written to out.ppm.

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <vortex.h>
#include "common.h"

#define RT_CHECK(expr) \
  do { int _ret = (expr); if (_ret != 0) { \
    std::cerr << "Error: '" #expr "' returned " << _ret << "\n"; \
    cleanup(); exit(1); } } while (0)

static vx_device_h device   = nullptr;
static vx_buffer_h krnl_buf = nullptr;
static vx_buffer_h args_buf = nullptr;
static void*       fb_svm   = nullptr;

static void cleanup() {
  if (fb_svm)   vx_svm_free(device, fb_svm);
  if (args_buf) vx_mem_free(args_buf);
  if (krnl_buf) vx_mem_free(krnl_buf);
  if (device)   vx_dev_close(device);
}

static void write_ppm(const char* path, const uint32_t* fb, uint32_t w, uint32_t h) {
  FILE* f = std::fopen(path, "wb");
  if (!f) return;
  std::fprintf(f, "P6\n%u %u\n255\n", w, h);
  for (uint32_t i = 0; i < w * h; ++i) {
    uint32_t px = fb[i];
    unsigned char rgb[3] = {
      (unsigned char)((px >> 16) & 0xff),
      (unsigned char)((px >> 8) & 0xff),
      (unsigned char)(px & 0xff)
    };
    std::fwrite(rgb, 1, 3, f);
  }
  std::fclose(f);
}

int main(int argc, char** argv) {
  uint32_t w = IMG_W, h = IMG_H, split = IMG_H / 2;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-w") && i + 1 < argc) w = (uint32_t)std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-h") && i + 1 < argc) h = (uint32_t)std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-s") && i + 1 < argc) split = (uint32_t)std::atoi(argv[++i]);
  }
  if (split > h) split = h;

  std::cout << "render_shared: " << w << "x" << h
            << "  GPU rows [0," << split << ")  CPU rows [" << split << "," << h << ")\n";

  RT_CHECK(vx_dev_open(&device));
  RT_CHECK(vx_upload_kernel_file(device, "kernel.vxbin", &krnl_buf));

  uint64_t num_warps = 1, num_threads = 1;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
  uint32_t block_size = (uint32_t)(num_warps * num_threads);

  // Single shared framebuffer (one uint32 RGB per pixel).
  uint64_t fb_bytes = (uint64_t)w * h * sizeof(uint32_t);
  RT_CHECK(vx_svm_alloc(device, fb_bytes,
                        VX_MEM_READ_WRITE | VX_SVM_SHARED, &fb_svm));
  uint32_t* fb     = reinterpret_cast<uint32_t*>(fb_svm);
  uint64_t  fb_va  = vx_svm_dev_addr(device, fb_svm);
  bool      same   = ((uint64_t)(uintptr_t)fb_svm == fb_va);
  std::cout << "  host_ptr=0x" << std::hex << (uintptr_t)fb_svm
            << " device_va=0x" << fb_va << std::dec
            << "  (" << (same ? "same-address" : "aliased-fallback") << ")\n";
  std::memset(fb, 0, fb_bytes);

  kernel_arg_t args{};
  args.fb_va = fb_va; args.w = w; args.h = h; args.split = split;
  RT_CHECK(vx_upload_bytes(device, &args, sizeof(args), &args_buf));

  // GPU band: launch enough blocks to cover the top band.
  uint32_t gpu_pixels = split * w;
  uint32_t num_blocks = gpu_pixels ? (gpu_pixels + block_size - 1) / block_size : 1;
  uint32_t grid_dim[1]  = {num_blocks};
  uint32_t block_dim[1] = {block_size};

  auto t0 = std::chrono::steady_clock::now();
  RT_CHECK(vx_start_g(device, krnl_buf, args_buf, 1, grid_dim, block_dim, 0));

  // --- concurrent window: host renders the BOTTOM band (no vx_* calls) ---
  auto hb0 = std::chrono::steady_clock::now();
  for (uint32_t y = split; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      fb[y * w + x] = shade(x, y, w, h);
  auto hb1 = std::chrono::steady_clock::now();
  // ----------------------------------------------------------------------

  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  auto t1 = std::chrono::steady_clock::now();

  auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  std::cout << "  host band wall time = " << ms(hb0, hb1) << " ms"
            << " (ran within the kernel's [start,join] = " << ms(t0, t1)
            << " ms window → concurrent)\n";

  // Reference: render the entire image single-threaded with the same shader.
  std::vector<uint32_t> ref((size_t)w * h);
  for (uint32_t y = 0; y < h; ++y)
    for (uint32_t x = 0; x < w; ++x)
      ref[y * w + x] = shade(x, y, w, h);

  // The co-rendered shared framebuffer must match the reference exactly.
  uint32_t mismatches = 0, gpu_band_mm = 0, cpu_band_mm = 0, shown = 0;
  for (uint32_t i = 0; i < w * h; ++i) {
    if (fb[i] != ref[i]) {
      ++mismatches;
      uint32_t y = i / w, x = i % w;
      if (y < split) ++gpu_band_mm; else ++cpu_band_mm;
      if (shown < 8) {
        std::cerr << "    mismatch (" << x << "," << y << ") band="
                  << (y < split ? "GPU" : "CPU") << " fb=0x" << std::hex << fb[i]
                  << " ref=0x" << ref[i] << std::dec << "\n";
        ++shown;
      }
    }
  }
  if (mismatches)
    std::cerr << "  mismatch split: GPU band=" << gpu_band_mm
              << " CPU band=" << cpu_band_mm << "\n";

  write_ppm("out.ppm", fb, w, h);
  std::cout << "  wrote out.ppm  mismatches=" << mismatches << "\n";

  if (mismatches) { std::cerr << "FAILED\n"; cleanup(); return 1; }
  std::cout << "PASSED\n";
  cleanup();
  return 0;
}
