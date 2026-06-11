// Simultaneous-access micro-check — the exact OpenCL fine-grained SVM example.
//
//   float* p = clSVMAlloc(...);           // here: VX_SVM_SHARED single store
//   clEnqueueNDRange(..., mykernel, ...);  // kernel writes p[1],p[3],p[5]
//   clFlush(...);
//   p[0]=0; p[2]=2; p[4]=4;                // host writes, NOT waiting
//   clFinish(...);
//   // p == {0,1,2,3,4,5}
//
// The host writes happen in the window between vx_start_g (which launches the
// kernel on a background thread and returns immediately) and vx_ready_wait
// (which joins) — i.e. concurrently with the running kernel. Correctness relies
// only on host and device touching DISJOINT elements of the one shared buffer.

#include <iostream>
#include <cstring>
#include <cstdint>
#include <vortex.h>
#include "common.h"

#define RT_CHECK(expr) \
  do { int _ret = (expr); if (_ret != 0) { \
    std::cerr << "Error: '" #expr "' returned " << _ret << "\n"; \
    cleanup(); exit(1); } } while (0)

static vx_device_h device   = nullptr;
static vx_buffer_h krnl_buf = nullptr;
static vx_buffer_h args_buf = nullptr;
static void*       buf_svm  = nullptr;

static void cleanup() {
  if (buf_svm)  vx_svm_free(device, buf_svm);
  if (args_buf) vx_mem_free(args_buf);
  if (krnl_buf) vx_mem_free(krnl_buf);
  if (device)   vx_dev_close(device);
}

int main() {
  std::cout << "Simultaneous-access SVM micro-check (OpenCL p[0..5] example)\n";

  RT_CHECK(vx_dev_open(&device));
  RT_CHECK(vx_upload_kernel_file(device, "kernel.vxbin", &krnl_buf));

  // Single-backing-store SVM buffer of 6 floats, zero-initialized.
  uint64_t bytes = N_ELEMS * sizeof(float);
  RT_CHECK(vx_svm_alloc(device, bytes,
                        VX_MEM_READ_WRITE | VX_SVM_SHARED, &buf_svm));

  float*   p      = reinterpret_cast<float*>(buf_svm);
  uint64_t buf_va = vx_svm_dev_addr(device, buf_svm);
  bool same_addr  = ((uint64_t)(uintptr_t)buf_svm == buf_va);

  std::cout << "  host_ptr=0x" << std::hex << (uintptr_t)buf_svm
            << "  device_va=0x" << buf_va << std::dec
            << "  (" << (same_addr ? "SAME ADDRESS (mmap)" : "aliased fallback")
            << ")\n";

  for (uint32_t i = 0; i < N_ELEMS; ++i) p[i] = 0.0f;

  kernel_arg_t args{};
  args.buf_va = buf_va;
  args.n      = N_ELEMS;
  RT_CHECK(vx_upload_bytes(device, &args, sizeof(args), &args_buf));

  // Launch the kernel (returns immediately; kernel runs on a background thread).
  uint32_t grid_dim[3]  = {1, 1, 1};
  uint32_t block_dim[3] = {1, 1, 1};
  RT_CHECK(vx_start_g(device, krnl_buf, args_buf, 1, grid_dim, block_dim, 0));

  // --- concurrent window: host writes the EVEN elements, no vx_* calls ---
  p[0] = 0.0f;
  p[2] = 2.0f;
  p[4] = 4.0f;
  // ---------------------------------------------------------------------

  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // After the join+flush the shared buffer must hold both contributions.
  bool ok = true;
  std::cout << "  result = {";
  for (uint32_t i = 0; i < N_ELEMS; ++i) {
    std::cout << p[i] << (i + 1 < N_ELEMS ? ", " : "");
    if (p[i] != (float)i) ok = false;
  }
  std::cout << "}  (expected {0, 1, 2, 3, 4, 5})\n";

  if (!ok) { std::cerr << "FAILED\n"; cleanup(); return 1; }
  std::cout << "PASSED\n";
  cleanup();
  return 0;
}
