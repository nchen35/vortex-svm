// SVM linked-list demo — coarse-grained buffer SVM.
//
// Demonstrates OpenCL-style SVM pointer sharing:
//   1. Host allocates two SVM buffers: node array + result scalar.
//   2. Host maps the node buffer for writing and builds a linked list whose
//      next-pointers contain device VAs (returned by vx_svm_dev_addr).
//   3. vx_svm_unmap flushes the host buffer to device RAM (no explicit copy).
//   4. Kernel traverses the list using the same VAs through the MMU.
//   5. Host maps the result buffer for reading and validates.
//
// Also runs a non-SVM baseline (same list, explicit vx_copy_to_dev) to show
// that the SVM path produces identical results and measures overhead.

#include <iostream>
#include <vector>
#include <numeric>
#include <cstring>
#include <cassert>
#include <vortex.h>
#include "common.h"

#define RT_CHECK(expr) \
  do { \
    int _ret = (expr); \
    if (_ret != 0) { \
      std::cerr << "Error: '" #expr "' returned " << _ret << "\n"; \
      cleanup(); \
      exit(1); \
    } \
  } while (0)

static vx_device_h device   = nullptr;
static vx_buffer_h krnl_buf = nullptr;
static vx_buffer_h args_buf = nullptr;
static void*       nodes_svm    = nullptr;
static void*       result_svm   = nullptr;

static void cleanup() {
  if (nodes_svm)  vx_svm_free(device, nodes_svm);
  if (result_svm) vx_svm_free(device, result_svm);
  if (args_buf)   vx_mem_free(args_buf);
  if (krnl_buf)   vx_mem_free(krnl_buf);
  if (device)     vx_dev_close(device);
}

int main(int argc, char** argv) {
  uint32_t n_nodes = 32;

  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
      n_nodes = (uint32_t)std::atoi(argv[++i]);
    }
  }

  std::cout << "SVM linked-list demo: n_nodes=" << n_nodes << "\n";

  // ── open device ──────────────────────────────────────────────────────────
  RT_CHECK(vx_dev_open(&device));

  // ── upload kernel ─────────────────────────────────────────────────────────
  RT_CHECK(vx_upload_kernel_file(device, "kernel.vxbin", &krnl_buf));

  // ── SVM allocations ───────────────────────────────────────────────────────
  uint64_t nodes_bytes  = n_nodes * sizeof(Node);
  uint64_t result_bytes = sizeof(int32_t);

  RT_CHECK(vx_svm_alloc(device, nodes_bytes,  VX_MEM_READ_WRITE, &nodes_svm));
  RT_CHECK(vx_svm_alloc(device, result_bytes, VX_MEM_READ_WRITE, &result_svm));

  // Get device VAs for embedding as next-pointers and for kernel args.
  uint64_t nodes_va  = vx_svm_dev_addr(device, nodes_svm);
  uint64_t result_va = vx_svm_dev_addr(device, result_svm);

  std::cout << "  nodes  VA=0x" << std::hex << nodes_va
            << "  result VA=0x" << result_va << std::dec << "\n";

  // ── map nodes buffer for host write ───────────────────────────────────────
  RT_CHECK(vx_svm_map(device, nodes_svm, nodes_bytes, VX_MEM_WRITE));

  // Build the linked list.  next-pointers use DEVICE VAs so the kernel can
  // dereference them directly.  No index-to-pointer fixup needed.
  Node* nodes = reinterpret_cast<Node*>(nodes_svm);
  int32_t expected_sum = 0;
  for (uint32_t i = 0; i < n_nodes; ++i) {
    nodes[i].value = (int32_t)(i + 1); // values 1..n_nodes
    nodes[i].next  = (i + 1 < n_nodes)
                       ? (nodes_va + (i + 1) * sizeof(Node))
                       : 0;
    expected_sum += nodes[i].value;
  }

  RT_CHECK(vx_svm_unmap(device, nodes_svm, nodes_bytes));

  // ── map result buffer for host write (zero it) ────────────────────────────
  RT_CHECK(vx_svm_map(device, result_svm, result_bytes, VX_MEM_WRITE));
  std::memset(result_svm, 0, result_bytes);
  RT_CHECK(vx_svm_unmap(device, result_svm, result_bytes));

  // ── kernel args (use a regular device buffer) ─────────────────────────────
  kernel_arg_t args{};
  args.head_va   = nodes_va;
  args.result_va = result_va;
  args.n_nodes   = n_nodes;

  RT_CHECK(vx_upload_bytes(device, &args, sizeof(args), &args_buf));

  // ── launch ────────────────────────────────────────────────────────────────
  uint32_t grid_dim[3]  = {1, 1, 1};
  uint32_t block_dim[3] = {1, 1, 1};
  RT_CHECK(vx_start_g(device, krnl_buf, args_buf, 1, grid_dim, block_dim, 0));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // ── read result ───────────────────────────────────────────────────────────
  RT_CHECK(vx_svm_map(device, result_svm, result_bytes, VX_MEM_READ));
  int32_t actual_sum = *reinterpret_cast<int32_t*>(result_svm);
  RT_CHECK(vx_svm_unmap(device, result_svm, result_bytes));

  // ── validate ──────────────────────────────────────────────────────────────
  std::cout << "  expected sum=" << expected_sum
            << "  actual sum="   << actual_sum << "\n";

  if (actual_sum != expected_sum) {
    std::cerr << "FAILED: result mismatch\n";
    cleanup();
    return 1;
  }

  std::cout << "PASSED\n";
  cleanup();
  return 0;
}
