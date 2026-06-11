// Pointer-linked SVM BFS — FINE-GRAINED buffer SVM.
//
// Builds a random graph as a true pointer structure in SVM memory: each node
// holds the device VA of its neighbor-VA list. The kernel follows those VAs
// through the MMU. BFS runs level-by-level from a host loop driven by a shared
// `changed` flag.
//
// Fine-grained: NO vx_svm_map / vx_svm_unmap. Coherence is automatic at kernel
// launch (host->device) and completion (device->host). The host writes the
// `changed` flag and reads it back by simply touching the buffer.

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <queue>
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
static void*       nodes_svm   = nullptr;
static void*       nbrs_svm    = nullptr;
static void*       flag_svm    = nullptr;

static void cleanup() {
  if (nodes_svm) vx_svm_free(device, nodes_svm);
  if (nbrs_svm)  vx_svm_free(device, nbrs_svm);
  if (flag_svm)  vx_svm_free(device, flag_svm);
  if (args_buf)  vx_mem_free(args_buf);
  if (krnl_buf)  vx_mem_free(krnl_buf);
  if (device)    vx_dev_close(device);
}

// Generate an undirected random graph as adjacency lists.
static void gen_graph(uint32_t n, uint32_t max_deg,
                      std::vector<std::vector<uint32_t>>& adj) {
  adj.assign(n, {});
  for (uint32_t u = 0; u < n; ++u) {
    uint32_t deg = 1 + (std::rand() % max_deg);
    for (uint32_t k = 0; k < deg; ++k) {
      uint32_t v = std::rand() % n;
      if (v == u) continue;
      adj[u].push_back(v);
      adj[v].push_back(u); // undirected
    }
  }
}

// Reference BFS on the host (golden result).
static void bfs_ref(const std::vector<std::vector<uint32_t>>& adj,
                    uint32_t source, std::vector<int32_t>& cost) {
  cost.assign(adj.size(), -1);
  std::queue<uint32_t> q;
  cost[source] = 0;
  q.push(source);
  while (!q.empty()) {
    uint32_t u = q.front(); q.pop();
    for (uint32_t v : adj[u]) {
      if (cost[v] == -1) { cost[v] = cost[u] + 1; q.push(v); }
    }
  }
}

int main(int argc, char** argv) {
  uint32_t num_nodes = 64;
  uint32_t max_deg   = 4;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-n") && i + 1 < argc)
      num_nodes = (uint32_t)std::atoi(argv[++i]);
  }
  std::srand(50);

  std::cout << "SVM BFS (fine-grained): num_nodes=" << num_nodes << "\n";

  std::vector<std::vector<uint32_t>> adj;
  gen_graph(num_nodes, max_deg, adj);

  uint64_t total_edges = 0;
  for (auto& a : adj) total_edges += a.size();
  if (total_edges == 0) total_edges = 1; // avoid zero-size alloc

  RT_CHECK(vx_dev_open(&device));
  RT_CHECK(vx_upload_kernel_file(device, "kernel.vxbin", &krnl_buf));

  // block size = one CTA on one core (single core keeps caches coherent within
  // a launch; the grid-stride kernel still covers all nodes).
  uint64_t num_warps = 1, num_threads = 1;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));
  uint32_t block_size = (uint32_t)(num_warps * num_threads);

  // ── fine-grained SVM allocations ──────────────────────────────────────────
  uint64_t nodes_bytes = num_nodes * sizeof(Node);
  uint64_t nbrs_bytes  = total_edges * sizeof(uint64_t);
  uint64_t flag_bytes  = sizeof(int32_t);
  int fflags = VX_MEM_READ_WRITE | VX_SVM_FINE_GRAINED;

  RT_CHECK(vx_svm_alloc(device, nodes_bytes, fflags, &nodes_svm));
  RT_CHECK(vx_svm_alloc(device, nbrs_bytes,  fflags, &nbrs_svm));
  RT_CHECK(vx_svm_alloc(device, flag_bytes,  fflags, &flag_svm));

  uint64_t nodes_va = vx_svm_dev_addr(device, nodes_svm);
  uint64_t nbrs_va  = vx_svm_dev_addr(device, nbrs_svm);
  uint64_t flag_va  = vx_svm_dev_addr(device, flag_svm);

  // ── build the pointer-linked graph directly (no map/unmap) ────────────────
  Node*     nodes = reinterpret_cast<Node*>(nodes_svm);
  uint64_t* nbrs  = reinterpret_cast<uint64_t*>(nbrs_svm);
  uint32_t  source = 0;
  uint64_t  off = 0;
  for (uint32_t u = 0; u < num_nodes; ++u) {
    nodes[u].neighbors_va  = nbrs_va + off * sizeof(uint64_t);
    nodes[u].num_neighbors = (uint32_t)adj[u].size();
    nodes[u].cost          = (u == source) ? 0 : -1;
    for (uint32_t v : adj[u])
      nbrs[off++] = nodes_va + (uint64_t)v * sizeof(Node); // neighbor Node VA
  }

  // ── BFS level loop (fine-grained: direct flag access, no map/unmap) ────────
  // Launch multiple thread blocks so the per-node work spreads across CTAs (and
  // cores, on a multi-core build). The grid-stride kernel covers any geometry.
  int32_t* flag = reinterpret_cast<int32_t*>(flag_svm);
  uint32_t num_blocks   = (num_nodes + block_size - 1) / block_size;
  uint32_t grid_dim[1]  = {num_blocks};
  uint32_t block_dim[1] = {block_size};

  int32_t level = 0;
  for (;;) {
    *flag = 0;                            // direct host write (synced at launch)

    kernel_arg_t args{};
    args.nodes_va  = nodes_va;
    args.flag_va   = flag_va;
    args.num_nodes = num_nodes;
    args.level     = level;
    RT_CHECK(vx_upload_bytes(device, &args, sizeof(args), &args_buf));

    RT_CHECK(vx_start_g(device, krnl_buf, args_buf, 1, grid_dim, block_dim, 0));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    int32_t changed = *flag;              // direct host read (synced at wait)
    if (!changed) break;
    ++level;
  }

  // ── read results directly and verify ──────────────────────────────────────
  std::vector<int32_t> ref;
  bfs_ref(adj, source, ref);

  uint32_t mismatches = 0;
  for (uint32_t u = 0; u < num_nodes; ++u) {
    if (nodes[u].cost != ref[u]) {
      if (mismatches < 10)
        std::cerr << "  mismatch node " << u << ": device=" << nodes[u].cost
                  << " ref=" << ref[u] << "\n";
      ++mismatches;
    }
  }

  std::cout << "  levels=" << level << "  mismatches=" << mismatches << "\n";
  if (mismatches) {
    std::cerr << "FAILED\n";
    cleanup();
    return 1;
  }
  std::cout << "PASSED\n";
  cleanup();
  return 0;
}
