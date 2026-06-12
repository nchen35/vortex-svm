// bfs_ptr — the SVM side of the matched BFS pair.
//
// Same frontier-driven algorithm, generator (seed), and launch geometry as
// bfs_csr, but the graph is a true POINTER structure in SVM memory: no CSR
// flattening, and (in shared mode) ZERO bytes copied between host and device.
//
//   -m shared (default): VX_SVM_SHARED — single backing store. Host and device
//       touch the same bytes; no copies exist at all.
//   -m fine:             VX_SVM_FINE_GRAINED — copy-based fallback. The driver
//       implicitly syncs every SVM buffer in full at each kernel launch and
//       completion; we report that (computed) traffic for comparison.

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <queue>
#include <vortex.h>
#include <VX_types.h>
#include "common.h"

#define RT_CHECK(expr) \
  do { int _ret = (expr); if (_ret != 0) { \
    std::cerr << "Error: '" #expr "' returned " << _ret << "\n"; \
    cleanup(); exit(1); } } while (0)

static vx_device_h device = nullptr;
static vx_buffer_h krnl_buf = nullptr, args_buf = nullptr;
static void *nodes_svm = nullptr, *nbrs_svm = nullptr, *frontier_svm = nullptr;

static void cleanup() {
  if (nodes_svm)    vx_svm_free(device, nodes_svm);
  if (nbrs_svm)     vx_svm_free(device, nbrs_svm);
  if (frontier_svm) vx_svm_free(device, frontier_svm);
  if (args_buf)     vx_mem_free(args_buf);
  if (krnl_buf)     vx_mem_free(krnl_buf);
  if (device)       vx_dev_close(device);
}

// Shared graph generator — identical (code + seed) in bfs_csr and bfs_ptr so
// both build the SAME adjacency.
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
  bool     shared    = true;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-n") && i + 1 < argc)
      num_nodes = (uint32_t)std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "-m") && i + 1 < argc)
      shared = (std::strcmp(argv[++i], "fine") != 0);
  }
  std::srand(50);

  std::cout << "bfs_ptr (SVM " << (shared ? "shared" : "fine")
            << ", pointer graph): num_nodes=" << num_nodes << "\n";

  std::vector<std::vector<uint32_t>> adj;
  gen_graph(num_nodes, max_deg, adj);

  uint64_t total_edges = 0;
  for (auto& a : adj) total_edges += a.size();
  if (total_edges == 0) total_edges = 1;

  RT_CHECK(vx_dev_open(&device));
  RT_CHECK(vx_upload_kernel_file(device, "kernel.vxbin", &krnl_buf));

  // SVM allocations: node array, neighbor-VA pool, frontier (of Node VAs).
  int mode_flag = shared ? VX_SVM_SHARED : VX_SVM_FINE_GRAINED;
  uint64_t nodes_bytes    = num_nodes * sizeof(Node);
  uint64_t nbrs_bytes     = total_edges * sizeof(uint64_t);
  uint64_t frontier_bytes = num_nodes * sizeof(uint64_t);

  RT_CHECK(vx_svm_alloc(device, nodes_bytes,    VX_MEM_READ_WRITE | mode_flag, &nodes_svm));
  RT_CHECK(vx_svm_alloc(device, nbrs_bytes,     VX_MEM_READ_WRITE | mode_flag, &nbrs_svm));
  RT_CHECK(vx_svm_alloc(device, frontier_bytes, VX_MEM_READ_WRITE | mode_flag, &frontier_svm));

  uint64_t nodes_va    = vx_svm_dev_addr(device, nodes_svm);
  uint64_t nbrs_va     = vx_svm_dev_addr(device, nbrs_svm);
  uint64_t frontier_va = vx_svm_dev_addr(device, frontier_svm);

  // Build the pointer graph IN PLACE (no flattening, no upload).
  Node*     nodes    = reinterpret_cast<Node*>(nodes_svm);
  uint64_t* nbrs     = reinterpret_cast<uint64_t*>(nbrs_svm);
  uint64_t* frontier = reinterpret_cast<uint64_t*>(frontier_svm);
  uint32_t  source   = 0;
  uint64_t  off      = 0;
  for (uint32_t u = 0; u < num_nodes; ++u) {
    nodes[u].neighbors_va  = nbrs_va + off * sizeof(uint64_t);
    nodes[u].num_neighbors = (uint32_t)adj[u].size();
    nodes[u].cost          = (u == source) ? 0 : -1;
    nodes[u].visited       = (u == source) ? 1 : 0;
    nodes[u].in_next       = 0;
    for (uint32_t v : adj[u])
      nbrs[off++] = nodes_va + (uint64_t)v * sizeof(Node);
  }
  frontier[0] = nodes_va + (uint64_t)source * sizeof(Node);
  uint32_t frontier_size = 1;

  // BFS level loop with host-side frontier compaction (direct SVM access; no
  // map/unmap, no copies in either mode).
  uint64_t args_bytes = 0;

  // Cumulative device counters: the per-launch PERF stats reset on every run,
  // so we query MCYCLE/MINSTRET after each launch and sum across launches.
  uint64_t dev_cycles = 0, dev_instrs = 0;

  std::cout << "  frontier sizes:";
  uint32_t levels = 0;
  while (frontier_size != 0) {
    std::cout << " " << frontier_size;

    kernel_arg_t ka{};
    ka.frontier_va   = frontier_va;
    ka.frontier_size = frontier_size;
    if (args_buf) { vx_mem_free(args_buf); args_buf = nullptr; }
    RT_CHECK(vx_upload_bytes(device, &ka, sizeof(ka), &args_buf));
    args_bytes += sizeof(ka);

    uint32_t grid_dim[1], block_dim[1];
    RT_CHECK(vx_max_occupancy_grid(device, 1, &frontier_size, grid_dim, block_dim));
    RT_CHECK(vx_start_g(device, krnl_buf, args_buf, 1, grid_dim, block_dim, 0));
    RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

    // Accumulate this launch's device counters (they reset per launch).
    {
      uint64_t c = 0, i = 0;
      RT_CHECK(vx_mpm_query(device, VX_DCR_MPM_CLASS_BASE, VX_CSR_MCYCLE, 0, &c));
      RT_CHECK(vx_mpm_query(device, VX_DCR_MPM_CLASS_BASE, VX_CSR_MINSTRET, 0, &i));
      dev_cycles += c; dev_instrs += i;
    }

    // Compact the next frontier by scanning the node array directly.
    frontier_size = 0;
    for (uint32_t v = 0; v < num_nodes; ++v) {
      if (nodes[v].in_next && !nodes[v].visited) {
        nodes[v].visited = 1;
        frontier[frontier_size++] = nodes_va + (uint64_t)v * sizeof(Node);
      }
      nodes[v].in_next = 0;
    }
    ++levels;
  }
  std::cout << "\n";

  // Verify against the host reference (read costs directly from SVM).
  std::vector<int32_t> ref;
  bfs_ref(adj, source, ref);
  uint32_t mismatches = 0;
  for (uint32_t v = 0; v < num_nodes; ++v)
    if (nodes[v].cost != ref[v]) ++mismatches;

  std::cout << "  levels=" << levels << "  mismatches=" << mismatches << "\n";
  std::cout << "  device totals: instrs=" << dev_instrs << " cycles=" << dev_cycles
            << " (summed over " << levels << " launches)\n";
  if (shared) {
    std::cout << "  data bytes copied: 0 (single backing store)"
              << "  (+args=" << args_bytes << ")\n";
  } else {
    // Fine-grained: the driver syncs every SVM buffer in full at each kernel
    // launch (host->device) and completion (device->host). Report that traffic.
    uint64_t per_boundary = nodes_bytes + nbrs_bytes + frontier_bytes;
    uint64_t implicit = per_boundary * 2ull * levels;
    std::cout << "  data bytes copied (implicit syncs): " << implicit
              << "  = (" << per_boundary << " B working set) x 2 x "
              << levels << " launches  (+args=" << args_bytes << ")\n";
  }

  if (mismatches) { std::cerr << "FAILED\n"; cleanup(); return 1; }
  std::cout << "PASSED\n";
  cleanup();
  return 0;
}
