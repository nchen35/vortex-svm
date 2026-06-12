// bfs_csr — the NO-SVM baseline of the matched BFS pair.
//
// Frontier-driven BFS (adapted from the Rodinia-style test in
// tests/regression/bfs): the graph is FLATTENED into index-CSR arrays, device
// buffers are managed with vx_mem_alloc, and data moves with explicit
// vx_copy_to_dev / vx_copy_from_dev every level. All host<->device traffic is
// counted and reported.
//
// Matched twin: tests/regression/bfs_ptr (same generator+seed, same algorithm,
// same launch geometry; pointer representation + SVM instead).

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
static vx_buffer_h nodes_buf = nullptr, edges_buf = nullptr, visit_buf = nullptr,
                   nextmask_buf = nullptr, frontier_buf = nullptr, cost_buf = nullptr;

static void cleanup() {
  if (nodes_buf)    vx_mem_free(nodes_buf);
  if (edges_buf)    vx_mem_free(edges_buf);
  if (visit_buf)    vx_mem_free(visit_buf);
  if (nextmask_buf) vx_mem_free(nextmask_buf);
  if (frontier_buf) vx_mem_free(frontier_buf);
  if (cost_buf)     vx_mem_free(cost_buf);
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
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-n") && i + 1 < argc)
      num_nodes = (uint32_t)std::atoi(argv[++i]);
  }
  std::srand(50);

  std::cout << "bfs_csr (no SVM, index-CSR + explicit copies): num_nodes="
            << num_nodes << "\n";

  std::vector<std::vector<uint32_t>> adj;
  gen_graph(num_nodes, max_deg, adj);

  // Flatten to CSR (the marshalling step SVM eliminates).
  std::vector<Node>    h_nodes(num_nodes);
  std::vector<int32_t> h_edges;
  for (uint32_t u = 0; u < num_nodes; ++u) {
    h_nodes[u].starting    = (int32_t)h_edges.size();
    h_nodes[u].no_of_edges = (int32_t)adj[u].size();
    for (uint32_t v : adj[u]) h_edges.push_back((int32_t)v);
  }
  uint32_t num_edges = (uint32_t)h_edges.size();
  if (num_edges == 0) h_edges.push_back(0), num_edges = 1;

  RT_CHECK(vx_dev_open(&device));
  RT_CHECK(vx_upload_kernel_file(device, "kernel.vxbin", &krnl_buf));

  uint64_t nodes_bytes    = num_nodes * sizeof(Node);
  uint64_t edges_bytes    = num_edges * sizeof(int32_t);
  uint64_t mask_bytes     = num_nodes * sizeof(uint8_t);
  uint64_t frontier_bytes = num_nodes * sizeof(uint32_t);
  uint64_t cost_bytes     = num_nodes * sizeof(int32_t);

  kernel_arg_t ka{};
  RT_CHECK(vx_mem_alloc(device, nodes_bytes,    VX_MEM_READ_WRITE, &nodes_buf));
  RT_CHECK(vx_mem_address(nodes_buf,    &ka.nodes_addr));
  RT_CHECK(vx_mem_alloc(device, edges_bytes,    VX_MEM_READ_WRITE, &edges_buf));
  RT_CHECK(vx_mem_address(edges_buf,    &ka.edges_addr));
  RT_CHECK(vx_mem_alloc(device, mask_bytes,     VX_MEM_READ_WRITE, &visit_buf));
  RT_CHECK(vx_mem_address(visit_buf,    &ka.visit_addr));
  RT_CHECK(vx_mem_alloc(device, mask_bytes,     VX_MEM_READ_WRITE, &nextmask_buf));
  RT_CHECK(vx_mem_address(nextmask_buf, &ka.nextmask_addr));
  RT_CHECK(vx_mem_alloc(device, frontier_bytes, VX_MEM_READ_WRITE, &frontier_buf));
  RT_CHECK(vx_mem_address(frontier_buf, &ka.frontier_addr));
  RT_CHECK(vx_mem_alloc(device, cost_bytes,     VX_MEM_READ_WRITE, &cost_buf));
  RT_CHECK(vx_mem_address(cost_buf,     &ka.cost_addr));

  // Host state.
  uint32_t source = 0;
  std::vector<uint8_t>  h_visit(num_nodes, 0);
  std::vector<uint8_t>  h_nextmask(num_nodes, 0);
  std::vector<int32_t>  h_cost(num_nodes, -1);
  std::vector<uint32_t> h_frontier;
  h_visit[source] = 1;
  h_cost[source]  = 0;
  h_frontier.push_back(source);

  // Traffic counters (the metric SVM-shared eliminates).
  uint64_t bytes_up = 0, bytes_down = 0, args_bytes = 0;

  // Cumulative device counters: the per-launch PERF stats reset on every run,
  // so we query MCYCLE/MINSTRET after each launch and sum across launches.
  uint64_t dev_cycles = 0, dev_instrs = 0;

  // One-time uploads: graph + initial cost.
  RT_CHECK(vx_copy_to_dev(nodes_buf, h_nodes.data(), 0, nodes_bytes)); bytes_up += nodes_bytes;
  RT_CHECK(vx_copy_to_dev(edges_buf, h_edges.data(), 0, edges_bytes)); bytes_up += edges_bytes;
  RT_CHECK(vx_copy_to_dev(cost_buf,  h_cost.data(),  0, cost_bytes));  bytes_up += cost_bytes;

  // BFS level loop with host-side frontier compaction.
  std::cout << "  frontier sizes:";
  uint32_t levels = 0;
  while (!h_frontier.empty()) {
    uint32_t frontier_size = (uint32_t)h_frontier.size();
    std::cout << " " << frontier_size;

    // Per-level uploads: frontier, visit (changed by compaction), zeroed nextmask.
    RT_CHECK(vx_copy_to_dev(frontier_buf, h_frontier.data(), 0,
                            frontier_size * sizeof(uint32_t)));
    bytes_up += frontier_size * sizeof(uint32_t);
    RT_CHECK(vx_copy_to_dev(visit_buf, h_visit.data(), 0, mask_bytes));
    bytes_up += mask_bytes;
    std::fill(h_nextmask.begin(), h_nextmask.end(), 0);
    RT_CHECK(vx_copy_to_dev(nextmask_buf, h_nextmask.data(), 0, mask_bytes));
    bytes_up += mask_bytes;

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

    // Per-level download: nextmask (needed to compact the next frontier).
    RT_CHECK(vx_copy_from_dev(h_nextmask.data(), nextmask_buf, 0, mask_bytes));
    bytes_down += mask_bytes;

    h_frontier.clear();
    for (uint32_t v = 0; v < num_nodes; ++v) {
      if (h_nextmask[v] && !h_visit[v]) {
        h_visit[v] = 1;
        h_frontier.push_back(v);
      }
    }
    ++levels;
  }
  std::cout << "\n";

  // Final download: cost.
  RT_CHECK(vx_copy_from_dev(h_cost.data(), cost_buf, 0, cost_bytes));
  bytes_down += cost_bytes;

  // Verify against the host reference.
  std::vector<int32_t> ref;
  bfs_ref(adj, source, ref);
  uint32_t mismatches = 0;
  for (uint32_t v = 0; v < num_nodes; ++v)
    if (h_cost[v] != ref[v]) ++mismatches;

  std::cout << "  levels=" << levels << "  mismatches=" << mismatches << "\n";
  std::cout << "  device totals: instrs=" << dev_instrs << " cycles=" << dev_cycles
            << " (summed over " << levels << " launches)\n";
  std::cout << "  data bytes copied: up=" << bytes_up << " down=" << bytes_down
            << " total=" << (bytes_up + bytes_down)
            << "  (+args=" << args_bytes << ")\n";

  if (mismatches) { std::cerr << "FAILED\n"; cleanup(); return 1; }
  std::cout << "PASSED\n";
  cleanup();
  return 0;
}
