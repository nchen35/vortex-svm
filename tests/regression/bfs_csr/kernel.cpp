// Frontier-driven CSR BFS kernel — clone of the Rodinia-style kernel in
// tests/regression/bfs. One thread per frontier entry: load the node's CSR
// range, walk its neighbor INDICES, relax unvisited neighbors.
//
// This is the matched twin of bfs_ptr/kernel.cpp: identical algorithm and branch
// structure; the only difference is the data representation (indices into flat
// arrays here vs device-VA pointers there).

#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->frontier_size)
    return;

  auto *__restrict nodes    = reinterpret_cast<Node *>((uintptr_t)arg->nodes_addr);
  auto *__restrict edges    = reinterpret_cast<int32_t *>((uintptr_t)arg->edges_addr);
  auto *__restrict visit    = reinterpret_cast<uint8_t *>((uintptr_t)arg->visit_addr);
  auto *__restrict nextmask = reinterpret_cast<uint8_t *>((uintptr_t)arg->nextmask_addr);
  auto *__restrict frontier = reinterpret_cast<uint32_t *>((uintptr_t)arg->frontier_addr);
  auto *__restrict cost     = reinterpret_cast<int32_t *>((uintptr_t)arg->cost_addr);

  uint32_t v   = frontier[tid];
  uint32_t end = nodes[v].starting + nodes[v].no_of_edges;
  int32_t  cv  = cost[v] + 1;

  for (uint32_t i = nodes[v].starting; i < end; ++i) {
    uint32_t nid = edges[i];
    if (!visit[nid]) {
      nextmask[nid] = 1;
      cost[nid] = cv;
    }
  }
}
