// Frontier-driven POINTER BFS kernel — the matched twin of bfs_csr/kernel.cpp.
//
// Identical algorithm and branch structure; the only difference is the data
// representation: the frontier holds Node VAs (not indices), neighbors are Node
// VAs (not indices into flat arrays), and each node's BFS state lives inside the
// Node struct (one 24 B access) instead of three separate flat arrays.

#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->frontier_size)
    return;

  auto *__restrict frontier =
      reinterpret_cast<uint64_t *>((uintptr_t)arg->frontier_va);

  Node*   v  = reinterpret_cast<Node *>((uintptr_t)frontier[tid]);
  int32_t cv = v->cost + 1;
  auto *__restrict nbrs =
      reinterpret_cast<uint64_t *>((uintptr_t)v->neighbors_va);
  uint32_t deg = v->num_neighbors;

  for (uint32_t i = 0; i < deg; ++i) {
    Node* nb = reinterpret_cast<Node *>((uintptr_t)nbrs[i]);
    if (!nb->visited) {
      nb->in_next = 1;
      nb->cost = cv;
    }
  }
}
