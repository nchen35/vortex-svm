// Pointer-linked SVM BFS kernel (topology-driven, one level per launch).
//
// Each launch expands the current frontier: every node whose cost == level
// relaxes its neighbors (cost == -1 -> level+1) and raises the `changed` flag.
// A grid-stride loop lets any launch geometry cover all nodes. Concurrent
// writes are idempotent (all set the SAME level+1 value), so the final cost
// array is correct without atomics — a standard property of topology-driven BFS.
//
// The kernel side is identical between bfs_coarse and bfs_fine; only the
// host-side coherence handling differs.

#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t tid    = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = gridDim.x * blockDim.x;

  Node*    nodes = reinterpret_cast<Node*>((uintptr_t)arg->nodes_va);
  int32_t* flag  = reinterpret_cast<int32_t*>((uintptr_t)arg->flag_va);
  int32_t  level = arg->level;

  for (uint32_t v = tid; v < arg->num_nodes; v += stride) {
    if (nodes[v].cost != level)
      continue;
    // Follow the pointer to this node's neighbor-VA list.
    uint64_t* nbrs = reinterpret_cast<uint64_t*>((uintptr_t)nodes[v].neighbors_va);
    uint32_t  deg  = nodes[v].num_neighbors;
    for (uint32_t i = 0; i < deg; ++i) {
      Node* nb = reinterpret_cast<Node*>((uintptr_t)nbrs[i]);
      if (nb->cost == -1) {
        nb->cost = level + 1;
        *flag = 1;
      }
    }
  }
}
