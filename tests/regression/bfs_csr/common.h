#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Index-CSR graph representation (adapted from the Rodinia-style BFS already in
// Vortex, tests/regression/bfs). Each node stores an offset+count into one flat
// edges[] array of neighbor INDICES. This is the "flattened" form a host must
// marshal a pointer graph into when there is no SVM.
struct Node {
  int32_t starting;     // first edge index in edges[]
  int32_t no_of_edges;  // out-degree
};

typedef struct {
  uint32_t frontier_size;
  uint32_t _pad;

  uint64_t nodes_addr;     // Node[num_nodes]
  uint64_t edges_addr;     // int32_t[num_edges]  (neighbor indices)
  uint64_t visit_addr;     // uint8_t[num_nodes]
  uint64_t nextmask_addr;  // uint8_t[num_nodes]
  uint64_t frontier_addr;  // uint32_t[frontier_size] (node indices)
  uint64_t cost_addr;      // int32_t[num_nodes]
} kernel_arg_t;

#endif
