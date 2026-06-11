#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Pointer-linked graph node for SVM BFS.
//
// Unlike the index-CSR BFS (tests/regression/bfs), the graph here is stored as
// a true pointer structure: `neighbors_va` is the device VA of an array of
// neighbor *Node VAs*. The kernel follows these VAs directly through the MMU —
// no CSR flattening, no index arithmetic. This is the data structure SVM makes
// possible to share between host and device without marshalling.
typedef struct Node {
  uint64_t neighbors_va;   // device VA of uint64_t[num_neighbors] of neighbor Node VAs
  uint32_t num_neighbors;  // out-degree
  int32_t  cost;           // BFS distance: -1 = unvisited, 0 = source
} Node;                    // 16 bytes (8 + 4 + 4)

typedef struct {
  uint64_t nodes_va;   // device VA of Node[num_nodes]
  uint64_t flag_va;    // device VA of int32_t "changed" flag
  uint32_t num_nodes;
  int32_t  level;      // BFS level currently being expanded
} kernel_arg_t;

#endif
