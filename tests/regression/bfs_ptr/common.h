#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Pointer-linked graph representation for SVM. Each node carries its OWN BFS
// state and the device VA of its neighbor-VA list — the natural pointer form
// that SVM lets host and device share without flattening.
//
// Explicit padding keeps sizeof(Node) == 24 on BOTH the x86-64 host and the
// RV32 device (uint64_t is 8-byte aligned in both ABIs).
typedef struct Node {
  uint64_t neighbors_va;   // device VA of uint64_t[num_neighbors] neighbor Node VAs
  uint32_t num_neighbors;  // out-degree
  int32_t  cost;           // BFS distance (-1 = unvisited)
  uint8_t  visited;        // host-maintained visited flag
  uint8_t  in_next;        // set by kernel: node belongs to the next frontier
  uint16_t _pad0;
  uint32_t _pad1;
} Node;                    // 24 bytes

typedef struct {
  uint64_t frontier_va;    // device VA of uint64_t[frontier_size] of Node VAs
  uint32_t frontier_size;
  uint32_t _pad;
} kernel_arg_t;

#endif
