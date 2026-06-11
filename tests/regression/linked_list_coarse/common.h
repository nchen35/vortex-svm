#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// A singly-linked list node. `next` stores a device virtual address (VA)
// so the kernel can dereference it without any pointer fixup — the MMU
// translates it transparently.  `_pad` keeps the struct 16 bytes on both
// 32-bit and 64-bit XLEN builds.
typedef struct Node {
  uint64_t next;   // device VA of the next node, or 0 for list end
  int32_t  value;
  int32_t  _pad;
} Node;

typedef struct {
  uint64_t head_va;    // device VA of the first node in the list
  uint64_t result_va;  // device VA of int32_t output scalar (sum of values)
  uint32_t n_nodes;    // number of nodes in the list (for sanity check only)
  uint32_t _pad;
} kernel_arg_t;

#endif
