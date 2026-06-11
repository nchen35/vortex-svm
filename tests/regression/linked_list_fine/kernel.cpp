// SVM linked-list traversal kernel (fine-grained variant).
// Identical to the coarse-grained kernel: the kernel side is unchanged between
// coarse and fine SVM — only the host-side coherence handling differs.
// One work-item traverses the entire list and writes the sum of node values
// to the result scalar.  Only thread 0 does real work; all others exit early.

#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx != 0)
    return;

  int32_t sum = 0;
  uint64_t cur = arg->head_va;

  // Walk the singly-linked list.  Each node->next is a device VA that the
  // MMU translates on every load — no pointer fixup needed.
  while (cur != 0) {
    const Node* node = reinterpret_cast<const Node*>((uintptr_t)cur);
    sum += node->value;
    cur  = node->next;
  }

  int32_t* result = reinterpret_cast<int32_t*>((uintptr_t)arg->result_va);
  *result = sum;
}
