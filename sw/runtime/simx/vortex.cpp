// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <common.h>

#include <constants.h>
#include <mem.h>
#include <processor.h>
#include <util.h>
#ifdef VM_ENABLE
#include <vm.h>
#include <memory>
#endif

#include <assert.h>
#include <chrono>
#include <future>
#include <iostream>
#include <new>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>

#include <VX_config.h>

using namespace vortex;

class vx_device {
public:
  vx_device()
      : ram_(0, MEM_PAGE_SIZE), processor_(), global_mem_(ALLOC_BASE_ADDR, GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR, MEM_PAGE_SIZE, CACHE_BLOCK_SIZE) {
    // attach memory module
    processor_.attach_ram(&ram_);
  }

  ~vx_device() {
    if (future_.valid()) {
      future_.wait();
    }
  }

  // ------------------------------------------------------------------
  // SVM region bookkeeping (coarse-grained buffer SVM)
  // ------------------------------------------------------------------

  // One record per vx_svm_alloc call, keyed by the host buffer pointer.
  struct svm_region_t {
    uint64_t va;       // device-side virtual address (minted by VMManager)
    uint64_t pa;       // device-side physical address (in global_mem_)
    uint64_t size;     // user-requested size (not page-aligned)
    uint8_t* host_buf; // host-side copy buffer (malloc'd); equals key
  };

  std::unordered_map<void*, svm_region_t> svm_regions_;

  int init() {
#ifdef VM_ENABLE
    // Boot-time VM init: allocate the page table inside RAM, push SATP
    // into the simulator. Must run after attach_ram (constructor) and
    // before the first vx_mem_alloc (so phy_to_virt_map can mint VAs).
    vm_mgr_ = std::make_unique<VMManager>(&ram_);
    CHECK_ERR(vm_mgr_->init(), { return err; });
#endif
    return 0;
  }

  int get_caps(uint32_t caps_id, uint64_t *value) {
    uint64_t _value;
    switch (caps_id) {
    case VX_CAPS_VERSION:
      _value = IMPLEMENTATION_ID;
      break;
    case VX_CAPS_NUM_THREADS:
      _value = NUM_THREADS;
      break;
    case VX_CAPS_NUM_WARPS:
      _value = NUM_WARPS;
      break;
    case VX_CAPS_NUM_CORES:
      _value = NUM_CORES * NUM_CLUSTERS;
      break;
    case VX_CAPS_NUM_CLUSTERS:
      _value = NUM_CLUSTERS;
      break;
    case VX_CAPS_SOCKET_SIZE:
      _value = SOCKET_SIZE;
      break;
    case VX_CAPS_ISSUE_WIDTH:
      _value = ISSUE_WIDTH;
      break;
    case VX_CAPS_CACHE_LINE_SIZE:
      _value = CACHE_BLOCK_SIZE;
      break;
    case VX_CAPS_GLOBAL_MEM_SIZE:
      _value = GLOBAL_MEM_SIZE;
      break;
    case VX_CAPS_LOCAL_MEM_SIZE:
      _value = (1 << LMEM_LOG_SIZE);
      break;
    case VX_CAPS_ISA_FLAGS:
      _value = ((uint64_t(MISA_EXT)) << 32) | ((log2floor(XLEN) - 4) << 30) | MISA_STD;
      break;
    case VX_CAPS_NUM_MEM_BANKS:
      _value = PLATFORM_MEMORY_NUM_BANKS;
      break;
    case VX_CAPS_MEM_BANK_SIZE:
      _value = 1ull << (MEM_ADDR_WIDTH / PLATFORM_MEMORY_NUM_BANKS);
      break;
    case VX_CAPS_CLOCK_RATE:
      _value = 0;
      break;
    case VX_CAPS_PEAK_MEM_BW:
      _value = PLATFORM_MEMORY_PEAK_BW;
      break;
    default:
      std::cout << "invalid caps id: " << caps_id << std::endl;
      std::abort();
      return -1;
    }
    *value = _value;
    return 0;
  }

  int mem_alloc(uint64_t size, int flags, uint64_t *dev_addr) {
#ifdef VM_ENABLE
    uint64_t asize = aligned_size(size, MEM_PAGE_SIZE);
#else
    uint64_t asize = size;
#endif
    uint64_t addr = 0;

    DBGPRINT("[RT:mem_alloc] size: 0x%lx, asize, 0x%lx,flag : 0x%d\n", size, asize, flags);
    CHECK_ERR(global_mem_.allocate(asize, &addr), {
      return err;
    });
    CHECK_ERR(this->mem_access(addr, asize, flags), {
      global_mem_.release(addr);
      return err;
    });
    *dev_addr = addr;
#ifdef VM_ENABLE
    // Replace the PA in *dev_addr with a freshly-minted VA. After this
    // call, the user-facing API uses VAs end-to-end (mirrors source
    // runtime/simx/vortex.cpp:121-142).
    vm_mgr_->phy_to_virt_map(asize, dev_addr, flags);
#endif
    return 0;
  }

  int mem_reserve(uint64_t dev_addr, uint64_t size, int flags) {
#ifdef VM_ENABLE
    uint64_t asize = aligned_size(size, MEM_PAGE_SIZE);
#else
    uint64_t asize = size;
#endif
    CHECK_ERR(global_mem_.reserve(dev_addr, asize), {
      return err;
    });
    DBGPRINT("[RT:mem_reserve] addr: 0x%lx, asize:0x%lx, size: 0x%lx\n", dev_addr, asize, size);
    CHECK_ERR(this->mem_access(dev_addr, asize, flags), {
      global_mem_.release(dev_addr);
      return err;
    });
#ifdef VM_ENABLE
    // mem_reserve places content at the caller-chosen PA (vs mem_alloc,
    // which mints a fresh VA). The kernel will later access this region
    // via that same PA through the MMU, so install identity PTEs.
    CHECK_ERR(vm_mgr_->install_identity_map(dev_addr, asize), {
      global_mem_.release(dev_addr);
      return err;
    });
#endif
    return 0;
  }

  int mem_free(uint64_t dev_addr) {
#ifdef VM_ENABLE
    // dev_addr is a VA; resolve to PA before releasing from the
    // physical-address-keyed global allocator.
    uint64_t paddr = vm_mgr_->page_table_walk(dev_addr);
    return global_mem_.release(paddr);
#else
    return global_mem_.release(dev_addr);
#endif
  }

  int mem_access(uint64_t dev_addr, uint64_t size, int flags) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (dev_addr + asize > GLOBAL_MEM_SIZE)
      return -1;

    ram_.set_acl(dev_addr, size, flags);
    return 0;
  }

  int mem_info(uint64_t *mem_free, uint64_t *mem_used) const {
    if (mem_free)
      *mem_free = global_mem_.free();
    if (mem_used)
      *mem_used = global_mem_.allocated();
    return 0;
  }

  int copy(uint64_t dest_addr, uint64_t src_addr, uint64_t size) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (src_addr + asize > GLOBAL_MEM_SIZE || dest_addr + asize > GLOBAL_MEM_SIZE)
      return -1;
    ram_.enable_acl(false);
    ram_.copy(dest_addr, src_addr, size);
    ram_.enable_acl(true);
    return 0;
  }

  int upload(uint64_t dest_addr, const void *src, uint64_t size) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (dest_addr + asize > GLOBAL_MEM_SIZE)
      return -1;
#ifdef VM_ENABLE
    // dest_addr is a VA; translate before touching backing RAM.
    dest_addr = vm_mgr_->page_table_walk(dest_addr);
#endif
    ram_.enable_acl(false);
    ram_.write((const uint8_t *)src, dest_addr, size);
    ram_.enable_acl(true);

    /*
    DBGPRINT("upload %ld bytes to 0x%lx\n", size, dest_addr);
    for (uint64_t i = 0; i < size && i < 1024; i += 4) {
        DBGPRINT("  0x%lx <- 0x%x\n", dest_addr + i, *(uint32_t*)((uint8_t*)src + i));
    }*/

    return 0;
  }

  int download(void *dest, uint64_t src_addr, uint64_t size) {
    uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
    if (src_addr + asize > GLOBAL_MEM_SIZE)
      return -1;

    // flush GPU caches before reading back results
    {
      uint32_t dummy;
      for (uint32_t cid = 0; cid < NUM_CORES * NUM_CLUSTERS; ++cid) {
        this->dcr_read(VX_DCR_BASE_CACHE_FLUSH, cid, &dummy);
      }
    }
#ifdef VM_ENABLE
    // src_addr is a VA; translate before reading from backing RAM.
    src_addr = vm_mgr_->page_table_walk(src_addr);
#endif
    ram_.enable_acl(false);
    ram_.read((uint8_t *)dest, src_addr, size);
    ram_.enable_acl(true);

    /*DBGPRINT("download %ld bytes from 0x%lx\n", size, src_addr);
    for (uint64_t i = 0; i < size && i < 1024; i += 4) {
        DBGPRINT("  0x%lx -> 0x%x\n", src_addr + i, *(uint32_t*)((uint8_t*)dest + i));
    }*/

    return 0;
  }

  int start() {
    // DCRs already written by stub; just trigger execution
    future_ = std::async(std::launch::async, [&] { processor_.run(); });
    return 0;
  }

  int ready_wait(uint64_t timeout) {
    if (!future_.valid())
      return 0;
    uint64_t timeout_sec = timeout / 1000;
    std::chrono::seconds wait_time(1);
    for (;;) {
      // wait for 1 sec and check status
      auto status = future_.wait_for(wait_time);
      if (status == std::future_status::ready)
        break;
      if (0 == timeout_sec--)
        return -1;
    }
    return 0;
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    if (future_.valid()) {
      future_.wait(); // ensure prior run completed
    }
    return processor_.dcr_write(addr, value);
  }

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t *value) {
    if (future_.valid()) {
      future_.wait(); // ensure prior run completed
    }
    return processor_.dcr_read(addr, tag, value);
  }

  // ---- SVM methods (coarse-grained buffer SVM) ----

  int svm_alloc(uint64_t size, int flags, void** out_host_ptr) {
    uint64_t asize = aligned_size(size, MEM_PAGE_SIZE);
    uint64_t pa = 0;

    // Allocate device physical memory.
    CHECK_ERR(global_mem_.allocate(asize, &pa), { return err; });
    CHECK_ERR(this->mem_access(pa, asize, flags), {
      global_mem_.release(pa);
      return err;
    });

    // In VM mode, mint a device VA and install PTEs.
    // In non-VM mode, va == pa (identity mapping; no TLB involved).
    uint64_t va = pa;
#ifdef VM_ENABLE
    CHECK_ERR(vm_mgr_->phy_to_virt_map(asize, &va, flags), {
      global_mem_.release(pa);
      return err;
    });
#endif

    // Allocate the host-side copy buffer (used by svm_map / svm_unmap).
    uint8_t* host_buf = new (std::nothrow) uint8_t[size]();
    if (!host_buf) {
#ifdef VM_ENABLE
      vm_mgr_->free_va_mapping(va, asize);
#endif
      global_mem_.release(pa);
      return -1;
    }

    svm_regions_[host_buf] = svm_region_t{va, pa, size, host_buf};
    *out_host_ptr = host_buf;
    DBGPRINT("[SVM] svm_alloc: size=0x%lx va=0x%lx pa=0x%lx host_buf=%p\n",
             size, va, pa, (void*)host_buf);
    return 0;
  }

  int svm_free(void* host_ptr) {
    auto it = svm_regions_.find(host_ptr);
    if (it == svm_regions_.end()) {
      std::cerr << "[SVM] svm_free: unknown pointer " << host_ptr << std::endl;
      return -1;
    }
    auto& r = it->second;
    uint64_t asize = aligned_size(r.size, MEM_PAGE_SIZE);

#ifdef VM_ENABLE
    vm_mgr_->free_va_mapping(r.va, asize);
#endif
    this->mem_access(r.pa, asize, 0);
    global_mem_.release(r.pa);
    delete[] r.host_buf;
    svm_regions_.erase(it);
    DBGPRINT("[SVM] svm_free: host_buf=%p\n", host_ptr);
    return 0;
  }

  // Transfer ownership to host. For READ access, flush caches and copy
  // device RAM → host buffer. For WRITE access, no copy (host will overwrite).
  int svm_map(void* host_ptr, uint64_t size, int flags) {
    auto it = svm_regions_.find(host_ptr);
    if (it == svm_regions_.end()) {
      std::cerr << "[SVM] svm_map: unknown pointer " << host_ptr << std::endl;
      return -1;
    }
    auto& r = it->second;
    if (size > r.size) size = r.size;

    if (flags & VX_MEM_READ) {
      // Flush all device caches so we read back the most recent data.
      {
        uint32_t dummy;
        for (uint32_t cid = 0; cid < NUM_CORES * NUM_CLUSTERS; ++cid) {
          this->dcr_read(VX_DCR_BASE_CACHE_FLUSH, cid, &dummy);
        }
      }
      // Copy device RAM at PA into host buffer.
      ram_.enable_acl(false);
      ram_.read(r.host_buf, r.pa, size);
      ram_.enable_acl(true);
    }
    // For WRITE-only, the host will overwrite; no copy needed.
    DBGPRINT("[SVM] svm_map: host_buf=%p size=0x%lx flags=0x%x\n",
             host_ptr, size, flags);
    return 0;
  }

  // Transfer ownership to device: copy host buffer → device RAM at PA.
  int svm_unmap(void* host_ptr, uint64_t size) {
    auto it = svm_regions_.find(host_ptr);
    if (it == svm_regions_.end()) {
      std::cerr << "[SVM] svm_unmap: unknown pointer " << host_ptr << std::endl;
      return -1;
    }
    auto& r = it->second;
    if (size > r.size) size = r.size;

    ram_.enable_acl(false);
    ram_.write(r.host_buf, r.pa, size);
    ram_.enable_acl(true);
    DBGPRINT("[SVM] svm_unmap: host_buf=%p pa=0x%lx size=0x%lx\n",
             host_ptr, r.pa, size);
    return 0;
  }

  uint64_t svm_dev_addr(void* host_ptr) {
    auto it = svm_regions_.find(host_ptr);
    if (it == svm_regions_.end()) {
      std::cerr << "[SVM] svm_dev_addr: unknown pointer " << host_ptr << std::endl;
      return 0;
    }
    return it->second.va;
  }

private:
  RAM ram_;
  Processor processor_;
  MemoryAllocator global_mem_;
  std::future<void> future_;
#ifdef VM_ENABLE
  std::unique_ptr<VMManager> vm_mgr_;
#endif
};

#include <callbacks.inc>

// SVM C API entry points (not dispatched through callbacks — called directly).
extern "C" {

int vx_svm_alloc(vx_device_h hdevice, uint64_t size, int flags, void** host_ptr) {
  if (!hdevice || !host_ptr || size == 0) return -1;
  DBGPRINT("VX_SVM_ALLOC: hdevice=%p size=%lu flags=0x%x\n", hdevice, size, flags);
  return ((vx_device*)hdevice)->svm_alloc(size, flags, host_ptr);
}

int vx_svm_free(vx_device_h hdevice, void* host_ptr) {
  if (!hdevice || !host_ptr) return -1;
  DBGPRINT("VX_SVM_FREE: hdevice=%p host_ptr=%p\n", hdevice, host_ptr);
  return ((vx_device*)hdevice)->svm_free(host_ptr);
}

int vx_svm_map(vx_device_h hdevice, void* host_ptr, uint64_t size, int flags) {
  if (!hdevice || !host_ptr || size == 0) return -1;
  DBGPRINT("VX_SVM_MAP: hdevice=%p host_ptr=%p size=%lu flags=0x%x\n",
           hdevice, host_ptr, size, flags);
  return ((vx_device*)hdevice)->svm_map(host_ptr, size, flags);
}

int vx_svm_unmap(vx_device_h hdevice, void* host_ptr, uint64_t size) {
  if (!hdevice || !host_ptr || size == 0) return -1;
  DBGPRINT("VX_SVM_UNMAP: hdevice=%p host_ptr=%p size=%lu\n",
           hdevice, host_ptr, size);
  return ((vx_device*)hdevice)->svm_unmap(host_ptr, size);
}

uint64_t vx_svm_dev_addr(vx_device_h hdevice, void* host_ptr) {
  if (!hdevice || !host_ptr) return 0;
  return ((vx_device*)hdevice)->svm_dev_addr(host_ptr);
}

} // extern "C"
