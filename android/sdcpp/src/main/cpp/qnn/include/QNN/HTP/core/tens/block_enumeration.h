// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef HEXNN_BLOCK_ENUMERATION_H
#define HEXNN_BLOCK_ENUMERATION_H 1
#include "allocator.h"
#include "macros_attribute.h"
#include "weak_linkage.h"
#include "minihash.h"

#include <cstddef>

PUSH_VISIBILITY(default)

class Tensor;

namespace hnnx {
//
// This type represent a set of block_id, across a tensor or group of
// tensors.
using blockid_set_t = miniset<void *>;

//
// This is an interface class; a reference to this
// is passed to tensor->enum_memory_blocks; the tensor then calls the 'supply_blocks_func' method
// (maybe using one of the handy wrappers) to generate one or more 'void*' which are the block
// ids.
//   Rules are:
//   - if 'supply_blocks_func' is called with memclass < 0, the memory class of the block is unspecified;
//     if it is called with memclass >=0, the value  is  MemoryClass and the tensor guarantees that all
//     of the blocks are in that class.
//   - Tensor may in general make multiple calls to supply_blocks_func in one call to enum_memory_blocks, and may
//     supply different values of mclass parameter. But currently it is only one.
//   - The tensor does *not* guarantee that the same id is not presented multiple times in one call
//     to enum_memory_blocks.
//
class MemBlockEnumerator {
  public:
    API_EXPORT virtual ~MemBlockEnumerator() {}
    inline MemBlockEnumerator() {}
    MemBlockEnumerator(MemBlockEnumerator const &) = delete;
    MemBlockEnumerator(MemBlockEnumerator &&) = delete;
    MemBlockEnumerator &operator=(MemBlockEnumerator const &) = delete;
    MemBlockEnumerator &operator=(MemBlockEnumerator &&) = delete;

    API_EXPORT virtual void supply_blocks_func(Tensor const *tensp, int memclass, void *const *ptr, size_t num,
                                               size_t blocksize) = 0;
    // Tensors can use these wrappers
    API_EXPORT inline void supply_blocks(Tensor const *tensp, void *const *ptr, size_t num, size_t blocksize)
    {
        supply_blocks_func(tensp, -1, ptr, num, blocksize);
    }
    API_EXPORT inline void supply_blocks(Tensor const *tensp, MemoryClass mc, void *const *ptr, size_t num,
                                         size_t blocksize)
    {
        supply_blocks_func(tensp, int(mc), ptr, num, blocksize);
    }
};
// utility class, to enumerate to a std::set
// if mclass_sel >=0, we skip tensors which have a different memory class.
class MemBlockEnumToSet : public MemBlockEnumerator {
    blockid_set_t &m_set;
    int m_memclass_sel;

  public:
    API_EXPORT explicit MemBlockEnumToSet(blockid_set_t &s, int mclass_sel = -1) : m_set(s), m_memclass_sel(mclass_sel)
    {
    }
    API_EXPORT MemBlockEnumToSet(blockid_set_t &s, MemoryClass mc) : m_set(s), m_memclass_sel(int(mc)) {}
    API_EXPORT virtual void supply_blocks_func(Tensor const *, int memclass, void *const *ptr, size_t num,
                                               size_t /* blocksize */) override
    {
        if (m_memclass_sel >= 0 && memclass >= 0 && m_memclass_sel != memclass) return;
        for (size_t i = 0; i < num; i++) {
            if (ptr[i] != Allocator::vacant()) m_set.emplace(ptr[i]);
        }
    }
};
// This is to support Tensor::enum_memory_blocks_withfunc( ..callable..)
//  and similar for Op methods
template <typename ENFUNC> class MemBlockEnumWrapper : public MemBlockEnumerator {
    ENFUNC m_enfunc;

    API_EXPORT virtual void supply_blocks_func(Tensor const *tensp, int memclass, void *const *ptr, size_t num,
                                               size_t blocksize) override
    {
        m_enfunc(tensp, memclass, ptr, num, blocksize);
    }

  public:
    API_EXPORT inline MemBlockEnumWrapper(ENFUNC &&ef) : m_enfunc(std::move(ef)) {}
    API_EXPORT inline MemBlockEnumWrapper(ENFUNC const &ef) : m_enfunc(ef) {}
};

// This is to support Tensor::replace_memory_blocks_withfunc( ..callable..)
//  and similar for Op methods
//  The 'replfunc' is called as: void* replfunc( Tensor const *tp, void *old_blkid)
//  for each block in the tensor; the returned value is used as the replacement blkid.
template <typename REPLFUNC> class MemBlockReplBlockWrapper : public MemBlockEnumerator {
    REPLFUNC m_replfunc;

    API_EXPORT virtual void supply_blocks_func(Tensor const *tensp, int /* memclass */, void *const *ptr, size_t num,
                                               size_t /* blocksize */) override
    {
        for (unsigned i = 0; i < num; i++) {
            void *newblk = m_replfunc(tensp, ptr[i]);
            const_cast<void *&>(ptr[i]) = newblk;
        }
    }

  public:
    API_EXPORT inline MemBlockReplBlockWrapper(REPLFUNC &&ef) : m_replfunc(std::move(ef)) {}
    API_EXPORT inline MemBlockReplBlockWrapper(REPLFUNC const &ef) : m_replfunc(ef) {}
};

} // namespace hnnx

POP_VISIBILITY()
#endif // HEXNN_BLOCK_ENUMERATION_H
