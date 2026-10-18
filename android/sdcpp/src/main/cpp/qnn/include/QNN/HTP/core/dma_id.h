// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#ifndef HEXNN_DMA_ID_H_
#define HEXNN_DMA_ID_H_ 1

namespace hnnx {
class DMA_Manager;

// opaque type for identifying dma operations.
// Ws previously defined as 'dma_id' inside class hnnx::DMA_Manager
// (and can still be referred to by hnnx::DMA_Manager::dma_id)
class dma_id_t final {
    friend class hnnx::DMA_Manager;

  protected:
    using seqno_t = unsigned;
    seqno_t seq;
    explicit dma_id_t(seqno_t s) : seq(s) {}

  public:
    dma_id_t() : seq(0) {}
    dma_id_t(dma_id_t const &r) = default;
    dma_id_t &operator=(dma_id_t const &r) = default;
    dma_id_t(dma_id_t &&) = default;
    dma_id_t &operator=(dma_id_t &&) = default;
    ~dma_id_t() = default;
    bool operator==(dma_id_t const &r) const { return seq == r.seq; }
    bool operator!=(dma_id_t const &r) const { return seq != r.seq; }
};
} // namespace hnnx
#endif // HEXNN_DMA_ID_H_
