// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef HEXNN_TENSOR_CONCRETE_H
#define HEXNN_TENSOR_CONCRETE_H 1
#include "macros_attribute.h"
#include "weak_linkage.h"
#include "shape.h"
#include "crate.h"
#include "padding.h"
#include "memory_layout.h"
#include "tensor_base.h"

#include <array>
#include <cstdint>
#include <cstddef>
#include <numeric>
#include <type_traits>

PUSH_VISIBILITY(default)

namespace hnnx {
class Allocator;

API_EXPORT extern uint64_t checksum_bytes(uint64_t prev, uint8_t const *bytes, unsigned n);

template <size_t N> //
size_t product_of_array(std::array<size_t, N> const &arr)
{
    return std::accumulate(arr.cbegin(), arr.cend(), size_t(1), std::multiplies<size_t>());
}
} // namespace hnnx

///////////////////////////////////////////
// this is contained within LayoutTensor, and implements the block pointer,
// or pointers.
// The first parameter indicates if the layout is indirect; we specialize
// the whole class on true vs. false.
// The remaining template parms are the same as those in the LayoutTensot containing
// it.

/// >> for contiguous tensors

template <typename STYPE, typename TLayout, typename Pad_t> struct layout_mem_contig {
    static constexpr unsigned Rank = TLayout::Rank;
    using Shape_t = Shape<Rank>;
    using storage_type = STYPE;
    static constexpr TLayout layout{};
    static constexpr Pad_t pad{};
    static constexpr bool is_singular = std::is_same_v<TLayout, SingularMemoryLayout<Rank>>;

    storage_type *bulk_data;

    API_EXPORT layout_mem_contig(Shape_t const * /* shp */, Graph & /* graph_in */) : bulk_data() {}

    // duplicate clone from another
    API_EXPORT layout_mem_contig(Shape_t const * /* shp */, layout_mem_contig const &other,
                                 hnnx::Allocator * /* alloc */, Tensor::clone_mode /* cmode */)
        : bulk_data(other.bulk_data)
    {
    }

    // construct from deserialize
    API_EXPORT layout_mem_contig(Shape_t const *, hnnx::Deserz &dctx)
        : bulk_data((storage_type *)Tensor::deserialize_block_pointer(dctx))
    {
    }

    // this implements raw_data in the containing tensor
    API_EXPORT ALWAYSINLINE void *raw_data() const noexcept { return (void *)bulk_data; }

    // this implements set_raw_data_despite_danger(void *buffer) override final { bulk_data = static_cast<T *>(buffer); }
    API_EXPORT ALWAYSINLINE void set_raw_data_despite_danger(void *buffer)
    {
        bulk_data = static_cast<storage_type *>(buffer);
    }

    // this implements element_addr in the containing tensor.
    API_EXPORT ALWAYSINLINE void *element_addr(Shape_t const *shp, size_t /* rank */,
                                               SIdx const coords_in[]) const noexcept
    {
        //assert(rank == Rank);
        static_assert(!is_singular);
        const std::array<size_t, Rank> padded_coords =
                pad.pad_coords(hnnx::ptr_to_stdarray<Rank, SIdx>(&coords_in[0]), shp->pad);
        size_t const offset = layout.linear_offset(padded_coords, shp->max_dims);
        return (void *)&bulk_data[offset];
    }

    // element_addr impl that takes into account dynamic valid_dims when calculating
    // the offset into the flat memory buffer
    ALWAYSINLINE void *element_addr(Shape_t const *shp, size_t /* rank */, SIdx const coords_in[],
                                    std::array<size_t, Rank> const &valid_dims) const noexcept
    {
        static_assert(!is_singular);
        const std::array<size_t, Rank> padded_coords =
                pad.pad_coords(hnnx::ptr_to_stdarray<Rank, SIdx>(&coords_in[0]), shp->pad);
        size_t const offset = layout.linear_offset(padded_coords, valid_dims);
        return (void *)&bulk_data[offset];
    }

    API_EXPORT inline ALWAYSINLINE size_t get_chunk_offset(Shape_t const *shp, size_t /* rank */,
                                                           SIdx const coords_in[]) const noexcept
    {
        static_assert(!is_singular);
        const std::array<size_t, Rank> padded_coords =
                pad.pad_coords(hnnx::ptr_to_stdarray<Rank, SIdx>(&coords_in[0]), shp->pad);
        return layout.linear_offset(padded_coords, shp->max_dims);
    }

    PUSH_WARNING()
    DISABLE_WARNING("-Wcast-qual", MSVC_NO_EQUIV)
    // get pointer to block table, and length
    API_EXPORT ALWAYSINLINE void **get_block_list_ptr() const { return (void **)&bulk_data; }
    POP_WARNING()
    API_EXPORT static ALWAYSINLINE size_t get_block_list_len(Shape_t const * /* shp */) { return 1; }
    // block size for allocation
    API_EXPORT ALWAYSINLINE static size_t get_elements_per_block(Shape_t const *shp)
    {
        if (is_singular) return 1;
        return hnnx::product_of_array(shp->max_dims);
    }
    // find the address of the block pointer containing the specified coords.
    // (not used for contig. tensor, but this is reasonable impl).
    API_EXPORT storage_type **block_ptr_addr(Shape_t const * /* shape */, std::array<SIdx, Rank> /* coords */) const
    {
        return &bulk_data;
    }
    // dummy for this
    API_EXPORT void realloc_blocktab(hnnx::Allocator * /* alloc */, Shape_t const * /* old_shape */,
                                     Shape_t const * /* new_shape */)
    {
        bulk_data = nullptr;
    }

    // compare memory (raw compare)
    API_EXPORT int compare_memory(Shape_t const *shp, layout_mem_contig const &rhs) const
    {
        size_t const len = get_elements_per_block(shp) * sizeof(storage_type);
        return memcmp(bulk_data, rhs.bulk_data, len);
    }
    // find content hash of memory.
    //
    API_EXPORT uint32_t find_content_hash(Shape_t const *shp, uint32_t oldhash, bool is_float) const
    {
        size_t const len = get_elements_per_block(shp) * sizeof(storage_type);
        return mulu32_modular(oldhash, 0x223131) ^ Tensor::content_hash_data(bulk_data, len, is_float);
    }
};

/// >> for indirect tensors
namespace indirect_layout_mem {
API_EXPORT inline void **make_blocktab(size_t n_blocks, Graph &graph_in)
{
    return hnnx::graph_crate(graph_in)->alloc_array_zero<void *>(n_blocks);
}

template <typename CRATE> // Crate or DCrate
void **make_blocktab_for_overwrite(const size_t n_blocks, CRATE *const crate_p)
{
    return crate_p->template alloc_array<void *>(n_blocks);
}

// TODO: make this not inline.
API_EXPORT inline int compare_indirect_blocks(void **ptr_a, void **ptr_b, size_t nblocks, size_t blocklen)
{
    for (size_t i = 0; i < nblocks; i++) {
        int const cmp = memcmp(ptr_a[i], ptr_b[i], blocklen);
        if (cmp != 0) return cmp;
    }
    return 0;
}
} // namespace indirect_layout_mem

//  layout_mem for indirect.
template <typename STYPE, typename TLayout, typename Pad_t> struct layout_mem_indirect {
    static constexpr unsigned Rank = TLayout::Rank;
    using Shape_t = Shape<Rank>;
    using storage_type = STYPE;
    static constexpr TLayout layout{};
    static constexpr Pad_t pad{};

    storage_type **blocktab;

    // construct table
    API_EXPORT layout_mem_indirect(Shape_t const *shp, Graph &graph_in)
        : blocktab((storage_type **)indirect_layout_mem::make_blocktab(layout.num_blocks(shp->max_dims), graph_in))
    {
    }
    // duplicate clone from another
    API_EXPORT layout_mem_indirect(Shape_t const *shp, layout_mem_indirect const &other, hnnx::Allocator *alloc,
                                   Tensor::clone_mode /* cmode */)
        : blocktab()
    {
        unsigned const nblocks = layout.num_blocks(shp->max_dims);
        hnnx::Crate *crate_p = hnnx::graph_crate(alloc->graph);
        blocktab = (storage_type **)indirect_layout_mem::make_blocktab_for_overwrite(nblocks, crate_p);
        std::memcpy(blocktab, other.blocktab, sizeof(void *) * nblocks);
    }
    // construct from deserialize
    API_EXPORT layout_mem_indirect(Shape_t const *shp, hnnx::Deserz &dctx) : blocktab()
    {
        // if we are not 'classic' format, we may not be able to access shape object here due to delayed
        // pointer resolution. But we don't need nblocks unless classic format. 1 is the 'don't know'
        // value.
        unsigned const nblocks = dctx.classic_format() ? layout.num_blocks(shp->max_dims) : 1;
        Tensor::deserialize_blocktable(dctx, blocktab, nblocks);
    }

    // this implements raw_data in the containing tensor
    API_EXPORT ALWAYSINLINE void *raw_data() const noexcept { return (void *)blocktab[0]; }
    // this implements set_raw_data_despite_danger(void *buffer) override final { bulk_data = static_cast<T *>(buffer); }
    API_EXPORT void set_raw_data_despite_danger(void * /* buffer */)
    {
        assert(false && "Invalid to set raw pointer on this type of tensor");
    }

    // this implements element_addr in the containing tensor.
    API_EXPORT ALWAYSINLINE void *element_addr(Shape_t const *shp, size_t rank, SIdx const coords_in[]) const noexcept
    {
        assert(rank == Rank);
        std::array<size_t, Rank> const padded_coords =
                pad.pad_coords(hnnx::ptr_to_stdarray<Rank, SIdx>(&coords_in[0]), shp->pad);
        size_t const block_offset = layout.chunk_offset(padded_coords, shp->max_dims);
        size_t const block_idx = layout.chunk_index(padded_coords, shp->max_dims);
        return (void *)&blocktab[block_idx][block_offset];
    }

    // element_addr impl for dynamic valid_dims code path. Block offset/index calculation
    // is identical to non dynamic variant for chunked memory layouts.
    ALWAYSINLINE void *element_addr(Shape_t const *shp, size_t rank, SIdx const coords_in[],
                                    std::array<size_t, Rank> const & /* valid_dims */) const noexcept
    {
        assert(rank == Rank);
        std::array<size_t, Rank> const padded_coords =
                pad.pad_coords(hnnx::ptr_to_stdarray<Rank, SIdx>(&coords_in[0]), shp->pad);
        size_t const block_offset = layout.chunk_offset(padded_coords, shp->max_dims);
        size_t const block_idx = layout.chunk_index(padded_coords, shp->max_dims);
        return (void *)&blocktab[block_idx][block_offset];
    }

    API_EXPORT inline ALWAYSINLINE size_t get_chunk_offset(Shape_t const *shp, size_t rank,
                                                           SIdx const coords_in[]) const noexcept
    {
        assert(rank == Rank);
        std::array<size_t, Rank> const padded_coords =
                pad.pad_coords(hnnx::ptr_to_stdarray<Rank, SIdx>(&coords_in[0]), shp->pad);
        return layout.chunk_offset(padded_coords, shp->max_dims);
    }

    // get pointer to block table, and length
    API_EXPORT ALWAYSINLINE void **get_block_list_ptr() const { return (void **)blocktab; }
    API_EXPORT static ALWAYSINLINE size_t get_block_list_len(Shape_t const *shp)
    {
        return layout.num_blocks(shp->max_dims);
    }
    // block size for allocation
    API_EXPORT ALWAYSINLINE static size_t get_elements_per_block(Shape_t const *shp)
    {
        return layout.block_total(shp->max_dims);
    }

    // find the address of the block pointer containing the specified coords.
    API_EXPORT storage_type **block_ptr_addr(Shape_t const *shape, std::array<SIdx, Rank> coords) const
    {
        std::array<size_t, Rank> const padded_coords = pad.pad_coords(coords, shape->pad);
        size_t const block_idx = layout.chunk_index(padded_coords, shape->max_dims);
        return &blocktab[block_idx];
    }
    // reallocate for change from old_shape to new_shape (typically just the padding
    // is changed) and zero the blocktab. If the shape is not actually changed, or if
    // the blocktab isn't larger than before, we keep the old one, but we still clear it.
    API_EXPORT void realloc_blocktab(hnnx::Allocator *alloc, Shape_t const *old_shape, Shape_t const *new_shape)
    {
        unsigned const nblocks = layout.num_blocks(new_shape->max_dims);
        if (old_shape != new_shape) {
            unsigned const old_nblocks = layout.num_blocks(old_shape->max_dims);
            if (nblocks > old_nblocks) { // need reallocate.
                blocktab = (storage_type **)indirect_layout_mem::make_blocktab(nblocks, alloc->graph);
                return; // already zeroed
            }
        }
        ::memset(blocktab, 0, nblocks * sizeof(void *));
    }

    // compare memory (raw compare)
    API_EXPORT int compare_memory(Shape_t const *shp, layout_mem_indirect const &rhs) const
    {
        size_t const nblocks = layout.num_blocks(shp->max_dims);
        size_t const blocklen = sizeof(storage_type) * layout.block_total(shp->max_dims);
        return indirect_layout_mem::compare_indirect_blocks((void **)blocktab, (void **)rhs.blocktab, nblocks,
                                                            blocklen);
    }
    // find content hash of memory.
    //
    API_EXPORT uint32_t find_content_hash(Shape_t const *shp, uint32_t oldhash, bool is_float) const
    {
        size_t const nblocks = layout.num_blocks(shp->max_dims);
        size_t const blocklen = sizeof(storage_type) * layout.block_total(shp->max_dims);
        return Tensor::content_hash_data_indirect(oldhash, (void **)blocktab, nblocks, blocklen, is_float);
    }
};
///////////////////////////////////////////
template <typename Linfo> class LayoutTensor;
template <typename Linfo> class BlockTableAccessor {
  protected:
    static constexpr unsigned Rank = Linfo::Rank;
    using storage_type = typename Linfo::storage_type;
    using pointer_type = storage_type *;
    using TLayout = typename Linfo::Tlayout;
    using Pad_t = typename Linfo::Pad_t;
    static_assert(Linfo::is_indirect && Linfo::is_chunked);
    pointer_type *blktab; // the base of the block table
    std::array<size_t, Rank> blkdims; // dims of the block table in blocks
    std::array<size_t, Rank> blkstrides; // 'strides' (note stride for dim i is blkstrides[i+1];
    // stride for dim RANK-1  is 1; blkstrides[0] is the whole size.
    std::array<unsigned, Rank> margin; // margin offset
    // support for 'variable_block':
    static constexpr unsigned n_blockshape_dims = tensor_traits<LayoutTensor<Linfo>>::is_variable_block
                                                          ? Rank - tensor_traits<LayoutTensor<Linfo>>::indirect_ranks
                                                          : 0;
    std::array<size_t, n_blockshape_dims> blockshape_dims;

  public:
    API_EXPORT explicit BlockTableAccessor(LayoutTensor<Linfo> const &tens) : blktab(tens.blocktab_ptr())
    {
        Shape<Rank> const &shp = *tens.shape;
        size_t allprod = 1;
        for (int i = Rank - 1; i >= 0; --i) {
            unsigned const blk = TLayout::ChunkSizes[i];
            size_t const blkdim = shp.max_dims[i] / blk;
            allprod *= blkdim;
            blkdims[i] = blkdim;
            margin[i] = shp.pad[i];
            blkstrides[i] = allprod;
        }
        // if any dims affect the block size, copy those out.
        std::copy_n(shp.max_dims.begin() + (Rank - n_blockshape_dims), n_blockshape_dims, blockshape_dims.begin());
    }
    // methods which have the same name as tensor methods, do
    // the same thing here.

    API_EXPORT static constexpr unsigned rank() { return Rank; }

    API_EXPORT size_t blocktab_len() const { return blkstrides[0]; }
    API_EXPORT pointer_type *blocktab_ptr() const { return blktab; }
    API_EXPORT size_t blocktab_blocksize() const
    {
        std::array<size_t, Rank> dummy_shape{};
        // the shape will be ignored (unless is_variable_block).
        std::copy_n(blockshape_dims.begin(), n_blockshape_dims, dummy_shape.begin() + (Rank - blockshape_dims));
        return TLayout::block_total(dummy_shape);
    }
    API_EXPORT size_t blocktab_blocksize_bytes() const { return blocktab_blocksize() * sizeof(storage_type); }

    API_EXPORT size_t blocktab_dim(int i) const { return blkdims[i]; }
    API_EXPORT size_t blocktab_dim_stride(int i) const { return (i < Rank - 1) ? blkstrides[i + 1] : 1; }

    // block_ptr_address(b,h,w,d) and block_ptr accept element coordinates.

    template <typename... ind_types> API_EXPORT pointer_type *block_ptr_address(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return block_ptr_calc(coords);
    }
    template <typename... ind_types> API_EXPORT pointer_type &block_ptr(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *block_ptr_calc(coords);
    }
    // blktab(b,h,w,d) accepts *block* coords
    //
    template <typename... ind_types> API_EXPORT pointer_type &blocktab(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *blktab_ptr_calc(coords);
    }
    // same_table_shape: the shape of the table is the same as the 'other'.
    API_EXPORT bool same_table_shape(BlockTableAccessor const &other) const
    {
        for (int i = 0; i < Rank; i++)
            if (blkdims[i] != other.blkdims[i]) return false;
        return true;
    }
    // 'same_layout' means the same table shape and the same padding offset. Dims may not be identical.
    API_EXPORT bool same_layout(BlockTableAccessor const &other) const
    {
        if (!same_table_shape(other)) return false;
        for (int i = 0; i < Rank; i++)
            if (margin[i] != other.margin[i]) return false;
        return true;
    }

  protected:
    API_EXPORT pointer_type *block_ptr_calc(std::array<SIdx, Rank> const &coords) const
    {
        size_t sum = 0;
        for (int i = 0; i < Rank; i++) {
            unsigned blk = TLayout::ChunkSizes[i];
            unsigned idx = (coords[i] + margin[i] + (blk - 1)) / blk;
            sum += idx * ((i < Rank - 1) ? blkstrides[i + 1] : 1);
        }
        return blktab + sum;
    }
    API_EXPORT pointer_type *blktab_ptr_calc(std::array<SIdx, Rank> const &coords) const
    {
        size_t sum = coords[Rank - 1];
        for (int i = 0; i < Rank - 1; i++) {
            sum += coords[i] * blkstrides[i + 1];
        }
        return blktab + sum;
    }
};

//
// Constructors of LayoutTensor (all protected; only used by subclass ctor):
// LayoutTensor(const Op * producer_in, const OutputDef &def, Graph &graph_in, <<func pointer>>)
//    - build for given shape, attached to given producer.
//  LayoutTensor(const Op *producer_in, hnnx::Deserz & dctx, <<funct pointer>>)
//    - deserialize. Notr that dctx contains a graph ref.
//  LayoutTensor(const ConcreteTensor &old, hnnx::Allocator *allocator,Tensor::clone_mode cmode)
//    - 'clone duplicate' of the given tensor. Note that cmode is ignored.
//
// The function pointers in the first two cases are used to construct the correct
// interface object, according to the Interface_t of the subclass.
//

template <typename Linfo> class LayoutTensor : public RankedTensor<Linfo::Rank> {
  protected:
    using BaseRT = RankedTensor<Linfo::Rank>;
    API_EXPORT static constexpr unsigned Rank = Linfo::Rank;
    using storage_type = typename Linfo::storage_type;
    using TLayout = typename Linfo::Tlayout;
    using Pad_t = typename Linfo::Pad_t;
    API_EXPORT static constexpr bool is_chunked = Linfo::is_chunked;
    static_assert(is_chunked == (TLayout::chunk_total > 1));
    API_EXPORT static constexpr bool is_indirect = Linfo::is_indirect;
    API_EXPORT static constexpr bool is_padded = !std::is_same<Pad_t, NoPadding<Rank>>::value;

    static_assert(!(is_indirect && !is_chunked), "non-chunked layouts can't be indirect");

    Interface const *const interface_ptr; // pointer to shared instance of Interface subclass.
    using Shape_t = Shape<Rank>;

  public:
    Shape_t const *shape;
    API_EXPORT static constexpr TLayout layout{};
    API_EXPORT static constexpr Pad_t pad{};
#ifndef PREPARE_DISABLED
    std::string get_shape_info() const override { return shape->get_shape_info(); }
#endif

  protected: // interface, then shape, then mem
    using layout_mem_t = std::conditional_t<is_indirect, layout_mem_indirect<storage_type, TLayout, Pad_t>,
                                            layout_mem_contig<storage_type, TLayout, Pad_t>>;
    layout_mem_t mem;

  public:
    struct API_EXPORT traits {
        using storage_type = LayoutTensor::storage_type;
        using raw_type = LayoutTensor::storage_type; // result from get_raw()
        static constexpr unsigned rank = Rank;
        static constexpr bool is_indirect = LayoutTensor::is_indirect;
        static constexpr bool is_chunked = LayoutTensor::is_chunked;
        static constexpr bool is_singular = std::is_same<TLayout, SingularMemoryLayout<Rank>>::value;
        static constexpr bool has_padding = !std::is_same<Pad_t, NoPadding<Rank>>::value;
        static constexpr unsigned indirect_ranks = TLayout::indirect_ranks;
        static constexpr bool is_variable_block = is_indirect && (indirect_ranks < Rank);
        using pad_type = Pad_t;
        using layout_type = TLayout;
        using layouttensor_type = LayoutTensor;
    };

  protected:
    static constexpr bool is_singular = traits::is_singular;
    // this function is used to construct the 'shape' pointer in prepare mode,
    // and is different for 'singular' case
    ALWAYSINLINE static Shape<Rank> const *init_shape_p(Graph &graph_in, const OutputDef &def)
    {
        Shape_t const shp(hnnx::ptr_to_stdarray<Rank, size_t>(&def.max_sizes[0]),
                          TLayout::pad(hnnx::ptr_to_stdarray<Rank, size_t>(&def.max_sizes[0])));
        if constexpr (is_singular) {
            if (std::find_if(shp.dims.begin(), shp.dims.end(), [](size_t d) { return d != 1; }) != shp.dims.end()) {
                throw std::runtime_error("singular tensor with shape not 1's");
            }
        }
        return Shape_t::canonical_shape(graph_in, shp);
    }
    // only used in the deserialize ctor
    Interface const *&interface_ptr_ref() { return const_cast<Interface const *&>(interface_ptr); }
    // ctors are marked noinline; otherwise they just get inlined
    // into all the ConcreteTensor ctors, which isn't really helpful.
    [[gnu::noinline]] API_EXPORT LayoutTensor(const Op *producer_in, const OutputDef &def, Graph &graph_in,
                                              Interface const *(*ifc_maker)(Graph &, OutputDef const &))
        : BaseRT(producer_in), interface_ptr((*ifc_maker)(graph_in, def)), //
          shape(init_shape_p(graph_in, def)), //
          mem(shape, graph_in)
    {
    }
    using interface_deser_func = Interface const *(*)(hnnx::Deserz &, Interface const **);
    [[gnu::noinline]] API_EXPORT LayoutTensor(hnnx::Deserz &dctx, interface_deser_func const ifc_deser_fp)
        : BaseRT(dctx), interface_ptr((*ifc_deser_fp)(dctx, &interface_ptr_ref())),
          shape(Shape_t::deserialize(dctx, &shape)), mem(shape, dctx)
    {
    }
    // clone ctor.
    [[gnu::noinline]] API_EXPORT LayoutTensor(const LayoutTensor &old, hnnx::Allocator *allocator,
                                              Interface const *ifc_in, Tensor::clone_mode cmode)
        : BaseRT(old, allocator, cmode), interface_ptr(ifc_in), shape(old.shape), mem(shape, old.mem, allocator, cmode)
    {
    }

  public:
    API_EXPORT virtual size_t dim(size_t index) const noexcept override final { return shape->dims[index]; }
    API_EXPORT const std::array<size_t, Rank> &dims() const { return shape->dims; }

    template <typename... T> API_EXPORT const std::array<size_t, sizeof...(T)> dims(T... indices) const
    {
        return Tensor::dims_extractor(get_dims(), indices...);
    }
    API_EXPORT virtual std::pair<size_t const *, size_t> get_dims() const noexcept final
    {
        return std::pair<size_t const *, size_t>(&shape->dims[0], Rank);
    }
#if defined(NDEBUG) || defined(NO_SETDIMS_CHECK)
    API_EXPORT virtual bool set_dims(const size_t dims[]) override final { return false; }
    API_EXPORT virtual bool set_dims(const Tensor &prototype) override final { return false; }
#else
    API_EXPORT virtual bool set_dims(const size_t /* dims */[]) override
    {
        // for (int i = 0; i < Rank; i++) {
        //     assert(dims[i] == shape->dims[i]);
        // }
        return false;
    }
    API_EXPORT virtual bool set_dims(const Tensor &prototype) override
    {
        auto [dims_p, dims_n] = prototype.get_dims();
        assert(dims_n == Rank);
        return set_dims(dims_p);
    }
#endif
    // 'interface()' needs to be overriden in ConcreteTensor
    API_EXPORT float interface_scale() const { return this->interface().get_scale(); }
    API_EXPORT float interface_scale_recip() const { return this->interface().get_scale_recip(); }
    API_EXPORT int32_t interface_offset() const { return this->interface().get_offset(); }

    // for direct access to bulk_data, in contiguous tensors only
    //  data_ptr() can be assigned to.
    API_EXPORT std::conditional_t<is_indirect, void, storage_type *&> data_ptr()
    {
        if constexpr (!is_indirect) {
            return mem.bulk_data;
        }
    }
    API_EXPORT std::conditional_t<is_indirect, void, storage_type *const &> data_ptr() const
    {
        if constexpr (!is_indirect) {
            return mem.bulk_data;
        } else {
            return;
        }
    }

    // block table access
    API_EXPORT storage_type **blocktab_ptr() const { return (storage_type **)mem.get_block_list_ptr(); }
    API_EXPORT storage_type *&blocktab_at(size_t i)
    {
        if constexpr (!is_indirect) {
            assert(i == 0);
            return *(storage_type **)mem.get_block_list_ptr();
        } else {
            return ((storage_type **)mem.get_block_list_ptr())[i];
        }
    }
    API_EXPORT storage_type *const &blocktab_at(size_t i) const
    {
        if constexpr (!is_indirect) {
            assert(i == 0);
            return *(storage_type **)mem.get_block_list_ptr();
        } else {
            return ((storage_type **)mem.get_block_list_ptr())[i];
        }
    }
    API_EXPORT size_t blocktab_len() const { return mem.get_block_list_len(shape); }
    API_EXPORT size_t blocktab_blocksize() const { return mem.get_elements_per_block(shape); }
    API_EXPORT size_t blocktab_blocksize_bytes() const
    {
        return mem.get_elements_per_block(shape) * sizeof(storage_type);
    }

    // TODO: make total_storage elements have an optional bool parameter
    // to return in bytes; and then total_storage_bytes is a wrapper.
    API_EXPORT virtual size_t total_storage_bytes() const final override
    {
        return total_storage_elements() * sizeof(storage_type);
    }
    API_EXPORT virtual size_t total_storage_elements() const final override
    {
        size_t const total_elements = hnnx::product_of_array(shape->max_dims);
        return total_elements;
    }
    API_EXPORT virtual void *raw_data() noexcept override final { return mem.raw_data(); }
    API_EXPORT virtual void set_raw_data_despite_danger(void *buffer) override final
    {
        mem.set_raw_data_despite_danger(buffer);
    }

  protected:
    // Underlying code for change_{shape,pad,shape_pad}
    API_EXPORT void change_shapepad_impl(hnnx::Allocator &allocator, size_t const *const p_new_dims,
                                         size_t const *const p_new_pads = nullptr) // optional pads
    {
#if !defined(PREPARE_DISABLED)
        Shape_t newshape = *shape; // copy old shape
        if (p_new_dims) {
            for (int i = 0; i < Rank; i++)
                newshape.dims[i] = p_new_dims[i];
        }
        if (p_new_pads) {
            for (int i = 0; i < Rank; i++)
                newshape.pad[i] = p_new_pads[i];
        }
        newshape.max_dims = layout.pad(pad.pad_coords(newshape.dims, newshape.pad));
        // nake a persistent copy of new shape
        Shape_t const *const new_shape_p = Shape_t::canonical_shape(allocator.graph, newshape);
        // new_shape_p will be same pointer as shape, if shape wasn't changed. realloc_blocktab
        // checks for that.
        mem.realloc_blocktab(&allocator, shape, new_shape_p);
        shape = new_shape_p;
#else
        throw std::runtime_error("change_pad or shape w/o prepare");
#endif
    }

  public:
    // change the padding; and reallocate blocktab if it's larger as a result.
    // in any case, all of the block pointers are zeroed.
    // Can only be done early in prepare (i.e. just as Op is created).
    void change_pad(std::array<size_t, Rank> const &new_pad, hnnx::Allocator &allocator)
    {
        change_shapepad_impl(allocator, nullptr, new_pad.data());
    }
    // Used to change the shape of a tensor;
    // Can only be done early in prepare (i.e. just as Op is created). Initially used only to
    // support 'scratch' outputs.
    // Please use 'change_shape_pad' if you also want to change padding.
    void change_shape(std::array<size_t, Rank> const &new_dims, hnnx::Allocator &allocator)
    {
        change_shapepad_impl(allocator, new_dims.data());
    }
    // special entry point for use by 'generic' change_shape operation.
    void change_shape_arr(size_t const *const p_new_dims, hnnx::Allocator &allocator)
    {
        change_shapepad_impl(allocator, p_new_dims);
    }

    // Use instead of change_shape if you want to change the padding too
    // Can only be done early in prepare (i.e. just as Op is created).
    void change_shape_pad(std::array<size_t, Rank> const &new_dims, std::array<size_t, Rank> const &new_pad,
                          hnnx::Allocator &allocator)
    {
        change_shapepad_impl(allocator, new_dims.data(), new_pad.data());
    }

    template <typename... ind_types> API_EXPORT storage_type const *const *block_ptr_address(ind_types... inds) const
    {
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return mem.block_ptr_addr(shape, coords);
    }
    template <typename... ind_types> API_EXPORT storage_type *const *block_ptr_address(ind_types... inds)
    {
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return mem.block_ptr_addr(shape, coords);
    }
    template <typename... ind_types> API_EXPORT storage_type const *block_ptr(ind_types... inds) const
    {
        return *block_ptr_address(inds...);
    }
    template <typename... ind_types> API_EXPORT storage_type *block_ptr(ind_types... inds)
    {
        return *block_ptr_address(inds...);
    }

    API_EXPORT std::conditional_t<is_indirect, BlockTableAccessor<Linfo>, void> blocktable_accessor() const
    {
        if constexpr (is_indirect) {
            return BlockTableAccessor<Linfo>(*this);
        }
    }

    // this only makes sense for indirect tensors.
    // For the contiguous dimensions, we set the return values to 0 avoid
    // mixing up last indirect dimension and the contiguous dimensions.
    // This is for marking purposes. We don't expect either ret[indir - 1] or ret[ind]
    // to be used for index calculation.
    API_EXPORT std::conditional_t<is_indirect, std::array<size_t, Linfo::Rank>, void> tile_strides() const
    {
        if constexpr (is_indirect) {
            std::array<size_t, Linfo::Rank> ret = {0};
#ifndef PT_512B
            ret[Linfo::Rank - 1] = 1;
            for (int i = Linfo::Rank - 2; i >= 0; i--) {
                ret[i] = ret[i + 1] * (shape->max_dims[i + 1] / layout.ChunkSizes[i + 1]);
            }
#else
            constexpr int indir = TLayout::indirect_ranks;
            for (size_t ind = indir; ind < Linfo::Rank; ++ind) {
                ret[ind] = 0;
            }
            ret[indir - 1] = 1;
            for (int i = indir - 2; i >= 0; i--) {
                ret[i] = ret[i + 1] * (shape->max_dims[i + 1] / layout.ChunkSizes[i + 1]);
            }
#endif
            return ret;
        } else {
            return;
        }
    }

    // get_raw_addr(...) on this class gives a storage_type *.
    template <typename... ind_types> API_EXPORT storage_type const *get_raw_addr(ind_types... inds) const
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return (storage_type const *)element_addr0(Rank, coords.data());
    }
    template <typename... ind_types> API_EXPORT storage_type *get_raw_addr(ind_types... inds)
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return (storage_type *)element_addr0(Rank, coords.data());
    }
    template <typename... ind_types> API_EXPORT storage_type const &get_raw(ind_types... inds) const
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *(storage_type const *)element_addr0(Rank, coords.data());
    }
    template <typename... ind_types> API_EXPORT storage_type &get_raw(ind_types... inds)
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *(storage_type *)element_addr0(Rank, coords.data());
    }

    template <typename... ind_types> API_EXPORT inline size_t get_addr_offset(ind_types... inds) const
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return get_chunk_offset0(Rank, coords.data());
    }

    template <typename... ind_types> API_EXPORT inline size_t get_addr_offset(ind_types... inds)
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return get_chunk_offset0(Rank, coords.data());
    }
    // tile interface. These are defined in tile_extract.h
    API_EXPORT virtual void const *read_tile(unsigned flags, void *buffer, size_t b, int h, int w,
                                             int d) const override;
    API_EXPORT virtual void write_tile(unsigned flags, void const *buffer, size_t b, int h, int w, int d) override;
    API_EXPORT virtual unsigned tile_support_bits() const override;

    // Return a reference to *this; useful to get the layout base class reference
    // for any tensor class which has one.
    // So if you call func(in.layout_base(), out.layout_base()), where 'func'
    // is a template func, it will be specialized according to the layout of
    // in and out, not the subclass.
    API_EXPORT LayoutTensor &layout_base() { return *this; }
    API_EXPORT LayoutTensor const &layout_base() const { return *this; }

    // checksum for debug
    [[gnu::noinline]] API_EXPORT virtual uint64_t get_checksum() const override
    {
        // NOLINTNEXTLINE(misc-const-correctness): Don't const this variable
        uint64_t chk = 0;
        if constexpr (Rank == 4) {
            auto [batch, heights, width, depth] = this->get_dims_4();
            // TODO : maybe add a special case for R4flat layout/no padding; one call to checksum_bytes.
            if (batch && heights && width && depth) {
                storage_type const x0 = *(storage_type const *)this->get_raw_addr(0, 0, 0, 0);
                for (size_t b = 0; b < batch; b++) {
                    for (size_t h = 0; h < heights; h++) {
                        for (size_t w = 0; w < width; w++) {
                            for (size_t d = 0; d < depth; d++) {
                                storage_type x = *(storage_type const *)this->get_raw_addr(b, h, w, d);
                                x ^= x0;
                                union {
                                    storage_type as_x;
                                    uint8_t as_byte[sizeof(storage_type)];
                                } uu = {x};
                                chk = hnnx::checksum_bytes(chk, uu.as_byte, sizeof(storage_type));
                            }
                        }
                    }
                }
                chk ^= x0;
            }
        }
        return chk;
    }

  protected:
    // element_addr is delegated to the particular specialization of layout_mem
    // virtual method 'element_addr' is defined only in the concrete subclasses, and calls this.
    // (in addition to sometimes returning an interface)
    ALWAYSINLINE void *element_addr0(size_t rank, const SIdx coords_in[]) const noexcept
    {
        if constexpr (!is_singular) {
            return mem.element_addr(shape, rank, coords_in, shape->dims);
        } else {
            return mem.raw_data();
        }
    }

    ALWAYSINLINE size_t get_chunk_offset0(size_t rank, const SIdx coords_in[]) const noexcept
    {
        if constexpr (!is_singular) {
            return mem.get_chunk_offset(shape, rank, coords_in);
        } else {
            return 0;
        }
    }

    // compare_sametype is not overloaded here; LayoutTensor is an abstract class

    // This is called from ConcreteTensor::compare_sametype to fully compare two tensors
    // which are already known to be the same type (and have same interface)
    [[gnu::noinline]] API_EXPORT int compare_sametype_layout(LayoutTensor const *rhs) const
    {
        if (shape->dims != rhs->shape->dims) {
            return std::lexicographical_compare(shape->dims.begin(), shape->dims.end(), rhs->shape->dims.begin(),
                                                rhs->shape->dims.end())
                           ? -1
                           : 1;
        }
        if (is_padded) {
            if (shape->max_dims != rhs->shape->max_dims) {
                return std::lexicographical_compare(shape->max_dims.begin(), shape->max_dims.end(),
                                                    rhs->shape->max_dims.begin(), rhs->shape->max_dims.end())
                               ? -1
                               : 1;
            }
            // TODO: compare padding too. Maybe have a Padding method for this.
        }
        // compare memory now (delegate to layout_mem).
        return mem.compare_memory(shape, rhs->mem);
    }
    // allocation and enumeration.
    [[gnu::noinline]] API_EXPORT void allocate_layout(hnnx::Allocator &allocator, unsigned options, MemoryClass mclass)
    {
        // get the pointer to block table; and number of entries in it.
        void **const blocktab = this->mem.get_block_list_ptr();
        size_t const nblocks = this->mem.get_block_list_len(this->shape);
        size_t const blocksize = sizeof(storage_type) * this->mem.get_elements_per_block(this->shape);
        size_t const align = traits::is_indirect ? blocksize : std::min(size_t(256), sizeof(storage_type));
        if constexpr (traits::is_singular) {
            options |= unsigned(hnnx::AllocOpts_packed);
        }
        allocator.allocate_n(blocktab, // pointer to pointers,
                             nblocks, // number of pointers
                             blocksize, align, mclass, options, this->get_dtype());
    }
    [[gnu::noinline]] API_EXPORT void enum_memory_blocks_layout(hnnx::MemBlockEnumerator &en, MemoryClass mclass) const
    {
        // get the pointer to block table; and number of entries in it.
        void **const blocktab = this->mem.get_block_list_ptr();
        size_t const nblocks = this->mem.get_block_list_len(this->shape);
        size_t const blocksize = sizeof(storage_type) * this->mem.get_elements_per_block(this->shape);
        en.supply_blocks(this, mclass, (void *const *)blocktab, nblocks, blocksize);
    }
    // called from find_content_hash in the ConcreteTensor class. hash_in includes
    // hash of dtype and interface.
    [[gnu::noinline]] API_EXPORT uint32_t find_content_hash_layout(uint32_t hash_in, bool is_float) const noexcept
    {
        uint32_t h = hash_in ^ (Linfo::Rank * 0x102401u);
        h = Tensor::build_hash(shape->dims.data(), Linfo::Rank, hash_in);
        if (is_padded) {
            h = Tensor::build_hash(shape->max_dims.data(), Linfo::Rank, h);
            // TODO: including padding too (or instead)
        }
        return mem.find_content_hash(shape, h, is_float);
    }
    API_EXPORT static const char *code_to_type_name() { return TensorTypeStruct<LayoutTensor<Linfo>>::name; }
};

//
// Constructors of ConcreteTensor:
// ConcreteTensor(const Op * producer_in, const OutputDef &def, Graph &graph_in)
//    - build for given shape, attached to given producer.
//  ConcreteTensor(const Op *producer_in, const OutputDef &def, Graph & graph_in, T * data_in)
//    - same, but initialize pointer to given. Only available in 'flat' tensors.
//  ConcreteTensor(hnnx::Deserz & dctx)
//    - deserialize. Note that dctx contains a grap ref.
//  ConcreteTensor(const ConcreteTensor &old, hnnx::Allocator *allocator,Tensor::clone_mode cmode)
//    - 'clone duplicate' of the given tensor. Note that cmode is ignored.
//

template <typename Tinfo> class ConcreteTensor : public LayoutTensor<typename Tinfo::Lconfig> {
  protected:
    using Interface_t = typename Tinfo::Interface_t;
    using Layout_t = typename Tinfo::Tlayout;
    using Pad_t = typename Tinfo::Pad_t;
    static constexpr DType dtype = dtype_of_type<Interface_t>();
    API_EXPORT static constexpr bool is_indirect = Tinfo::is_indirect;
    API_EXPORT static constexpr unsigned Rank = Layout_t::Rank;
    using BaseLayout = LayoutTensor<typename Tinfo::Lconfig>;
    static constexpr bool is_singular = BaseLayout::traits::is_singular;
    using BaseRT = typename BaseLayout::BaseRT;

    // make sure it's compatible with supplied base class
    static_assert(Rank == BaseLayout::Rank && is_indirect == BaseLayout::traits::is_indirect &&
                          std::is_same<Layout_t, typename BaseLayout::traits::layout_type>::value &&
                          std::is_same<Pad_t, typename BaseLayout::traits::pad_type>::value,
                  "incompatible base class for ConcreteTensor");

    Interface_t const *interface_typed() const { return static_cast<Interface_t const *>(this->interface_ptr); }

  public:
    API_EXPORT const char *true_name() const override { return Tinfo::typetag; }
    using Accessor_t = typename Interface_t::Accessor;
    using Const_Accessor_t = typename Interface_t::AccessorRO;
    using element_type = typename Interface_t::element_type;

    struct API_EXPORT traits : public BaseLayout::traits {
        static constexpr DType dtype = ConcreteTensor::dtype;
        using element_type = typename dtype_traits<dtype>::element_type;
        using raw_type = element_type; // result from get_raw()
        using interface_type = Interface_t;
        static constexpr MemoryClass memclass = Tinfo::memclass;
    };
    //
    //  - build for given shape, attached to given producer.
    //  - pass the nase class ctor a specialized ctor, it uses to make the interface
    //   from the output def.
    API_EXPORT ConcreteTensor(const Op *producer_in, const OutputDef &def, Graph &graph_in)
        : BaseLayout(producer_in, def, graph_in, hnnx::make_interface<Interface_t>::from_odef)
    {
    }
    API_EXPORT ConcreteTensor(const Op *producer_in, const OutputDef &def, Graph &graph_in, element_type *data_in)
        : BaseLayout(producer_in, def, graph_in, hnnx::make_interface<Interface_t>::from_odef)
    {
        this->mem.set_raw_data_despite_danger((void *)data_in);
    }
    //   - deserialize. Note that dctx contains a graph ref.
    //   We pass the base class a pointer to specialized function, which it uses to
    //  deserialize the interface.
    API_EXPORT explicit ConcreteTensor(hnnx::Deserz &dctx)
        : BaseLayout(dctx, &hnnx::make_interface<Interface_t>::from_deser)
    {
    }
    //    - 'clone duplicate' of the given tensor. Note that cmode is ignored.
    API_EXPORT ConcreteTensor(const ConcreteTensor &old, hnnx::Allocator *allocator, Tensor::clone_mode cmode,
                              const OutputDef *def = nullptr)
        : BaseLayout(old, allocator,
                     (def ? hnnx::make_interface<Interface_t>::from_odef(allocator->graph, *def) : old.interface_ptr),
                     cmode)
    {
    }

    API_EXPORT virtual DTypeScaleOff get_dtype_intfc() const noexcept override
    {
        return interface_typed()->get_dtype_scaleoff();
    }

    API_EXPORT virtual hnnx::InterfaceRef interface() const noexcept override final
    {
        return interface_typed()->get_refobj();
    }
    API_EXPORT float interface_scale() const { return interface_typed()->get_scale(); }
    API_EXPORT float interface_scale_recip() const { return interface_typed()->get_scale_recip(); }
    API_EXPORT int32_t interface_offset() const { return interface_typed()->get_offset(); }

    API_EXPORT ALWAYSINLINE const element_type *element_ptr(size_t rank, const SIdx coords[]) const
    {
        return (element_type const *)this->element_addr0(rank, coords);
    }
    API_EXPORT ALWAYSINLINE element_type *element_ptr(size_t rank, const SIdx coords[])
    {
        return (element_type *)this->element_addr0(rank, coords);
    }

    // Some methods return the same thing as in LayoutTensor, but
    // with the type being element_type instead of storage_type.
    API_EXPORT std::conditional_t<is_indirect, void, element_type *&> data_ptr()
    {
        if constexpr (!is_indirect) {
            return (element_type *&)this->mem.bulk_data;
        }
    }
    API_EXPORT std::conditional_t<is_indirect, void, element_type *const &> data_ptr() const
    {
        if constexpr (!is_indirect) {
            return (element_type *const &)this->mem.bulk_data;
        }
    }

    // block table access
    API_EXPORT element_type **blocktab_ptr() const { return (element_type **)this->mem.get_block_list_ptr(); }
    API_EXPORT element_type *&blocktab_at(size_t i) { return (element_type *&)BaseLayout::blocktab_at(i); }
    API_EXPORT element_type *const &blocktab_at(size_t i) const
    {
        return (element_type *const &)BaseLayout::blocktab_at(i);
    }

    template <typename... ind_types> API_EXPORT element_type const *const *block_ptr_address(ind_types... inds) const
    {
        return (element_type const *const *)BaseLayout::block_ptr_address(inds...);
    }
    template <typename... ind_types> API_EXPORT element_type *const *block_ptr_address(ind_types... inds)
    {
        return (element_type *const *)BaseLayout::block_ptr_address(inds...);
    }
    template <typename... ind_types> API_EXPORT element_type const *block_ptr(ind_types... inds) const
    {
        return *this->block_ptr_address(inds...);
    }
    template <typename... ind_types> API_EXPORT element_type *block_ptr(ind_types... inds)
    {
        return *this->block_ptr_address(inds...);
    }

    // direct access methods.
    //
    template <typename... ind_types> API_EXPORT Const_Accessor_t operator()(ind_types... inds) const
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {static_cast<SIdx>(inds)...};
        return Const_Accessor_t(this->element_addr0(Rank, coords.data()), interface_typed());
    }
    template <typename... ind_types> API_EXPORT Accessor_t operator()(ind_types... inds)
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return Accessor_t(this->element_addr0(Rank, coords.data()), interface_typed());
    }
    template <typename... ind_types> API_EXPORT element_type const &get_raw(ind_types... inds) const
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *(element_type const *)this->element_addr0(Rank, coords.data());
    }
    template <typename... ind_types> API_EXPORT element_type &get_raw(ind_types... inds)
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return *(element_type *)this->element_addr0(Rank, coords.data());
    }
    template <typename... ind_types> API_EXPORT element_type const *get_raw_addr(ind_types... inds) const
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return (element_type const *)this->element_addr0(Rank, coords.data());
    }
    template <typename... ind_types> API_EXPORT element_type *get_raw_addr(ind_types... inds)
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return (element_type *)this->element_addr0(Rank, coords.data());
    }

    template <typename... ind_types> API_EXPORT inline size_t get_addr_offset(ind_types... inds) const
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return this->get_chunk_offset0(Rank, coords.data());
    }

    template <typename... ind_types> API_EXPORT inline size_t get_addr_offset(ind_types... inds)
    {
        static_assert(is_singular || Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const std::array<SIdx, Rank> coords = {{static_cast<SIdx>(inds)...}};
        return this->get_chunk_offset0(Rank, coords.data());
    }

    API_EXPORT virtual uint32_t get_tensor_format_code() const noexcept override
    {
        return Tensor::formatcode_for_general<traits>();
    }

    API_EXPORT virtual uint32_t get_tensor_info() const noexcept override
    {
        return Tensor::pack_tensor_info(traits::dtype, Rank, traits::memclass);
    }
    // allocation and enumeration.
    API_EXPORT virtual void allocate_func(hnnx::Allocator &allocator, unsigned options) override final
    {
        this->allocate_layout(allocator, options, traits::memclass);
    }
    API_EXPORT virtual void enum_memory_blocks(hnnx::MemBlockEnumerator &en) const override
    {
        this->enum_memory_blocks_layout(en, traits::memclass);
    }
    // hash the dtype and interface, and let find_content_hash_layout do the rest.
    API_EXPORT virtual uint32_t find_content_hash() const noexcept override final
    {
        uint32_t const h = interface().interface_hash() ^ mulu32_modular(unsigned(dtype), 0x107301);
        static constexpr bool is_float = dtype_traits<dtype>::is_float;
        return this->find_content_hash_layout(h, is_float);
    }

  protected:
    // because this (may) need to return an "InterfaceRef" via iref pointer, it's defined
    // here in the 'Concrete' class, but it uses the non-virtual 'element_addr0'
    // in the LayoutTensor base class to find the address, and adds the InterfaceRef if
    // requested.
    API_EXPORT virtual ALWAYSINLINE void *
    element_addr(size_t rank, const SIdx coords_in[],
                 hnnx::InterfaceRef *const iref = nullptr) const noexcept final override
    {
        if (iref) *iref = interface_typed()->get_refobj();
        return this->element_addr0(rank, coords_in);
    }
    API_EXPORT virtual int compare_sametype(const Tensor *rhs_in) const override
    {
        // compare the interface, and then all the rest is done in compare_sametype_layout.
        auto *rhs = static_cast<ConcreteTensor const *>(rhs_in);
        if (rhs == nullptr) {
            throw std::runtime_error("compare_sametype called with null rhs");
        }
        int const icmp = interface_typed()->compare(*rhs->interface_typed());
        if (icmp != 0) return icmp;
        return this->compare_sametype_layout(rhs);
    }

    API_EXPORT virtual void **clone_util(hnnx::Allocator *allocator, std::unique_ptr<Tensor> *tensp,
                                         Tensor::tensor_blockinfo *infop,
                                         const OutputDef *od /* nullptr */) const override
    {
        void **retval = nullptr;
        ConcreteTensor const *newtens = nullptr;
        if (tensp) {
            if constexpr (!dtype_traits<dtype>::is_quant) od = nullptr;
            *tensp = std::make_unique<ConcreteTensor>(*this, allocator, Tensor::clone_mode::duplicate, od);
            newtens = static_cast<ConcreteTensor const *>(tensp->get());
            retval = (void **)newtens->mem.get_block_list_ptr();
        }
        if (infop) {
            infop->setup(traits::dtype, traits::memclass);
            infop->blkptrs = (void **)this->mem.get_block_list_ptr();
            // pretend that a pointer to Shape<Rank> is really a pointer to its base class ShapeFlags
            // we provide a pointer to the shape field in the cloned tensor, if applicable; otherwise in 'this'.
            infop->shapepp = (const hnnx::ShapeFlags *const *)&(newtens ? newtens : this)->shape;
            infop->interfacepp = &(newtens ? newtens : this)->interface_ptr;
            infop->nblocks = this->mem.get_block_list_len(this->shape);
            infop->blocksize = sizeof(element_type) * this->mem.get_elements_per_block(this->shape);
            infop->is_indirect = this->is_indirect;
            infop->is_chunked = traits::is_chunked;
            infop->is_variable_block = traits::is_variable_block;
            return retval;
        }
        return nullptr;
    }
    API_EXPORT static const char *code_to_type_name() { return TensorTypeStruct<ConcreteTensor<Tinfo>>::name; }
};

namespace hnnx {
// map a tensor type to its layout tensor.
template <typename TT> using layout_of = typename tensor_traits<TT>::layouttensor_type;
} // namespace hnnx

POP_VISIBILITY()

#endif // HEXNN_TENSOR_CONCRETE_H
