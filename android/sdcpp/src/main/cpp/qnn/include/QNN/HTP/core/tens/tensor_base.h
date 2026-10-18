// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#ifndef HEXNN_TENSOR_BASE_H
#define HEXNN_TENSOR_BASE_H 1
#include "block_enumeration.h"
#include "dtype_enum.h"
#include "interface_defs.h"
#include "macros_attribute.h"
#include "weak_linkage.h"
#include "padding.h"
#include "interface.h"
#include "template_help.h"
#include "memory_layout.h"
#include "serialize_oplist.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <stdexcept>

class Op;

namespace hnnx {
class Allocator;
class Serializer;
class Deserializer;
struct ShapeFlags;
} // namespace hnnx

// Base 'Tensor' class, also:
//  - FakeTensor
//  - RankedTensor
//  - TensorShape
//  - TensorSclrDT<DType> (known as TensorScalar<T>)
//
//  LayoutTensor and ConcreteTensor are in tensor_concrete.h
//
PUSH_VISIBILITY(default)

//////////////////////////////////////////////////////////////////////////////////////////
/// @brief compile-time traits of tensor classes
/// E.g. the construct tensor_traits<TYPE>::element_type will obtain the element_type
/// of any Tensor subclass which has one.
/// Note that tensor_traits<Tensor> has no defined attributes.
///
/// Full list of traits is below
///
/// These are defined in all non-abstract Tensor subclasses:
///  - constexpr DType dtype            // always present (except Tensor,RankedTensor,LayoutTensor); sometimes UNKNOWN
///  - constexpr unsigned rank          // always present (except Tensor); sometimes 0
///  - typedef element_type;            // always present (except Tensor,RankedTensor,LayoutTensor); void in TensorShape
///  - typedef storage_type             // always present (except Tensor,RankedTensor); void in TensorShape
///
/// In LayoutTensor and ConcreteTensor:
///  - typedef layouttensor_type        // The LayoutTensor<> class
///  - typedef layout_type
///  - typedef pad_type                 // Padding<rank> or NoPadding<rank>
///  - constexpr bool has_padding
///  - constexpr bool is_chunked;
///  - constexpr bool is_indirect;      // usually same as is_chunked; always <= is_chunked
///  - constexpr bool is_variable_block; // always <= is_indirect; implies that blocksize depends on at least 1 dim.
///  - constexpr unsigned indirect_ranks;  // number of dims which address blocktable; < rank when 'is_variable_block',
///                                        // not relevant in other cases.
///  - constexpr bool is_singular;     // set when SingularLayout is used, Only when !is_chunked, !is_indirect
///  - typedef raw_type                 // See below [1]
///
/// Only in ConcreteTensor:
///  - constexpr MemoryClass memclass;
///
/// Only in ConcreteTensor, TensorScalar:
///  - typedef interface_type
///
///  [1] raw_type is defined in the classes which have get_raw(...),
///     and is the type which get_raw returns a ref to.
///     For LayoutTensor, it is the same as storage_type; for ConcreteTensor,
///     it is the same as element_type.
///
template <typename TENST> using tensor_traits = typename TENST::traits;

//////////////////////////////////////////////////////////////////////////////////////////

/*
 * Now that we have Interfaces and Accessors, which we will use to give a consistent interface to Tensors,
 * let's work on the actual Tensors
 */

/*
 * @class Tensor
 *
 * @brief This is the abstract base class for Tensors.
 * All tensors allow you to index into them with foo(a,b,c);
 * You can query rank, dim, etc
 *
 * But, you're probably better off with one of the more specific Tensor types for performance,
 * since a lot of the virtual functions become trivial for the compiler if they can be inlined.
 *
 */

class Tensor {
  public:
    enum class clone_mode {
        duplicate,
        UNUSED_persistent,
    };

    // Use with 'dims' to query dimension sizes by name e.g. auto [h, d] = tensor.dims(Tensor::HEIGHT, Tensor::DEPTH)
    enum dimensions { BATCH, HEIGHT, WIDTH, DEPTH, CHANNEL };

    API_EXPORT virtual hnnx::InterfaceRef interface() const noexcept = 0;

    struct traits { // empty
    };

    API_EXPORT virtual const char *true_name() const { return typeid(*this).name(); }
    API_EXPORT explicit Tensor(const Op * /* producer_in */) {}
    API_EXPORT explicit Tensor(hnnx::Deserz &) {}
    API_EXPORT Tensor(const Tensor & /* old */, hnnx::Allocator * /* allocator */, clone_mode) {}
    API_EXPORT virtual ~Tensor() {} // virtual destructor

    Tensor(Tensor const &) = delete;
    Tensor(Tensor &&) = delete;
    Tensor &operator=(Tensor const &) = delete;
    Tensor &operator=(Tensor &&) = delete;

    API_EXPORT virtual size_t rank() const noexcept = 0; // What's the rank of this tensor?
    API_EXPORT virtual size_t dim(size_t index) const noexcept = 0; // What's the length of some dimension?
    API_EXPORT virtual std::pair<size_t const *, size_t>
    get_dims() const noexcept = 0; // return rank, and address of dims[0..n-1]

    // this is the first virtual method defined externally.
    API_EXPORT virtual uint32_t find_content_hash() const noexcept; // find 'content hash' of the data.

  protected:
    // Note, this is a const method returning a non-const pointer;
    // but we only allow it to publicly return a non-const
    // pointer when used in non-const wrapper methods.
    // if 'iref' is not null, *iref is also set to what interface() would return.
    API_EXPORT virtual void *element_addr(size_t rank, SIdx const coords_in[],
                                          hnnx::InterfaceRef *iref = nullptr) const noexcept = 0;
    // this is for implementing dims(..indices...) in Tensor and subclasses.
    // The 'dims_r' is a return from get_dims() method.
    template <typename... T> //
    static std::array<size_t, sizeof...(T)> //
    dims_extractor(std::pair<size_t const *, size_t> const dims_r, T... indices)
    {
        auto const read_dim = [dims_r](unsigned i) -> size_t { return (i < dims_r.second) ? dims_r.first[i] : 1; };
        return {read_dim(indices)...};
    }
    // this is for implementing dims()
    template <unsigned R, typename... T> //
    static constexpr std::array<size_t, R> //
    dims_extractor_all(std::pair<size_t const *, size_t> const dims_r)
    {
        std::array<size_t, R> result{};
        for (unsigned i = 0; i < R; i++) {
            result[i] = (i < dims_r.second) ? dims_r.first[i] : 1;
        }
        return result;
    }

  public:
    // element_ptr on insufficiently specialized class gives the result as a void *.
    API_EXPORT ALWAYSINLINE void const *element_ptr(size_t rank, const SIdx coords[]) const
    {
        return (void const *)element_addr(rank, coords);
    }
    API_EXPORT ALWAYSINLINE void *element_ptr(size_t rank, const SIdx coords[]) { return element_addr(rank, coords); }

    API_EXPORT std::tuple<size_t, size_t, size_t, size_t> get_dims_4() const
    {
        size_t const *ptr = nullptr;
        size_t n = 0;
        std::tie(ptr, n) = get_dims(); // virtual call
        if (n != 4) throw std::runtime_error("rank not 4");
        return std::make_tuple(ptr[0], ptr[1], ptr[2], ptr[3]);
    }
    // this is a common case.
    API_EXPORT std::tuple<size_t, size_t> get_dims_1_2() const
    {
        size_t const *ptr = nullptr;
        size_t n = 0;
        std::tie(ptr, n) = get_dims(); // virtual call
        if (n < 3) throw std::runtime_error("rank not >=3");
        return std::make_tuple(ptr[1], ptr[2]);
    }

    ALWAYSINLINE std::array<size_t, 4> dims() const
    { // make compatible with typical concrete tensor.
        return dims_extractor_all<4>(get_dims());
    }

    template <typename... T> API_EXPORT const std::array<size_t, sizeof...(T)> dims(T... indices) const
    {
        return dims_extractor(get_dims(), indices...);
    }

    API_EXPORT virtual DTypeScaleOff get_dtype_intfc() const noexcept = 0;
    // if you need more than one of these, it is recommended to unpack
    // the result from get_dtype_intfc()
    API_EXPORT DType get_dtype() const { return get_dtype_intfc().dtype; }
    API_EXPORT float interface_scale() const { return get_dtype_intfc().scale; }
    API_EXPORT NN_INT32_T interface_offset() const { return get_dtype_intfc().offset; }
    API_EXPORT OutputDef gen_output_def() const;

    template <typename... ind_types> API_EXPORT GenericAccessorRO operator()(ind_types... inds) const
    {
        const std::array<SIdx, sizeof...(ind_types)> indarr = {static_cast<SIdx>(inds)...};
        hnnx::InterfaceRef intfc = hnnx::InterfaceRef::make_null();
        void *const ptr = element_addr(sizeof...(ind_types), indarr.data(), &intfc);
        return GenericAccessorRO(ptr, intfc);
    }
    template <typename... ind_types> API_EXPORT GenericAccessor operator()(ind_types... inds)
    {
        const std::array<SIdx, sizeof...(ind_types)> indarr = {static_cast<SIdx>(inds)...};
        hnnx::InterfaceRef intfc = hnnx::InterfaceRef::make_null();
        void *const ptr = element_addr(sizeof...(ind_types), indarr.data(), &intfc);
        return GenericAccessor(ptr, intfc);
    }

    /*
     * Returned by virtual method get_tensor_format_code:
     *
     *                                   (General)           (Shape)             (Scalar)
     *  Bits  3:0  dtype code             x                  0 = UNKNOWN          x
     *  Bits  5:4  (reserved, zero)
     *  Bits  7:6  log2(element_size)     x                  0                    x
     *  Bits 11:8  rank                   x                  x                    0
     *  Bits 15:12 (reserved, 0)
     *  Bit  16     is_tcm                x                  0                    0
     *  Bit  17     is_quantized          x                  0                    x
     *  Bit  18     is_indirect           x                  0                    0
     *  Bit  19     is_chunked            x                  0                    0
     *  Bit  20     is_not_flat           x                  0                    0
     *  Bits  27:31	 (reserved, 0)
     *  Bits 31:28  mode                  tensMODE_general   tensMODE_shape       tensMODE_scalar
	 *-------------------------------------
     * Returned by get_tensor_info():
     * This is a bit weird, due to a legacy bug, but I'm restating it as below,
     * which remains compatible:
     *
     *  For Concrete tensor:
     *      Bits 3:0    DType
     *      Bits 7:4    '0001'
     *      Bits 11:8   rank
     *      Bits 15:12  '0000'
     *      Bits 19:16  memory class
     *      Bits 27:20  zero
     *      Bits 31:28  tensMODE_general
     *
     *  For Shape and Scalar tensors: Bits 31:28 contain tensMODE_shape, or tensMODE_scalar; others bits ar 0.
     *  Classes which cannot be serialized return 0 in the upper 4 bits.
     */
    static constexpr unsigned tformat_dtype_shift = 0u;
    static constexpr unsigned tformat_dtype_mask = 0xFu;
    static constexpr unsigned tformat_log2sz_shift = 6u;
    static constexpr unsigned tformat_log2sz_mask = 3u;
    static constexpr unsigned tformat_rank_shift = 8u;
    static constexpr unsigned tformat_rank_mask = 0xFu;
    static constexpr unsigned tformat_is_tcm = 1u << 16u;
    static constexpr unsigned tformat_is_quantized = 1u << 17u;
    static constexpr unsigned tformat_is_indirect = 1u << 18u;
    static constexpr unsigned tformat_is_chunked = 1u << 19u;
    static constexpr unsigned tformat_is_not_flat = 1u << 20u;
    static constexpr unsigned tformat_is_singular = 1u << 21u;
    static constexpr unsigned tformat_tmode_shift = 28u;
    static constexpr unsigned tformat_tmode_mask = 0xFu;

  protected:
    template <typename IFC> static constexpr uint32_t formatcode_for_interface()
    {
        constexpr DType dt = dtype_of_type<IFC>();
        uint32_t result = unsigned(dt);
        constexpr unsigned elbytes = sizeof(typename IFC::element_type);
        constexpr unsigned log2sz = (elbytes == 8) ? 3 : (elbytes == 4) ? 2 : (elbytes == 2) ? 1 : 0;
        static_assert(elbytes == (1u << log2sz));
        result |= log2sz << tformat_log2sz_shift;
        if (dtype_traits<dt>::is_quant) result |= tformat_is_quantized;
        return result;
    }
    template <typename TRAITS> static constexpr uint32_t formatcode_for_general()
    {
        constexpr unsigned rankval = TRAITS::rank;
        uint32_t result = formatcode_for_interface<typename TRAITS::interface_type>();
        result |= (rankval << tformat_rank_shift);
        if (TRAITS::memclass == MemoryClass::TCM) result |= tformat_is_tcm;
        if (TRAITS::is_indirect) result |= tformat_is_indirect;
        if (TRAITS::is_chunked) result |= tformat_is_chunked;
        if (!std::is_base_of_v<FlatMemoryLayout<rankval>, typename TRAITS::layout_type>) {
            result |= tformat_is_not_flat;
        }
        if (TRAITS::is_singular) result |= tformat_is_singular;
        return (hnnx::SerOpsInterface::tensMODE_general << tformat_tmode_shift) | result;
    }

    template <unsigned RANK> static constexpr uint32_t formatcode_for_shape()
    {
        static_assert(RANK <= tformat_rank_mask);
        return (hnnx::SerOpsInterface::tensMODE_shape << tformat_tmode_shift) | (RANK << tformat_rank_shift);
    }
    template <typename IFC> static constexpr uint32_t formatcode_for_scalar()
    {
        return (hnnx::SerOpsInterface::tensMODE_scalar << tformat_tmode_shift) | formatcode_for_interface<IFC>();
    }

    static constexpr uint32_t pack_tensor_info(DType type, uint32_t rank, MemoryClass mclass)
    {
        uint32_t tinfo = 0x10;
        tinfo |= static_cast<uint32_t>(type) & 0xFu;
        tinfo |= (rank & 0xFu) << 8u;
        tinfo |= (static_cast<uint32_t>(mclass) & 0xF) << 16u;
        tinfo |= hnnx::SerOpsInterface::tensMODE_general << tformat_tmode_shift;
        return tinfo;
    }

  public:
#ifndef PREPARE_DISABLED
    virtual std::string get_shape_info() const { return {}; }
#endif
    API_EXPORT virtual uint32_t get_tensor_info() const noexcept; // returns 0;
    API_EXPORT virtual uint32_t get_tensor_format_code() const noexcept; // returns 0;
    // returns false if the dims are the same; true if different, or maybe different.
    API_EXPORT virtual bool set_dims(const size_t dims[]) = 0; // Set the shape of the tensor
    API_EXPORT virtual bool set_dims(const Tensor &prototype) = 0; // Set the shape of the tensor same as another.

    API_EXPORT void allocate(hnnx::Allocator &allocator, unsigned options = 0)
    {
        return allocate_func(allocator, options);
    }

    /* EJP: FIXME: temporary functions */
    /*
	 * Some of these functions are convenient for now, but don't necessarily
	 * need to live for a long time if we find better ways of doing things.
	 */
    API_EXPORT virtual void *raw_data() noexcept = 0; // Get pointer to raw data
    API_EXPORT void const *raw_data_const() const noexcept { return const_cast<Tensor *>(this)->raw_data(); }
    API_EXPORT virtual void set_raw_data_despite_danger(void * /* buffer */)
    {
        assert(false && "Invalid to set raw pointer on this type of tensor");
    }
    API_EXPORT virtual size_t total_storage_elements() const = 0;
    API_EXPORT virtual size_t total_storage_bytes() const = 0;
    API_EXPORT const char *truetype() const noexcept { return typeid(*this).name(); }

    // Append the set of allocated memory blocks to blocklist.
    API_EXPORT void get_memory_blocks(hnnx::blockid_set_t &blocklist, int mc_sel = -1) const;
    API_EXPORT void get_memory_blocks(hnnx::blockid_set_t &blocklist, MemoryClass mc) const
    {
        get_memory_blocks(blocklist, int(mc));
    }
    // return the set of memory blocks
    API_EXPORT hnnx::blockid_set_t get_memory_blocks(int mc_sel = -1) const;
    API_EXPORT hnnx::blockid_set_t get_memory_blocks(MemoryClass mc) const { return get_memory_blocks(int(mc)); }

    // Supply the allocated memory blocks to the enumerator.
    API_EXPORT virtual void enum_memory_blocks(hnnx::MemBlockEnumerator &) const = 0;

    // The 'ef' parameter to these functions is a callable (function, lambda, std::function...)
    // compatible with MemBlockEnumerator::supply_blocks_func
    template <typename ENFUNC> API_EXPORT void enum_memory_blocks_withfunc(ENFUNC &&ef) const
    {
        hnnx::MemBlockEnumWrapper<std::remove_reference_t<ENFUNC>> enumer(std::forward<ENFUNC>(ef));
        this->enum_memory_blocks(enumer);
    }
    // The 'rf' parameter to these functions is a callable (function, lambda, std::function...)
    // .. called as ( Tensor const *, void *old_blkid) -> void *new_blkid
    template <typename REPLFUNC> API_EXPORT void replace_memory_blocks_withfunc(REPLFUNC &&rf)
    {
        hnnx::MemBlockReplBlockWrapper<std::remove_reference_t<REPLFUNC>> enumer(std::forward<REPLFUNC>(rf));
        this->enum_memory_blocks(enumer);
    }
    // this is passed a map<void*,void*> or any similar type with find() and end(),
    // and uses it to edit the blocks in the tensor.
    template <typename MAPTYPE> API_EXPORT void replace_memory_blocks_withmap(MAPTYPE const &map)
    {
        replace_memory_blocks_withfunc([&map](Tensor const *, void *oldid) {
            auto found_at = map.find(oldid);
            return (found_at != map.end()) ? found_at->second : oldid;
        });
    }

    API_EXPORT void serialize(hnnx::SerOpsInterface &sctx) const { sctx.tensor_serialize(this); }
    // The same tensor in the same layout, but with persistent storage.

    API_EXPORT std::unique_ptr<Tensor> persistent_clone(hnnx::Allocator *allocator, bool zoneb = false) const;
    // same thing, but does refcounts in 'zone B'
    API_EXPORT std::unique_ptr<Tensor> persistent_clone_Op(hnnx::Allocator *allocator) const
    {
        return persistent_clone(allocator, true);
    }
    // similar in effect to persistent_clone_Op, but can onlt be applied to
    // existing persistent tensors; and only copies the tensor, not the data.

    API_EXPORT std::unique_ptr<Tensor> shallow_clone_Op(hnnx::Allocator *allocator) const;

    // decref the ref counts of any contained blocks (all must be persistent)
    API_EXPORT void persistent_decref(hnnx::Allocator *allocator, bool zoneb = false) const;
    // same thing, but does refcounts in 'zone B'
    API_EXPORT void persistent_decref_Op(hnnx::Allocator *allocator) const
    {
        return persistent_decref(allocator, true);
    }

    // a 'duplicate' - same type,layout,dims; references the same
    // memory block(s) (where applicable).
    API_EXPORT std::unique_ptr<Tensor> duplicate_clone(hnnx::Allocator *allocator) const
    {
        return reallocate_clone(allocator, true);
    }
    // do a 'reallocate clone': the new tensor is the same type, layout, dims
    // but the block table is zeroed.
    // If dup=true, this is the same as duplicate_clone.
    API_EXPORT std::unique_ptr<Tensor> reallocate_clone(hnnx::Allocator *allocator, bool dup = false,
                                                        const OutputDef *od = nullptr) const;
    // 'compare' in the base class:
    //   - if the types are different, return -1 or 1 depending on that.
    //   - otherwise call protected virtual compare_sametype(), which can then use static_cast
    //     to downcast (and doesn't need to recurse back to the base).

    API_EXPORT int compare(const Tensor *rhs) const
    {
        Tensor const *const lhs = this;
        std::type_info const &lhs_type = typeid(*lhs);
        std::type_info const &rhs_type = typeid(*rhs);
        if (lhs_type == rhs_type) {
            return lhs->compare_sametype(rhs);
        } else {
            return lhs_type.before(rhs_type) ? -1 : 1;
        }
    }

    API_EXPORT virtual uint64_t get_checksum() const { return 0LL; }
    // these only work on specific types; in others, you inherit the base class implementation
    // which raises a runtime error. You can use tile_support() to find out if support exists
    API_EXPORT virtual void const *read_tile(unsigned flags, void *buffer, size_t b, int h, int w, int d) const;
    API_EXPORT virtual void write_tile(unsigned flags, void const *buffer, size_t b, int h, int w, int d);

    static constexpr unsigned tile_8bit = 1; // set when the data is 8 bit and the tensor supports tile access
    static constexpr unsigned tile_16bit = 2; // set when the data is 16 bit and the tensor supports tile access
    static constexpr unsigned tile_32bit = 4; // set when the data is 32 bit and the tensor supports tile access
    static constexpr unsigned tile_any = (1 + 2 + 4); // one of these bits is set if there is any support
    static constexpr unsigned tile_fast = 16; // set only when one of the above is set, and support is HVX accelerated.
    static constexpr unsigned tile_direct = 32; // set only when when 'fast' is set, and a direct mapping is possible

    API_EXPORT virtual unsigned tile_support_bits() const;
    API_EXPORT bool tile_support() const { return (tile_support_bits() & tile_any) != 0; }
    API_EXPORT bool tile_support_fast() const { return (tile_support_bits() & tile_fast) != 0; }
    API_EXPORT bool tile_support_direct() const { return (tile_support_bits() & tile_direct) != 0; }
    // this is currently a wrapper on tile_write, which inserts the 'write_strategy' flag, and suppresses broadcast
    // and copy flags. It may change to a separate virtual func.
    // (this is defined as an inline, in tile_extract.h).
    API_EXPORT void *write_tile_strategy(unsigned flags, void *buffer, size_t b, int h, int w, int d);

    API_EXPORT static uint32_t content_hash_data(void const *, size_t nbytes, bool is_float) noexcept;
    API_EXPORT static uint32_t content_hash_data_indirect(uint32_t inhash, void **blocks, unsigned nblocks,
                                                          size_t blockbytes, bool is_float) noexcept;

    API_EXPORT static uint32_t build_hash(size_t const *dims, int n, uint32_t previous) noexcept;

    struct API_EXPORT tensor_blockinfo {
        void **blkptrs; // pointer to block table (nullptr if no blocks)
        // shapepp is a pointer to the shape pointer (where applicable; otherwise null). If a clone
        // is done, it points to the field in the cloned tensor.
        hnnx::ShapeFlags const *const *shapepp;
        // This is a pointer to the tensor's 'interface' pointer, if it has one; otherwise nullptr.
        // If a clone is being done, it points to the field in the cloned tensor.
        Interface const *const *interfacepp;
        size_t nblocks; // number of blocks
        size_t blocksize; // size of block, in bytes
        DType dtype;
        MemoryClass mclass;
        bool is_indirect; // indicates that the layout is indirect.
        bool is_chunked; // indicates that the layout is chunked;
        bool is_variable_block; // indicates variable block size
        void setup(DType dt = DType::UNKNOWN, MemoryClass mc = MemoryClass::Default)
        {
            blkptrs = nullptr;
            shapepp = nullptr;
            interfacepp = nullptr;
            nblocks = 0;
            blocksize = 0;
            dtype = dt;
            mclass = mc;
            is_indirect = false;
            is_chunked = false;
            is_variable_block = false;
        }
    };
    API_EXPORT void get_tensor_blockinfo(tensor_blockinfo *infop) const { clone_util(nullptr, nullptr, infop); }

    // deserialize a single block pointer for a contiguous tensor.
    API_EXPORT static auto deserialize_block_pointer(hnnx::Deserz &dctx) -> void *;
    // deserialize an indirect blocktable, given ref to pointer.
    // 'nblocks' may be 1, instead of actual len, if we are not decoding 'classic' format.
    template <typename T> // T = storage_type
    static void deserialize_blocktable(hnnx::Deserz &dctx, T **&blockptr, unsigned const nblocks)
    {
        deserialize_blocktable_generic(dctx, (void ***)&blockptr, nblocks);
    }
    API_EXPORT static void deserialize_blocktable_generic(hnnx::Deserz &dctx, void ***blockp_loc, unsigned nblocks);

  protected:
    API_EXPORT virtual void allocate_func(hnnx::Allocator &allocator, unsigned options) = 0;
    API_EXPORT virtual int compare_sametype(const Tensor *rhs) const = 0;

    // clone_util is an overburdened virtual function, which performs duplicate_clone almost directly,
    // and provides other info by which the other clone methods, and the decref methods,
    // can all be done generically in the base class.
    //
    // - If tensp != null, it will create a duplicate_clone, and store it at *tensp;
    // - If infop != null, it will fill in *infop with the tensor info.
    // If *both* are not null, then infop->blkptrs will point to the block table in the
    //  original tensor, and the return value is the block pointer in the new tensor.
    //  Otherwise the return value is null (and it will be null in any case, if the tensor
    //  has no blocks).
    //
    //
    // Note: allocator may be null if tensp is null.

    API_EXPORT virtual void **clone_util(hnnx::Allocator *allocator, std::unique_ptr<Tensor> *tensp,
                                         tensor_blockinfo *infop, const OutputDef *od = nullptr) const = 0;
};

// A FakeTensor is intended as an intermediate base for special subclasses which
// need to be based on Tensor but don't need to support most of the interface.
// subclassing should be done in .cc files or private headers where possible.
//
// All of the abstract 'virtual=0' methods (other than get_dtype) are overridden here;
// many (those shown as protected) will all throw exceptions if called; the others do
// null things as shown.
// So when you subclass, just override whatever ones you need and leave the rest.
//
// In particular, get_dtype() returns DType::None.
//
class FakeTensor : public Tensor {
  public:
    explicit FakeTensor(const Op *producer_in) : Tensor(producer_in) {}
    API_EXPORT explicit FakeTensor(hnnx::Deserz &);

  protected:
    // all will throw exception if called
    API_EXPORT virtual void *element_addr(size_t rank, SIdx const coords_in[],
                                          hnnx::InterfaceRef *iref = nullptr) const noexcept override;
    API_EXPORT virtual hnnx::InterfaceRef interface() const noexcept override;
    API_EXPORT virtual void allocate_func(hnnx::Allocator &allocator, unsigned options) override;
    API_EXPORT virtual void *raw_data() noexcept override;
    API_EXPORT virtual size_t total_storage_elements() const override;
    API_EXPORT virtual size_t total_storage_bytes() const override;
    API_EXPORT virtual void **clone_util(hnnx::Allocator *allocator, std::unique_ptr<Tensor> *tensp,
                                         Tensor::tensor_blockinfo *infop, const OutputDef *od = nullptr) const override;
    API_EXPORT virtual int compare_sametype(const Tensor *rhs) const override;

  public:
    // defined as shown
    API_EXPORT virtual size_t rank() const noexcept override; //->0
    API_EXPORT virtual size_t dim(size_t index) const noexcept override; //->0
    API_EXPORT virtual std::pair<size_t const *, size_t> get_dims() const noexcept override; //->{null,0}
    API_EXPORT virtual bool set_dims(const size_t dims[]) override; // -> false
    API_EXPORT virtual bool set_dims(const Tensor &prototype) override; // ->false
    API_EXPORT virtual void enum_memory_blocks(hnnx::MemBlockEnumerator &) const override; // nothing

    API_EXPORT virtual DTypeScaleOff get_dtype_intfc() const noexcept override; // { return DTypeScaleOff(None); }
};
/**
 * @class RankedTensor
 *
 * @brief Almost as abstract as Tensor, but we template the Rank.
 * This allows us to have compile-time checking of the operator(), as well as a public Rank that is static constexpr
 * so it doesn't take a lot of space or performance...
 * The other benefit here is that we can specify RankedTensor<4> to have a fairly generic tensor,
 * but enforce the number of dimensions of the tensor.
 */

template <unsigned TRank> class RankedTensor : public Tensor {
  public:
    struct traits {
        static constexpr unsigned Rank = TRank;
    };

    API_EXPORT explicit RankedTensor(const Op *producer_in) : Tensor(producer_in) {}
    API_EXPORT explicit RankedTensor(hnnx::Deserz &dctx) : Tensor(dctx) {}
    API_EXPORT RankedTensor(const RankedTensor &old, hnnx::Allocator *allocator, clone_mode cmode)
        : Tensor(old, allocator, cmode)
    {
    }
    static constexpr auto Rank = TRank;
    API_EXPORT virtual size_t rank() const noexcept override final { return Rank; }
    template <typename... ind_types> API_EXPORT GenericAccessorRO operator()(ind_types... inds) const
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const SIdx indarr[] = {static_cast<SIdx>(inds)...};
        hnnx::InterfaceRef intfc = hnnx::InterfaceRef::make_null();
        void *const ptr = element_addr(Rank, indarr, &intfc);
        return GenericAccessorRO(ptr, intfc);
    }
    template <typename... ind_types> API_EXPORT GenericAccessor operator()(ind_types... inds)
    {
        static_assert(Rank == (sizeof...(ind_types)), "# of coords must match Rank");
        const SIdx indarr[] = {static_cast<SIdx>(inds)...};
        hnnx::InterfaceRef intfc = hnnx::InterfaceRef::make_null();
        void *const ptr = element_addr(Rank, indarr, &intfc);
        return GenericAccessor(ptr, intfc);
    }
};

/**
 * @class TensorShape
 *
 * @brief This is a tensor that just has a shape, no memory or type or anything.
 * This needs to be non-abstract
 *
 * EJP: FIXME: should we really use this, or just use Const? Or special like-Const op?
 * EJP: FIXME: Performance is not so criticial here, we need it to respect the interface
 * but we really want to make this convenient representation and be formable from an OutputDef
 *
 * TensorShape should already be canonized by the nature of being a const op.
 * We might be able to share TensorShapes shapes and Tensor shapes, but it seems unnecessary.
 */

template <unsigned TRank> class TensorShape : public RankedTensor<TRank> {
    using Parent = RankedTensor<TRank>;

  protected:
    API_EXPORT static constexpr NullInterface null_interface{};
    // These functions are not really part of the interface, but we need them to implement operator()
    API_EXPORT virtual void *element_addr(size_t /* rank */, SIdx const /* coords_in */[],
                                          hnnx::InterfaceRef *const iref = nullptr) const noexcept override final
    {
        if (iref) *iref = NullInterface::get_refobj();
        return nullptr;
    }
    API_EXPORT virtual hnnx::InterfaceRef interface() const noexcept override final
    {
        return NullInterface::get_refobj();
    }

  public:
    API_EXPORT const char *true_name() const override { return type_name<TensorShape<TRank>>(); }

    using Parent::Rank;
    struct traits {
        using element_type = void;
        using storage_type = void;
        static constexpr DType dtype = DType::UNKNOWN;
        static constexpr unsigned rank = TRank;
    };

    //using Shape_t = Shape<Rank>;
    //const Shape_t *shape;
    const std::array<size_t, Rank> shape;
    API_EXPORT virtual size_t dim(size_t index) const noexcept override final { return shape[index]; }
    API_EXPORT const std::array<size_t, Rank> &dims() const { return shape; }
    template <typename... T> API_EXPORT const std::array<size_t, sizeof...(T)> dims(T... indices) const
    {
        return Tensor::dims_extractor(get_dims(), indices...);
    }
    API_EXPORT virtual std::pair<size_t const *, size_t> get_dims() const noexcept override
    {
        return std::pair<size_t const *, size_t>(&shape[0], Rank);
    }
    API_EXPORT virtual DTypeScaleOff get_dtype_intfc() const noexcept override
    {
        return NullInterface::get_dtype_scaleoff();
    }

    API_EXPORT virtual uint32_t get_tensor_format_code() const noexcept override
    {
        return Tensor::formatcode_for_shape<Rank>();
    }

    API_EXPORT virtual uint32_t get_tensor_info() const noexcept override
    {
        return hnnx::SerOpsInterface::tensMODE_shape << Tensor::tformat_tmode_shift;
    }

    // Optional, but maybe helpful?
    API_EXPORT virtual bool set_dims(const size_t /* dims */[]) override
    {
        // static_assert("Shapes are immutable");
        return true;
    } // immutable
    API_EXPORT virtual bool set_dims(const Tensor & /* prototype */) override
    {
        // static_assert("Shapes are immutable");
        return true;
    } // immutable

    // EJP: FIXME: temporary functions
    API_EXPORT virtual void *raw_data() noexcept override
    {
        return nullptr;
    } // Allocate storage ourselves instead of fancy memory allocator
    API_EXPORT virtual size_t total_storage_elements() const override { return 0; }
    API_EXPORT virtual size_t total_storage_bytes() const override { return 0; }
    TensorShape(const Op *producer_in, const OutputDef &def, Graph & /* graph_in */)
        : Parent(producer_in), shape(hnnx::ptr_to_stdarray<Rank, size_t>(&def.max_sizes[0]))
    {
    }
    explicit TensorShape(hnnx::Deserz &dctx) : Parent(dctx), shape(dctx.deserialize_uint32_array_sizet<Rank>()) {}

    TensorShape(const TensorShape &old, hnnx::Allocator *allocator, Tensor::clone_mode cmode)
        : Parent(old, allocator, cmode), shape(old.shape)
    {
    }

    API_EXPORT virtual void enum_memory_blocks(hnnx::MemBlockEnumerator &) const override { return; }

    API_EXPORT virtual void allocate_func(hnnx::Allocator & /* allocator */, unsigned /* options */) override final {}

    API_EXPORT virtual void **clone_util(hnnx::Allocator *allocator, std::unique_ptr<Tensor> *tensp,
                                         Tensor::tensor_blockinfo *infop,
                                         const OutputDef * /* od */ = nullptr) const override
    {
        if (tensp) *tensp = std::make_unique<TensorShape>(*this, allocator, Tensor::clone_mode::duplicate);
        if (infop) infop->setup();
        return nullptr;
    }

  protected:
    API_EXPORT virtual int compare_sametype(const Tensor *rhs_in) const override
    {
        auto *rhs = static_cast<const TensorShape *>(rhs_in);
        for (int i = 0; i < Rank; i++) {
            int const dimdiff = this->shape[i] - rhs->shape[i];
            if (dimdiff != 0) return dimdiff;
        }
        return 0;
    }
    API_EXPORT virtual uint32_t find_content_hash() const noexcept override
    {
        return Tensor::build_hash(&shape[0], Rank, 0x113014);
    }
    API_EXPORT static const char *code_to_type_name() { return TensorTypeStruct<TensorShape<TRank>>::name; }
};

/*
 * I think we should have a Scalar Constant
 * * Immutable shape
 * * Rank 0
 * * Coords ignored
 * * Dim(x) == 0
 * * Templated type / interface?
 */

//
// Tensor Scalar depending on DType

template <DType DT> class TensorSclrDT : public Tensor {
  protected:
    using T = typename dtype_traits<DT>::element_type;
    using Interface_t = std::conditional_t<dtype_traits<DT>::is_quant, ScaleOffsetInterface<T>, PlainInterface<T>>;
    using Accessor_t = typename Interface_t::Accessor;
    using Const_Accessor_t = typename Interface_t::AccessorRO;
    Interface_t interface_inst;

  public:
    API_EXPORT virtual hnnx::InterfaceRef interface() const noexcept override final
    {
        return interface_inst.get_refobj();
    }
    API_EXPORT float interface_scale() const { return interface_inst.get_scale(); }
    API_EXPORT float interface_scale_recip() const { return interface_inst.get_scale_recip(); }
    API_EXPORT int32_t interface_offset() const { return interface_inst.get_offset(); }

    API_EXPORT virtual uint32_t get_tensor_format_code() const noexcept override
    {
        return Tensor::formatcode_for_scalar<Interface_t>();
    }

    API_EXPORT virtual uint32_t get_tensor_info() const noexcept override
    {
        return hnnx::SerOpsInterface::tensMODE_scalar << tformat_tmode_shift;
    }

    // EJP: FIXME: this should just be the value, but then GenericAccessor constructor
    // complains about const value going to a const Accessor where the constructor in
    // const Accessor is written to have a normal void pointer input... sigh.
    T value;

  protected:
    // These functions are not really part of the interface, but we need them to implement operator()
    API_EXPORT virtual void *element_addr(size_t /* rank */, SIdx const /* coords_in */[],
                                          hnnx::InterfaceRef *const iref = nullptr) const noexcept override final
    {
        if (iref) *iref = interface_inst.get_refobj();
        return (void *)&value;
    }

  public:
    API_EXPORT const char *true_name() const override { return type_name<TensorSclrDT<DT>>(); }

    struct traits {
        using element_type = T;
        using storage_type = typename dtype_traits<DT>::storage_type;
        using interface_type = Interface_t;
        static constexpr DType dtype = DT;
        static constexpr unsigned rank = 0;
    };

    API_EXPORT virtual size_t rank() const noexcept override { return 0; } // What's the rank of this tensor?
    API_EXPORT virtual size_t dim(size_t /* index */) const noexcept override { return 1; }
    API_EXPORT virtual std::pair<size_t const *, size_t> get_dims() const noexcept override
    {
        return std::pair<size_t const *, size_t>(nullptr, 0);
    }
    static constexpr DType dtype = dtype_of_type<Interface_t>();
    API_EXPORT virtual DTypeScaleOff get_dtype_intfc() const noexcept override
    {
        return interface_inst.get_dtype_scaleoff();
    }

    // Optional, but maybe helpful?
    API_EXPORT virtual bool set_dims(const size_t /* dims */[]) override
    {
        // static_assert("Scalar dims are immutable");
        return true;
    } // immutable
    API_EXPORT virtual bool set_dims(const Tensor & /* prototype */) override
    {
        // static_assert("Scalar dims are immutable");
        return true;
    } // immutable
    // EJP: FIXME: temporary functions
    API_EXPORT virtual void *raw_data() noexcept override final
    {
        return &value;
    } // Allocate storage ourselves instead of fancy memory allocator
    API_EXPORT const void *raw_data() const noexcept
    {
        return &value;
    } // Allocate storage ourselves instead of fancy memory allocator
    API_EXPORT virtual size_t total_storage_elements() const override { return 0; }
    API_EXPORT virtual size_t total_storage_bytes() const override { return 0; }
    API_EXPORT virtual void allocate_func(hnnx::Allocator & /* allocator */, unsigned /* options */) override final {}
    API_EXPORT TensorSclrDT(const Op *producer_in, T value_in) : Tensor(producer_in), value(value_in)
    {
        static_assert(!dtype_traits<DT>::is_quant, "FIXME: need different constructor");
    }
    API_EXPORT explicit TensorSclrDT(hnnx::Deserz &dctx)
        : Tensor(dctx), interface_inst(dctx), value(dctx.deserialize_type<T>())
    {
    }
    API_EXPORT TensorSclrDT(const TensorSclrDT &old, hnnx::Allocator *allocator, clone_mode cmode)
        : Tensor(old, allocator, cmode), interface_inst(old.interface_inst), value(old.value)
    {
    }

    template <typename... ind_types> API_EXPORT const Const_Accessor_t operator()(ind_types... /* inds */) const
    {
        return Const_Accessor_t((void *)&value, &interface_inst);
    }
    template <typename... ind_types> API_EXPORT Accessor_t operator()(ind_types... /* inds */)
    {
        return Accessor_t(&value, &interface_inst);
    }
    API_EXPORT virtual void enum_memory_blocks(hnnx::MemBlockEnumerator &) const override { return; }

    API_EXPORT virtual void **clone_util(hnnx::Allocator *allocator, std::unique_ptr<Tensor> *tensp,
                                         Tensor::tensor_blockinfo *infop,
                                         const OutputDef * /* od */ = nullptr) const override
    {
        if (tensp) *tensp = std::make_unique<TensorSclrDT>(*this, allocator, clone_mode::duplicate);
        if (infop) infop->setup(DT);
        return nullptr;
    }

  protected:
    API_EXPORT virtual int compare_sametype(const Tensor *rhs_in) const override
    {
        // FIXME @@ if Interface_t is quantized, we should compare quantization too.
        auto *rhs = static_cast<const TensorSclrDT *>(rhs_in);
        if (rhs == nullptr) {
            throw std::runtime_error("rhs is nullptr");
        }
        if (this->value < rhs->value) return -1;
        if (this->value == rhs->value) return 0;
        return 1;
    }
    API_EXPORT virtual uint32_t find_content_hash() const noexcept override
    {
        uint32_t const h = interface().interface_hash() ^ mulu32_modular(unsigned(DT), 0x107301);
        return mulu32_modular(h, 0x104301) ^ content_hash_data(&this->value, sizeof(T), dtype_traits<DT>::is_float);
    }
    API_EXPORT static const char *code_to_type_name() { return TensorTypeStruct<TensorSclrDT<DT>>::name; }
};
// Tensor Scalar depending on type (assuming PlainInterface)

template <typename T> using TensorScalar = TensorSclrDT<dtype_of_type<PlainInterface<T>>()>;

size_t get_tensor_size(const OutputDef &outdef);

POP_VISIBILITY()
#endif // HEXNN_TENSOR_BASE_H
