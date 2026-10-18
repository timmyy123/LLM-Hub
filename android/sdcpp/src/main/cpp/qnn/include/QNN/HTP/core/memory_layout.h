// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef HEXNN_MEMORY_LAYOUT_H
#define HEXNN_MEMORY_LAYOUT_H 1

#include <array>
#include <cstddef>
#include <utility>

/*
 * Rewrite memory layout
 *
 * Use more recursion for less complexity at each level
 *
 * Separate Offset and Index for use by non-contiguous tensor representations
 */

namespace hnnx {

// is_power_of_two: check for some number of zeros, followed by 1, followed by some number of zeros.
// FIXME: maybe should use bitset?
static inline constexpr bool is_power_of_two(unsigned long in)
{
    return (in > 0) && ((in & (in - 1)) == 0);
}

/*
 * Making a constexpr std::array is kind of tough if a lot of the std::array
 * member functions are not constexpr, which is true if you have pre-c++17 header
 * files...
 */
// LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
template <typename T, size_t Rank, size_t... I>
static inline constexpr std::array<T, Rank> make_stdarray_helper(const T val, std::index_sequence<I...>)
{
    std::array<T, Rank> out = {((void)I, val)...};
    return out;
}

template <typename T, size_t Rank> static inline constexpr std::array<T, Rank> make_stdarray(const T val)
{
    return make_stdarray_helper<T, Rank>(val, std::make_index_sequence<Rank>{});
}
// LCOV_EXCL_STOP

} // namespace hnnx

/*
 * We use std::get in a lot of places below because operator[] is not constexpr
 * if you have pre-C++17 system header files
 */

/*
 * The base template... do not use (is not actually defined anywhere)
 */
template <size_t... Stuff> struct IChunkedMemoryLayout;

/*
 * The smallest Chunk is just 1 element
 */
template <size_t RankVal, size_t IndirRanks> struct IChunkedMemoryLayout<RankVal, IndirRanks> {
    static constexpr size_t Rank = RankVal;
    static constexpr size_t indirect_ranks = IndirRanks;
    static constexpr std::array<size_t, Rank> ChunkSizes = hnnx::make_stdarray<size_t, Rank>(1);
    static constexpr size_t chunk_total = 1;
    static constexpr bool is_valid_chunk = true;
    static constexpr unsigned inner_dim = 99; // invalid value

    static inline constexpr size_t chunk_offset(const std::array<size_t, Rank> & /*padded_coords*/,
                                                const std::array<size_t, Rank> & /*dims_total*/)
    {
        return 0;
    }
    static inline constexpr size_t linear_offset(const std::array<size_t, Rank> & /*padded_coords*/,
                                                 const std::array<size_t, Rank> & /*dims_total*/)
    {
        return 0;
    }
    static inline constexpr size_t chunk_index(const std::array<size_t, Rank> & /*padded_coords*/,
                                               const std::array<size_t, Rank> & /*dims_total*/, size_t offset = 0)
    {
        return offset;
    }
    static inline constexpr std::array<size_t, Rank> pad_dims(const std::array<size_t, Rank> dims_in)
    {
        return dims_in;
    }
};

/*
 * This should boil down to nothing... no non-constexpr storage, no non-constexpr functions.
 */

/*
 * Define a memory layout that includes an indirection table
 * The memory layout has a certain minimum size in each direction.  This is the "Chunk".
 * Additionally, one ore more chunks (all the chunks in some final dimensions) may be contiguous.
 * These contiguous elements form a "block".
 * The indirection table contains pointers for each "block"
 * The "Chunk" size is constant for a type
 * The "Block" size depends on the max dims of the tensor.
 */

template <size_t RankVal, size_t IndirRanks, size_t Dim, size_t ChunkSize, size_t... Rest>
struct IChunkedMemoryLayout<RankVal, IndirRanks, Dim, ChunkSize, Rest...> {
    using Smaller = IChunkedMemoryLayout<RankVal, IndirRanks, Rest...>;
    static constexpr size_t Rank = RankVal;
    static constexpr size_t indirect_ranks = IndirRanks;
    static_assert(Dim < RankVal);
    static_assert(RankVal >= IndirRanks);
    //static_assert(ChunkSize > 0);
    static_assert((ChunkSize == 0) || hnnx::is_power_of_two(ChunkSize));
    static_assert((ChunkSize == 0) || Smaller::is_valid_chunk);
    static constexpr bool is_valid_chunk = ((ChunkSize > 0) && (Smaller::is_valid_chunk));
    // inner_dim is the fastest_moving 'outer dim' - i.e. the last one to appear with a 0 chunksize.
    static constexpr unsigned inner_dim = ((ChunkSize == 0) && Smaller::is_valid_chunk) ? Dim : Smaller::inner_dim;
    static constexpr std::array<size_t, Rank> embiggen_chunksize(const std::array<size_t, Rank> smaller_chunksize)
    {
        std::array<size_t, Rank> out = smaller_chunksize;
        if (ChunkSize) std::get<Dim>(out) *= ChunkSize;
        return out;
    }
    static constexpr std::array<size_t, Rank> ChunkSizes = embiggen_chunksize(Smaller::ChunkSizes);
    static constexpr size_t chunk_total = ChunkSize ? Smaller::chunk_total * ChunkSize : Smaller::chunk_total;
    /* Where in the chunk is this element? */
    /*
	 *  FIXME sooner than later: recommendation to return std::pair or similar of chunk_index and chunk_offset
	 * Can keep compatibility easily enough with a single wrapper.
	 */
    static inline constexpr size_t chunk_offset(const std::array<size_t, Rank> &padded_coords,
                                                const std::array<size_t, Rank> &dims_total, //
                                                size_t block_off = 0)
    {
        if constexpr (ChunkSize > 0) {
            const size_t smaller_offset = Smaller::chunk_offset(padded_coords, dims_total);
            const size_t dim_coord = padded_coords[Dim];
            const size_t smaller_idx = dim_coord / std::get<Dim>(Smaller::ChunkSizes);
            const size_t thischunk_smaller_idx = smaller_idx % ChunkSize;
            const size_t smaller_chunk_total = Smaller::chunk_total;
            return block_off * chunk_total + thischunk_smaller_idx * smaller_chunk_total + smaller_offset;
        } else if constexpr (Dim < IndirRanks) {
            // Is an indirect dimension, doesn't participate in block offset
            size_t const chunk_off = Smaller::chunk_offset(padded_coords, dims_total, block_off);
            return chunk_off;
        } else {
            // Calculate our intra-block / inter-chunk offset in terms of chunks?
            block_off *= std::get<Dim>(dims_total) / std::get<Dim>(ChunkSizes);
            block_off += std::get<Dim>(padded_coords) / std::get<Dim>(ChunkSizes);
            size_t const chunk_off = Smaller::chunk_offset(padded_coords, dims_total, block_off);
            return chunk_off;
        }
    }
    /* FIXME later: we're going to assume last to first dimension ordering */
    /*
     * This returns the index in the block table, so it should probably be called block_index
     * now that we are going to distinguish "blocks" from "chunks"
     */
    static inline constexpr size_t chunk_index(const std::array<size_t, Rank> &padded_coords,
                                               const std::array<size_t, Rank> &dims_total, size_t offset = 0)
    {
        if constexpr (is_valid_chunk) {
            // This is already at the fixed size, so no further index adjustment into the block table
            return offset;
        } else if constexpr (Dim >= IndirRanks) {
            // This Dim does not participate in the indirection table, skip it
            return Smaller::chunk_index(padded_coords, dims_total, offset);
        } else {
            // Figure out the index in the block table
            offset *= std::get<Dim>(dims_total) / std::get<Dim>(ChunkSizes);
            offset += std::get<Dim>(padded_coords) / std::get<Dim>(ChunkSizes);
            size_t const chunk_idx = Smaller::chunk_index(padded_coords, dims_total, offset);
            return chunk_idx;
        }
    }
    static inline constexpr size_t linear_offset(const std::array<size_t, Rank> &padded_coords,
                                                 const std::array<size_t, Rank> &dims_total)
    {
        const size_t offset = chunk_offset(padded_coords, dims_total);
        const size_t index = chunk_index(padded_coords, dims_total);
        return index * chunk_total + offset;
    }
    static inline std::array<size_t, Rank> pad(const std::array<size_t, Rank> dims_in)
    {
        std::array<size_t, Rank> newdims;
        for (int i = 0; i < Rank; i++) {
            auto dim_chunk_size = ChunkSizes[i];
            newdims[i] = ((dims_in[i] + (dim_chunk_size - 1)) & (~(dim_chunk_size - 1)));
        }
        return newdims;
    }
    // Number of blocks in the tensor of the given size, the size of the indirection table
    static inline size_t num_blocks(const std::array<size_t, Rank> max_dims)
    {
        size_t blocks = 1;
        for (int i = 0; i < IndirRanks; i++) {
            auto dim_chunk_size = ChunkSizes[i];
            blocks *= max_dims[i] / dim_chunk_size;
        }
        return blocks;
    }
    // Number of elements in the block across all dimensions for the given size
    // This should be the total maximum size
    static inline size_t block_total(const std::array<size_t, Rank> max_dims)
    {
        size_t blocks = 1;
        for (int i = 0; i < Rank; i++) {
            auto const dim_chunk_size = ChunkSizes[i];
            if (i < IndirRanks)
                blocks *= dim_chunk_size;
            else
                blocks *= max_dims[i];
        }
        return blocks;
    }
};

// ChunkedMemoryLayout is just IChunkedMemoryLayout with IDirRanks = Rank
template <size_t Rank, size_t... Etc> //
using ChunkedMemoryLayout = IChunkedMemoryLayout<Rank, Rank, Etc...>;

// Simplified case,
// E.g. FlatMemoryLayout<4>
//  equiv to IChunkedMemoryLayout<4, 0,0, 1,0, 2,0, 3,0>

template <size_t RankVal> struct FlatMemoryLayout {
    static constexpr size_t Rank = RankVal;
    static constexpr size_t indirect_ranks = RankVal; // to be consistent; only applies when chunk_total > 1.
    static constexpr std::array<size_t, Rank> ChunkSizes = hnnx::make_stdarray<size_t, Rank>(1);
    static constexpr size_t chunk_total = 1;
    static constexpr unsigned inner_dim = Rank - 1;
    static inline constexpr size_t chunk_offset(const std::array<size_t, Rank> & /* padded_coords */,
                                                const std::array<size_t, Rank> & /* dims_total */)
    {
        return 0;
    }
    static inline constexpr size_t chunk_index(const std::array<size_t, Rank> &padded_coords,
                                               const std::array<size_t, Rank> &dims_total)
    {
        size_t offset = padded_coords[0];
        for (size_t i = 1; i < Rank; i++) {
            offset = offset * dims_total[i] + padded_coords[i];
        }
        return offset;
    }
    static inline constexpr size_t linear_offset(const std::array<size_t, Rank> &padded_coords,
                                                 const std::array<size_t, Rank> &dims_total)
    {
        return chunk_index(padded_coords, dims_total);
    }
    static inline constexpr std::array<size_t, Rank> pad(const std::array<size_t, Rank> dims_in) { return dims_in; }

    static inline constexpr size_t num_blocks(const std::array<size_t, Rank> max_dims)
    {
        size_t blocks = max_dims[0];
        for (int i = 1; i < Rank; i++) {
            blocks *= max_dims[i];
        }
        return blocks;
    }
};
using R4FlatMemoryLayout = FlatMemoryLayout<4>; //NHWC
using R5FlatMemoryLayout = FlatMemoryLayout<5>; //NHWDC
using R6FlatMemoryLayout = FlatMemoryLayout<6>;

/////////////
// SingularMemoryLayout is a layout of given rank which contains
// a single value (the dimensions are ignored).

template <size_t RankVal> struct SingularMemoryLayout {
    static constexpr size_t Rank = RankVal;
    static constexpr size_t indirect_ranks = RankVal;
    static constexpr std::array<size_t, Rank> ChunkSizes = hnnx::make_stdarray<size_t, Rank>(1);
    static constexpr size_t chunk_total = 1;
    static constexpr unsigned inner_dim = Rank - 1;
    static inline constexpr size_t chunk_offset(const std::array<size_t, Rank> & /*padded_coords*/,
                                                const std::array<size_t, Rank> & /*dims_total*/)
    {
        return 0;
    }
    static inline constexpr size_t chunk_index(const std::array<size_t, Rank> & /*padded_coords*/,
                                               const std::array<size_t, Rank> & /*dims_total*/)
    {
        return 0;
    }
    static inline constexpr size_t linear_offset(const std::array<size_t, Rank> & /*padded_coords*/,
                                                 const std::array<size_t, Rank> & /*dims_total*/)
    {
        return 0;
    }
    static inline constexpr std::array<size_t, Rank> pad(const std::array<size_t, Rank> dims_in) { return dims_in; }
    static inline constexpr size_t num_blocks(const std::array<size_t, Rank> /*max_dims*/) { return 1; }
};
using R4SingularMemoryLayout = SingularMemoryLayout<4>;

// Note : I think we could use typedefs instead of subclass for all of these;
// a long time ago, when the layout information was mangled into tensor types,
// we changed from typedefs to trivial subclassing to shorten those names, but
// that's no longer an issue.

class R4NCHWMemoryLayout : public ChunkedMemoryLayout<4, 0, 0, 3, 0, 2, 0, 1, 0> {}; // NCHW
class R4Depth32MemoryLayout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 3, 0, 2, 0, 2, 4, 3, 32> {};

// Croutons for HMX, YYYXXXDDDDD chunks
class R4CroutonLayout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 1, 8, 2, 8, 3, 32> {};
// Croutons for HMX, YXXXXXDDDDD chunks (wide aspect ratio)
class R4WideCroutonLayout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 1, 2, 2, 32, 3, 32> {};

// Croutons for HMX, YYYXDDDDDXX chunks
class R4Crouton4x1Layout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 1, 8, 2, 2, 3, 32, 2, 4> {};

// Croutons for HMX, YYXXDDDDDYX chunks
class R4Crouton2x2Layout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 1, 4, 2, 4, 3, 32, 1, 2, 2, 2> {};

// Croutons for HMX, YYXXDDDDDYX chunks (wide aspect ratio)
class R4WideCrouton2x2Layout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 2, 16, 3, 32, 1, 2, 2, 2> {};

// Croutons2 for HMX, 8x4x32 chunks where the data is 16b
class R4Crouton2Layout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 1, 8, 2, 2, 3, 32, 2, 2> {};

// Note, we want the lists of numbers to be paired properly, some of the commas won't have spaces after.
// clang-format off
// AR4 8*32==256 deep 1H
using R4DeepAR4_16bLayout = IChunkedMemoryLayout<4,3, 0,0, 1,0, 2,0, 3,0, 2,2, 3,32, 2,2>;
// AR8 chunk format 8x32 == 256 elements / 512B
using R4DeepAR8_16bLayout = IChunkedMemoryLayout<4,3, 0,0, 1,0, 2,0, 3,0, 2,4, 3,32, 2,2>;

// clang-format on

class R4Crouton4Layout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 1, 8, 2, 2, 3, 32> {};

class R4WideCrouton4Layout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 2, 0, 3, 0, 1, 2, 2, 8, 3, 32> {};

class R4Weights8x4Layout : public ChunkedMemoryLayout<4, 0, 0, 1, 0, 3, 0, 2, 0, 0, 8, 1, 4, 2, 16, 3, 32, 2, 2> {};

class R5CroutonLayout : public ChunkedMemoryLayout<5, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 2, 8, 3, 8, 4, 32> {};

//typedef ChunkedMemoryLayout<4, 0,0, 3,0, 2,0, 1,0> R4NCHWMemoryLayout; // NCHW
//typedef ChunkedMemoryLayout<4, 0,0, 1,0, 3,0, 2,0, 2,4, 3,32> R4Depth32MemoryLayout;
//typedef ChunkedMemoryLayout<4, 0,0, 1,0, 2,0, 3,0, 1,8, 2,8, 3,32> R4CroutonLayout;		// Croutons for HMX, 8x8x32 chunks

//typedef ChunkedMemoryLayout<3, 2,0, 1,0, 0,0> R3FlatMemoryLayout; // HWC
//typedef ChunkedMemoryLayout<2, 1,0, 0,0> RowMajorMatrixLayout; // 2D
//typedef ChunkedMemoryLayout<2, 0,0, 1,0> ColMajorMatrixLayout; // 2D
//typedef ChunkedMemoryLayout<1, 1,0> VectorLayout; // 1D

#endif
