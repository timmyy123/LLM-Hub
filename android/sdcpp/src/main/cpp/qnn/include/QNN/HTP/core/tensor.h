// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef HEXNN_TENSOR_H
#define HEXNN_TENSOR_H 1
/*
 * This file is trying to figure out a nice Tensor class, which allows for access
 * to a data structure with potentially unknown underlying data types and layout.
 *
 * What is a Tensor? It's a multidimensional array of data
 * It has a "Rank": the number of dimensions.
 * It has a shape.
 * It contains data values.
 * There is a mechanism to access the data values.
 *
 * From an abstract perspective, that's about all we should have to know about a tensor.
 * However, to form a concrete tensor, it should also be observed that:
 *
 * The data values have some type.  They may be encoded/decoded with some extra information.
 * The data is laid out in some fashion.  It may be a contiguous block, it may have the data
 *   shuffled in some way, it may have a table of pointers to fixed-size blocks...
 * There might be extra padding values around the "true" data
 *
 * To facilitate the most abstract interfaces being available while also being
 * able to specify concrete tensors and have the compiler understand the mechanics
 * of the concrete tensor, we probably need to have:
 * * Abstract tensor as a base class that provides a generic interface always, using runtime polymorphism
 * * Subclasses that provide more concrete tensor representations, finalizing aspects of the tensor,
 * * Concrete classes that provide the compiler with full visibility in the details of the tensor
 *
 * Because values may be encoded/decoded from their internal representation, in the most abstract
 * representation we can not just return a data element.  Instead, we return an accessor object.
 * The accessor object works like an rvalue or lvalue, but is able to decode (rvalue) or encode (lvalue)
 * the data as appropriate.
 * (At least, that's how I think it should work...)
 *
 * We'd like to use the operator() to allow us to have multidimentional-array-indices-like interface,
 * much in the same way we have in Eigen.  So for a 4D tensor, with indices batchidx,row,col,channel,
 * we should be able to say out_tensor(batchidx,row,col,channel) = in_tensor(batchidx,row,col,channel)
 *
 * Although we might consider a variety of different types for tensor internal values, including int32,
 * I propose to use "float" as the defualt interface type.  It should work well for many integers, and
 * is the appropriate interface for real data whether quantized or not.  A reasonable alternative would
 * be double, but double is quite a bit more expensive on Hexagon.  Of course, other methods of accessing
 * the data could be made for different types.
 *
 * Having extremely abstract tensors allows us to have extremely generic
 * functions, but having easily available less abstract tensors should allow us
 * to easily specify constraints for ops that are more optimal or that are
 * demanding certain parameters for their inputs.  For example, if we always want
 * our convolutional op to have a 4D input tensor, we might specify that it is
 * a RankedTensor<4> instead of a Tensor, indicating that the op can only use
 * a tensor with rank 4.
 *
 * Tensors are very fundamental to how we are going to work on things, so it's incredibly important to
 * be on our best behavior here.
 *
 */

/*
 * EJP: FIXME:
 * * The helper classes, Accessors and Interfaces and such, should be in a sub-namespace for cleanliness.
 * * Make the Abstract/Base/Generic naming consistent.  I like "Generic" at the moment.
 */
#include <cassert>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <typeindex>

#include "allocator.h"
#include "shape.h"
#include "serialize_oplist.h"
#include "deserializer.h"
#include "dtype.h"
#include "float16.h"
#include "graph_status.h"
#include "interface_defs.h"
#include "log.h"
#include "memory_layout.h"
#include "padding.h"
#include "template_help.h"
#include "conversions.h"
#include "crate.h"
#include "minihash.h"
#include "macros_attribute.h"
#include "weak_linkage.h"
#include "interface.h"

#define TENSOR_MAGIC 0x1337beef

// BELOW are the different sections that make up tensor.h
// *** With few exceptions, they need to be in the order below ***
//
// All of the .h files in tens/ subdir are intended to be included *only* here.
// Please add new definitions to the proper section, or create a new one if appropriate
//
#include "tens/block_enumeration.h" // MemBlockEnumerator and subclasses; used in Tensor methods
#include "tens/tensor_base.h" // class Tensor, and most subclasses
#include "tens/tensor_concrete.h" // LayoutTensor, ConcreteTensor classes, and related
#include "tens/tensor_definitions.h" // all of the code layout and tensor types
#include "tens/tensor_generator.h" // tensor generator mechanism
#include "tile_extract.h" // tile extract methods.
#endif
