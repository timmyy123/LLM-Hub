// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef GRAPH_HANDLE_DEFS_H
#define GRAPH_HANDLE_DEFS_H 1

#include <type_traits>
#include "forward_classes.h"
class Graph;

namespace gHN {
class gH;
class gX;
} // namespace gHN

namespace hnnx {
class GraphHandleBase;
}
namespace graph_handle_private {
Graph const &graph_from(hnnx::GraphHandleBase const &);
}

//
// This header does not require 'graph.h'
//
namespace hnnx {

// Any subclass of GraphHandleBase may be used as 'pass-by-value' parameter in an Op function.
// The class hierarchy is defined in this header.
//
class GraphHandleBase {
    friend Graph const &graph_handle_private::graph_from(hnnx::GraphHandleBase const &);

  protected:
    Graph const *m_graphp;
    explicit GraphHandleBase(Graph const &g) : m_graphp(&g) {}

    inline Graph const &graph_ref() const { return *m_graphp; }

    // All these are protected, since only subclasses may actually exist.
    ~GraphHandleBase() = default;
    GraphHandleBase(GraphHandleBase const &) = default;
    GraphHandleBase(GraphHandleBase &&) = default;
    GraphHandleBase &operator=(GraphHandleBase const &) = delete;
    GraphHandleBase &operator=(GraphHandleBase &&) = delete;
};

using GHandle = gHN::gH; // hnnx::GHandle
using GHandleX = gHN::gX; // hnnx::GHandleX
} // namespace hnnx

////////////////////
// Note: GHandle and GHandleX are the ones which appear in all the 'Op' functions,
// and are therefore mangled into everything. They are thus given short names
// gHN::gH and gHN::gX, and 'typedef'd to hnnx::GHandle and hnnx::GHandleX.

class gHN::gH : public hnnx::GraphHandleBase {
  public:
    gH(Graph const &g) : hnnx::GraphHandleBase(g) {}
    ~gH() = default;
    gH(gH const &) = default;
    gH(gH &&) = default;
    gH &operator=(gH const &) = delete;
    gH &operator=(gH &&) = delete;

    // This being false, means that adding a parameter of this type to the op function has
    // no effect on the "op id" string used in the pickle.
    static constexpr bool appears_in_opid_string = false;
};

// GHandleX is effectively identical to GHandle, but a '.' will appear at the end the Op id string
// if this is used as an Op function parameter.
// The two types will copy-construct from each other.
class gHN::gX : public hnnx::GHandle {
  public:
    gX(Graph const &g) : hnnx::GHandle(g) {}
    inline ~gX() = default;
    gX(gX const &) = default;
    gX(gX &&) = default;
    gX &operator=(gX const &) = delete;
    gX &operator=(gX &&) = delete;

    // It is ok to copy-construct this from GHandle, too.
    inline gX(hnnx::GHandle const &gh) : hnnx::GHandle(gh) {}

    // This being true means the parameter type appears in opid string used in pickle;
    // but for now, only as "" (meaning there will be a trailing ".").
    static constexpr bool appears_in_opid_string = true;
};
///////////////////////

#if 1 // will be excluded in some situations.
inline Graph const &graph_handle_private::graph_from(hnnx::GraphHandleBase const &gh)
{
    return gh.graph_ref();
}
#endif

#ifndef GHANDLE_EXPORT
#if defined(_MSC_VER)
#define GHANDLE_EXPORT __declspec(dllexport)
#define GHANDLE_IMPORT __declspec(dllimport)
#else
#define GHANDLE_EXPORT __attribute__((visibility("default")))
#define GHANDLE_IMPORT
#endif
#endif

// --- Build policy for graph-handle accessors ---
#if defined(GHANDLE_BUILD_IMPL)
#define GHANDLE_ACCESS_FUNC_QUALIFIER GHANDLE_EXPORT
#define GHANDLE_ACCESS_FUNC_INLINE    GHANDLE_EXPORT
#if !defined(GHANDLE_INCLUDE_IMPL)
#define GHANDLE_INCLUDE_IMPL 1
#endif

#elif defined(CUSTOM_OPS_BUILD)
#define GHANDLE_ACCESS_FUNC_QUALIFIER GHANDLE_IMPORT
#define GHANDLE_ACCESS_FUNC_INLINE
#undef GHANDLE_INCLUDE_IMPL
#define GHANDLE_INCLUDE_IMPL 0

#else
// GHANDLE_ACCESS_FUNC_QUALIFIER will change to API_EXPORT in some situations
#define GHANDLE_ACCESS_FUNC_QUALIFIER
// changes to 'nothing' in some situations.
#define GHANDLE_ACCESS_FUNC_INLINE inline

#if !defined(GHANDLE_INCLUDE_IMPL)
#define GHANDLE_INCLUDE_IMPL 1
#endif
#endif

// In GHANDLE_BUILD_IMPL and CUSTOM_OPS_BUILD, emit symbols in runtime_graph2
// to avoid any clash with inline definitions under runtime_graph.
// In all other builds, keep using runtime_graph.
#if defined(GHANDLE_BUILD_IMPL) || defined(CUSTOM_OPS_BUILD)
#define GHANDLE_NS runtime_graph2
#else
#define GHANDLE_NS runtime_graph
#endif

#if defined(CUSTOM_OPS_BUILD)
namespace runtime_graph2 {
}
// Consumers can keep calling runtime_graph::foo(), which resolves to runtime_graph2::foo().
namespace runtime_graph {
using namespace runtime_graph2;
}
#endif

#endif // GRAPH_HANDLE_DEFS_H
