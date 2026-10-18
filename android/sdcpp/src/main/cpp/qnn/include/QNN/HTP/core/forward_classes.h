// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef HEXNN_FORWARD_CLASSES_H
#define HEXNN_FORWARD_CLASSES_H 1

#include <memory>
#include <cstddef>

#include "macros_attribute.h"
#include "weak_linkage.h"
PUSH_VISIBILITY(default)

class Graph;
class Op;
class OpDef;
class Tensor;
class Interface;
template <unsigned TRank> class TensorShape;
template <typename T> class PlainInterface;
template <typename T> class ScaleOffsetInterface;

namespace hnnx {

class Serializer;
class Deserializer;
struct ShapeFlags;

// this is a deleter for class T, for use in uniqe_ptr, by default it has the same
// effect as default_delete, but it can be created
// with a parameter 'true' that will cause it to do nothing instead of normal deletion.
template <typename T> class DeleterWithDisable {
    bool skip_delete;

  public:
    ALWAYSINLINE DeleterWithDisable() : skip_delete(false) {}
    ALWAYSINLINE explicit DeleterWithDisable(bool skip) : skip_delete(skip) {}
    ALWAYSINLINE DeleterWithDisable(DeleterWithDisable const &) = default;
    ALWAYSINLINE DeleterWithDisable(DeleterWithDisable &&) = default;
    ALWAYSINLINE ~DeleterWithDisable() = default;
    ALWAYSINLINE DeleterWithDisable &operator=(DeleterWithDisable const &) = default;
    ALWAYSINLINE DeleterWithDisable &operator=(DeleterWithDisable &&) = default;
    // this conversion allows us to convert a unique_ptr<T> to unique_ptr<T,DeleterWithDisable<T> >
    ALWAYSINLINE DeleterWithDisable(std::default_delete<T> const &) : skip_delete(false) {}
    ALWAYSINLINE void operator()(T const *p) const;
    ALWAYSINLINE bool delete_disabled() const { return skip_delete; }
};
template <typename T> ALWAYSINLINE void DeleterWithDisable<T>::operator()(T const *p) const
{
    if (!skip_delete) delete p;
}

// Making uptr_DWD<T> a subclass of std::unique_ptr<T, DeleterWithDisable<T>>
// allows a conversion to uptr_DWD<T> from unique_ptr<S>, where S is a subclass of T.
template <typename T> //
class uptr_DWD : public std::unique_ptr<T, DeleterWithDisable<T>> {
    using base_t = std::unique_ptr<T, DeleterWithDisable<T>>;

  public:
    template <typename D> //
    ALWAYSINLINE uptr_DWD(std::unique_ptr<D> &&d) : base_t(std::unique_ptr<T>(std::move(d)))
    {
        static_assert(std::is_base_of_v<T, D>);
    }
    ALWAYSINLINE explicit uptr_DWD(T *ptr) : base_t(ptr, DeleterWithDisable<T>(false)) {}
    ALWAYSINLINE uptr_DWD(T *ptr, bool del_disabled) : base_t(ptr, DeleterWithDisable<T>(del_disabled)) {}
    ALWAYSINLINE uptr_DWD() = default;
    ALWAYSINLINE uptr_DWD(uptr_DWD const &) = delete;
    ALWAYSINLINE uptr_DWD(uptr_DWD &&) = default;
    ALWAYSINLINE uptr_DWD &operator=(uptr_DWD const &) = delete;
    ALWAYSINLINE uptr_DWD &operator=(uptr_DWD &&) = default;
    ALWAYSINLINE ~uptr_DWD() = default;

    ALWAYSINLINE uptr_DWD(std::nullptr_t) : base_t(nullptr) {}
};

// convert a pointet to T (within crate) to uptr_DWD that has the deletion disabled
template <typename T> //
uptr_DWD<T> uptr_NoDelete(T *ptr)
{
    return uptr_DWD<T>(ptr, true);
}

using uptr_Op = uptr_DWD<Op>;
using uptr_Tensor = uptr_DWD<Tensor>;

// this can be applied to a uptr_Op or uptr_Tensor;
// it will return true if the skip flag is set (i.e the object
// is in a crate).
//
template <typename TA, typename TB> API_FUNC_EXPORT bool is_in_crate(std::unique_ptr<TA, DeleterWithDisable<TB>> &tp)
{
    return tp.get() != nullptr && tp.get_deleter().delete_disabled();
}

} // namespace hnnx

POP_VISIBILITY()

#endif
