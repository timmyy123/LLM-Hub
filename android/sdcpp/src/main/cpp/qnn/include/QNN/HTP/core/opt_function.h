// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef OPT_FUNCTION_H_
#define OPT_FUNCTION_H_

#include "weak_linkage.h"
#include "macros_attribute.h"
#include "crate.h"

#include <type_traits>
#include <mutex>

PUSH_VISIBILITY(default)

API_EXPORT hnnx::Crate *get_lambda_crate();

template <typename R> class OptFunction;
extern std::mutex lambda_mutex;

template <typename R, typename... Args> class OptFunction<R(Args...)> {
  public:
    using thisType = OptFunction<R(Args...)>;
    using OptFunctionTType = R (*)(void *, Args...);
    using OptFunctionType = R (*)(Args...);

    template <typename L> API_EXPORT static R LambdaWrapper(void *t, Args... args)
    {
        L *const obj = (L *)t;
        return obj->operator()(args...);
    }
    API_EXPORT static R FunctionWrapper(void *t, Args... args)
    {
        OptFunctionType const obj = (OptFunctionType)t;
        return obj(args...);
    }
    template <typename L>
    API_EXPORT static typename std::enable_if<!std::is_lvalue_reference_v<L>, thisType>::type create(L &&lambda)
    {
        std::scoped_lock my_lock{lambda_mutex};

        L *const l = get_lambda_crate()->emplace<L>(std::forward<L>(lambda));
        return thisType(LambdaWrapper<L>, l);
    }

    OptFunctionTType mFunc{nullptr};
    void *mObj{nullptr};

    API_EXPORT OptFunction() = default;
    API_EXPORT OptFunction(OptFunctionTType f, void *o) : mFunc(f), mObj(o) {}

    API_EXPORT R operator()(Args... args) const { return mFunc(mObj, args...); }
    API_EXPORT operator bool() const { return (mFunc != nullptr); }
};

POP_VISIBILITY()
#endif
