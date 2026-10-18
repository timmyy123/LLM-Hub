// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================

#ifndef COMPILE_TIME_STRING_H_
#define COMPILE_TIME_STRING_H_

#include <array>
#include <cstddef>

namespace hnnx {
#if defined(__clang__)
#if __clang_major__ >= 18
#if __has_warning("-Wunsafe-buffer-usage")
#pragma clang unsafe_buffer_usage begin
#endif
#endif
#endif
// LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
template <int N, int M> constexpr auto ConcatStr(const char *a, const char *b)
{
    std::array<char, static_cast<std::size_t>(N + M + 1)> result{};

    for (std::size_t i = 0; i < N; i++) {
        result[i] = a[i];
    }

    for (std::size_t i = 0; i < M; i++) {
        result[i + N] = b[i];
    }

    result.back() = '\0';

    return std::move(result);
}
template <std::size_t N, std::size_t M>
constexpr auto ConcatStr(const char (&a)[N], const char (&b)[M]) -> std::array<char, N + M - 1>
{
    std::array<char, N + M - 1> result{};

    constexpr std::size_t a_offset{N - 1};

    for (std::size_t i = 0; i < a_offset; i++) {
        result[i] = a[i];
    }

    for (std::size_t i = 0; i < M - 1; i++) {
        result[a_offset + i] = b[i];
    }

    result.back() = '\0';

    return result;
}
// LCOV_EXCL_STOP

// LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
template <std::size_t... Ns>
constexpr auto ConcatStr(std::array<char, Ns> const &...arrays) -> std::array<char, (0 + ... + Ns) - sizeof...(Ns) + 1>
{
    std::array<char, (0 + ... + Ns) - sizeof...(Ns) + 1> result{};

    std::size_t offset{0};
    auto copy_array{[&result, &offset](const auto &arr) {
        auto const n{arr.size()};
        auto const length_without_null{n - 1};

        for (std::size_t i{0}; i < length_without_null; i++) {
            result[offset + i] = arr[i];
        }

        offset += length_without_null;
    }};

    (copy_array(arrays), ...);
    result.back() = '\0';

    return result;
}
// LCOV_EXCL_STOP

template <typename T> constexpr auto ConstexprStrLen(T) -> std::size_t
{
    return 0;
}

// LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
template <> constexpr auto ConstexprStrLen<const char *>(const char *str) -> std::size_t
{
    std::size_t len = 0;
    while (*str != 0) {
        len++;
        str++;
    }
    return len;
}
#if defined(__clang__)
#if __clang_major__ >= 18
#if __has_warning("-Wunsafe-buffer-usage")
#pragma clang unsafe_buffer_usage end
#endif
#endif
#endif

template <std::size_t N> constexpr auto ConstexprStrLen(const char (&)[N]) -> std::size_t
{
    return N - 1; // exclude null terminator
}

// LCOV_EXCL_STOP
} // namespace hnnx
#endif
