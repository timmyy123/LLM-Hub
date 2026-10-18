// ==============================================================================
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause-Clear
//
// ==============================================================================
#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace hnnx {
/** @brief sv_size_wrapper a wrapper template for string_view that carries the view size as
 *  a template parameter. This allows the size to be inferred by the built_array
 *  constructor template.
 */

// LCOV_EXCL_START [SAFTYSWCCB-1736] constexprs resolved during compile time
template <std::string_view::size_type S> struct sv_size_wrapper {
    using view_type = std::string_view;
    using size_type = view_type::size_type;
    using value_type = view_type::value_type;

    static constexpr auto size() { return S; }

    constexpr bool operator==(sv_size_wrapper const &other) const noexcept { return v == other.v; }
    constexpr bool operator!=(sv_size_wrapper const &other) const noexcept { return v != other.v; }
    constexpr bool operator==(view_type const other) const noexcept { return v == other; }
    constexpr bool operator!=(view_type const other) const noexcept { return v != other; }

    view_type v;
};

// Deduction guides
template <std::size_t N> sv_size_wrapper(const char (&)[N]) -> sv_size_wrapper<N - 1>; // drop null terminator

using built_array_size_type = std::uint32_t;

#if defined(__clang__)
#pragma clang diagnostic push
#if __has_warning("-Wunsafe-buffer-usage")
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif
#pragma clang diagnostic ignored "-Wpadded"
#endif
template <typename T> struct built_array_view {
    using value_type = T;
    using size_type = built_array_size_type;
    constexpr auto operator[](size_type const i) const noexcept -> value_type const & { return ptr[i]; }
    constexpr operator bool() const noexcept { return ptr != nullptr; }

    T const *ptr{nullptr};
    size_type size{0u};
};
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

/** @brief built_array */
template <typename T, std::uint32_t N> class built_array {
    /** @brief arr */
    std::array<T, N> arr{};

  public:
    using value_type = T;
    using size_type = built_array_size_type;
    using view_type = built_array_view<value_type>;

    template <std::string_view::size_type I> constexpr built_array(sv_size_wrapper<I> newElem) noexcept
    {
        for (uint32_t i{0}; i < I; i++) {
            arr[i] = newElem.v[i];
        }
    }
    constexpr built_array(const char (&array_arg)[N]) noexcept
    {
        static_assert(std::is_same<value_type, char>::value, "Only valid for char");
#if defined(__clang__)
#if __has_warning("-Wunsafe-buffer-usage")
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif
#endif
        for (size_type i{0}; i < N; i++) {
            arr[i] = array_arg[i];
        }
#if defined(__clang__)
#if __has_warning("-Wunsafe-buffer-usage")
#pragma clang diagnostic pop
#endif
#endif
    }

    /** @brief built_array
     *  @param old the previous array
     *  @param newElem a view of the array of new elements to append
     */
    template <std::string_view::size_type I>
    constexpr built_array(built_array<T, N - I> const &old, sv_size_wrapper<I> newElem) noexcept
    {
        if constexpr (N > I) {
            auto old_view{old.get_view()};
            for (size_type i{0U}; i < (N - I); i++) {
                arr[i] = old_view[i];
            }
        }
        for (size_type i = (N - I); i < N; i++) {
            arr[i] = newElem.v[i - (N - I)];
        }
    }
    constexpr built_array(built_array<T, N - 1> const &old, T const newElem) noexcept
    {
        if constexpr (N > 1) {
            auto old_view{old.get_view()};
            for (size_type i = 0U; i < N - 1U; i++) {
                arr[i] = old_view[i];
            }
        }
        arr[N - 1U] = newElem;
    }

    /** @brief size
     *  @return the array size
     */
    static constexpr auto size() { return N; }
    /** @brief get_arr
     *  @return the array
     */
    constexpr const std::array<T, N> get_arr() const noexcept { return arr; }
    /** @brief built_array
     *  @param old the previous array
     *  @param newElem the new element to append
     */

    /** @brief append
     *  @param newElem the new element to append
     *  @return the new array
     */
    constexpr built_array<T, N + 1> append(T newElem) const { return built_array<T, N + 1>(*this, newElem); }

    /** @brief append
     *  @param newElem the new element to append
     *  @return the new array
     */
    template <std::string_view::size_type I> constexpr built_array<T, N + I> append(sv_size_wrapper<I> newElem) const
    {
        return built_array<T, N + I>(*this, newElem);
    }

    constexpr auto to_string_view() const -> std::string_view
    {
        if constexpr (std::is_same_v<value_type, char>) {
            return {arr.data(), arr.size()};
        } else {
            return {};
        }
    }

    constexpr auto get_view() const noexcept -> view_type
    {
        return view_type{arr.data(), static_cast<size_type>(arr.size())};
    }
};

/** @brief built_array specialization for N = 0 */
template <typename T> class built_array<T, 0> {
    // MSVC C3615 workaround: value-initializing std::array<T,0>{} in a constexpr function
    // calls __builtin_array_init_helper which is not constexpr in MSVC 14.34 when T is
    // non-trivially constructible.  Storing arr as a member (no brace-initializer) causes
    // the explicitly-defaulted constructor to default-initialize arr, calling T's constexpr
    // default constructor directly and bypassing __builtin_array_init_helper.  get_arr()
    // then trivially returns the member, so the constexpr requirement is satisfied.
    std::array<T, 0> arr; // intentionally no {} – default-initialized, not value-initialized
  public:
    using value_type = T;
    using size_type = built_array_size_type;

    /** @brief built_array constructor */
    constexpr built_array() noexcept = default;

    static constexpr size_type size() { return 0; }

    /** @brief append
     *  @param newElem the new element to append
     *  @return the new array
     */
    constexpr built_array<T, 1> append(T newElem) const { return built_array<T, 1>(*this, newElem); }

    /** @brief append
     *  @param newElem the new element to append
     *  @return the new array
     */
    template <std::string_view::size_type I> constexpr built_array<T, I> append(sv_size_wrapper<I> newElem) const
    {
        return built_array<T, I>(*this, newElem);
    }

    /** @brief get_arr
     *  @return the array
     */
    constexpr const std::array<T, 0> get_arr() const noexcept { return arr; }
};

// Deduction guides
template <std::size_t N> built_array(sv_size_wrapper<N>) -> built_array<typename sv_size_wrapper<N>::value_type, N>;

template <std::size_t N> built_array(const char (&)[N]) -> built_array<char, N>;

template <typename T, std::uint32_t N, std::size_t M>
built_array(built_array<T, N> const &, sv_size_wrapper<M>) -> built_array<T, N + M>;

template <typename T, std::uint32_t N> built_array(built_array<T, N> const &, T) -> built_array<T, N + 1>;
// LCOV_EXCL_STOP
} // namespace hnnx
