/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_COMPILE_PRIVATE_H
#define MLX_COMPILE_PRIVATE_H

#include "mlx/c/compile.h"
#include "mlx/compile_impl.h"

inline mlx_compile_cache mlx_compile_cache_new_() {
  return mlx_compile_cache({nullptr});
}

inline mlx_compile_cache mlx_compile_cache_new_(
    const mlx::core::detail::CompileCacheWeakPtr& s) {
  return mlx_compile_cache({new mlx::core::detail::CompileCacheWeakPtr(s)});
}

inline mlx_compile_cache mlx_compile_cache_new_(
    mlx::core::detail::CompileCacheWeakPtr&& s) {
  return mlx_compile_cache(
      {new mlx::core::detail::CompileCacheWeakPtr(std::move(s))});
}

inline mlx_compile_cache& mlx_compile_cache_set_(
    mlx_compile_cache& d,
    const mlx::core::detail::CompileCacheWeakPtr& s) {
  if (d.ctx) {
    *static_cast<mlx::core::detail::CompileCacheWeakPtr*>(d.ctx) = s;
  } else {
    d.ctx = new mlx::core::detail::CompileCacheWeakPtr(s);
  }
  return d;
}

inline mlx_compile_cache& mlx_compile_cache_set_(
    mlx_compile_cache& d,
    mlx::core::detail::CompileCacheWeakPtr&& s) {
  if (d.ctx) {
    *static_cast<mlx::core::detail::CompileCacheWeakPtr*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new mlx::core::detail::CompileCacheWeakPtr(std::move(s));
  }
  return d;
}

inline mlx::core::detail::CompileCacheWeakPtr& mlx_compile_cache_get_(
    mlx_compile_cache d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_compile_cache");
  }
  return *static_cast<mlx::core::detail::CompileCacheWeakPtr*>(d.ctx);
}

inline void mlx_compile_cache_free_(mlx_compile_cache d) {
  if (d.ctx) {
    delete static_cast<mlx::core::detail::CompileCacheWeakPtr*>(d.ctx);
  }
}

#endif
