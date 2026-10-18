/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_IO_TYPES_PRIVATE_H
#define MLX_IO_TYPES_PRIVATE_H

#include "mlx/c/io_types.h"
#include "mlx/mlx.h"

inline mlx_io_gguf mlx_io_gguf_new_() {
  return mlx_io_gguf({nullptr});
}

inline mlx_io_gguf mlx_io_gguf_new_(mlx::core::GGUFLoad&& s) {
  return mlx_io_gguf({new mlx::core::GGUFLoad(std::move(s))});
}

inline mlx_io_gguf& mlx_io_gguf_set_(mlx_io_gguf& d, mlx::core::GGUFLoad&& s) {
  if (d.ctx) {
    delete static_cast<mlx::core::GGUFLoad*>(d.ctx);
  }
  d.ctx = new mlx::core::GGUFLoad(std::move(s));
  return d;
}

inline mlx::core::GGUFLoad& mlx_io_gguf_get_(mlx_io_gguf d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_io_gguf");
  }
  return *static_cast<mlx::core::GGUFLoad*>(d.ctx);
}

inline void mlx_io_gguf_free_(mlx_io_gguf d) {
  if (d.ctx) {
    delete static_cast<mlx::core::GGUFLoad*>(d.ctx);
  }
}

#endif
