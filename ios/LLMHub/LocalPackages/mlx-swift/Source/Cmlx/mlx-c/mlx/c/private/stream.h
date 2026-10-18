/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_STREAM_PRIVATE_H
#define MLX_STREAM_PRIVATE_H

#include "mlx/c/stream.h"
#include "mlx/mlx.h"

inline mlx_stream mlx_stream_new_() {
  return mlx_stream({nullptr});
}

inline mlx_stream mlx_stream_new_(const mlx::core::Stream& s) {
  return mlx_stream({new mlx::core::Stream(s)});
}

inline mlx_stream mlx_stream_new_(mlx::core::Stream&& s) {
  return mlx_stream({new mlx::core::Stream(std::move(s))});
}

inline mlx_stream& mlx_stream_set_(mlx_stream& d, const mlx::core::Stream& s) {
  if (d.ctx) {
    *static_cast<mlx::core::Stream*>(d.ctx) = s;
  } else {
    d.ctx = new mlx::core::Stream(s);
  }
  return d;
}

inline mlx_stream& mlx_stream_set_(mlx_stream& d, mlx::core::Stream&& s) {
  if (d.ctx) {
    *static_cast<mlx::core::Stream*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new mlx::core::Stream(std::move(s));
  }
  return d;
}

inline mlx::core::Stream& mlx_stream_get_(mlx_stream d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_stream");
  }
  return *static_cast<mlx::core::Stream*>(d.ctx);
}

inline void mlx_stream_free_(mlx_stream d) {
  if (d.ctx) {
    delete static_cast<mlx::core::Stream*>(d.ctx);
  }
}

inline mlx_stream_thread_local mlx_stream_thread_local_new_() {
  return mlx_stream_thread_local({nullptr});
}

inline mlx_stream_thread_local mlx_stream_thread_local_new_(
    const mlx::core::ThreadLocalStream& s) {
  return mlx_stream_thread_local({new mlx::core::ThreadLocalStream(s)});
}

inline mlx_stream_thread_local mlx_stream_thread_local_new_(
    mlx::core::ThreadLocalStream&& s) {
  return mlx_stream_thread_local(
      {new mlx::core::ThreadLocalStream(std::move(s))});
}

inline mlx_stream_thread_local& mlx_stream_thread_local_set_(
    mlx_stream_thread_local& d,
    const mlx::core::ThreadLocalStream& s) {
  if (d.ctx) {
    *static_cast<mlx::core::ThreadLocalStream*>(d.ctx) = s;
  } else {
    d.ctx = new mlx::core::ThreadLocalStream(s);
  }
  return d;
}

inline mlx_stream_thread_local& mlx_stream_thread_local_set_(
    mlx_stream_thread_local& d,
    mlx::core::ThreadLocalStream&& s) {
  if (d.ctx) {
    *static_cast<mlx::core::ThreadLocalStream*>(d.ctx) = std::move(s);
  } else {
    d.ctx = new mlx::core::ThreadLocalStream(std::move(s));
  }
  return d;
}

inline mlx::core::ThreadLocalStream& mlx_stream_thread_local_get_(
    mlx_stream_thread_local d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_stream_thread_local");
  }
  return *static_cast<mlx::core::ThreadLocalStream*>(d.ctx);
}

inline void mlx_stream_thread_local_free_(mlx_stream_thread_local d) {
  if (d.ctx) {
    delete static_cast<mlx::core::ThreadLocalStream*>(d.ctx);
  }
}

#endif
