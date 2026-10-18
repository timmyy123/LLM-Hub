/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_GRAPH_UTILS_PRIVATE_H
#define MLX_GRAPH_UTILS_PRIVATE_H

#include "mlx/c/graph_utils.h"
#include "mlx/graph_utils.h"

inline mlx_node_namer mlx_node_namer_new_() {
  return mlx_node_namer({nullptr});
}

inline mlx_node_namer mlx_node_namer_new_(mlx::core::NodeNamer&& s) {
  return mlx_node_namer({new mlx::core::NodeNamer(std::move(s))});
}

inline mlx_node_namer& mlx_node_namer_set_(
    mlx_node_namer& d,
    mlx::core::NodeNamer&& s) {
  if (d.ctx) {
    delete static_cast<mlx::core::NodeNamer*>(d.ctx);
  }
  d.ctx = new mlx::core::NodeNamer(std::move(s));
  return d;
}

inline mlx::core::NodeNamer& mlx_node_namer_get_(mlx_node_namer d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_node_namer");
  }
  return *static_cast<mlx::core::NodeNamer*>(d.ctx);
}

inline void mlx_node_namer_free_(mlx_node_namer d) {
  if (d.ctx) {
    delete static_cast<mlx::core::NodeNamer*>(d.ctx);
  }
}

#endif
