/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#include "mlx/c/graph_utils.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/graph_utils.h"

extern "C" mlx_node_namer mlx_node_namer_new() {
  try {
    return mlx_node_namer_new_(mlx::core::NodeNamer());
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}
extern "C" int mlx_node_namer_free(mlx_node_namer namer) {
  try {
    mlx_node_namer_free_(namer);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}
extern "C" int mlx_node_namer_set_name(
    mlx_node_namer namer,
    const mlx_array arr,
    const char* name) {
  try {
    mlx_node_namer_get_(namer).set_name(mlx_array_get_(arr), name);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}
extern "C" int mlx_node_namer_get_name(
    const char** name,
    mlx_node_namer namer,
    const mlx_array arr) {
  try {
    *name = mlx_node_namer_get_(namer).get_name(mlx_array_get_(arr)).c_str();
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}
extern "C" int mlx_export_to_dot(
    FILE* os,
    const mlx_node_namer namer,
    const mlx_vector_array outputs) {
  try {
    mlx::core::export_to_dot(
        CFileOutputStream::as_lvalue(CFileOutputStream(os)),
        mlx_node_namer_get_(namer),
        mlx_vector_array_get_(outputs));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_print_graph(
    FILE* os,
    const mlx_node_namer namer,
    const mlx_vector_array outputs) {
  try {
    mlx::core::print_graph(
        CFileOutputStream::as_lvalue(CFileOutputStream(os)),
        mlx_node_namer_get_(namer),
        mlx_vector_array_get_(outputs));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
