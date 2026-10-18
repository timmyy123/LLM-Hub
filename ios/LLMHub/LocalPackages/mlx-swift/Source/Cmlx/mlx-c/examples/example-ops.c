/* Copyright © 2023-2024 Apple Inc. */

#include <stdio.h>
#include "mlx/c/mlx.h"

void print_array(const char* msg, mlx_array arr) {
  mlx_string str = mlx_string_new();
  mlx_array_tostring(&str, arr);
  printf("%s\n%s\n", msg, mlx_string_data(str));
  mlx_string_free(str);
}

/* Exercises a range of array, elementwise, and linalg ops. */
int main(void) {
  mlx_stream stream = mlx_default_cpu_stream_new();

  float data[] = {1, -2, 0, 4, -5, 6};
  int shape[] = {2, 3};
  mlx_array arr = mlx_array_new_data(data, shape, 2, MLX_FLOAT32);
  mlx_array res = mlx_array_new();

  mlx_positive(&res, arr, stream);
  print_array("positive", res);

  mlx_trunc(&res, arr, stream);
  print_array("trunc", res);

  mlx_count_nonzero_axis(&res, arr, 1, false, stream);
  print_array("count_nonzero (axis=1)", res);

  mlx_diff(&res, arr, 1, 1, stream);
  print_array("diff (n=1, axis=1)", res);

  mlx_flip_axis(&res, arr, 1, stream);
  print_array("flip (axis=1)", res);

  float rhs_data[] = {6, 5, 4, 3, 2, 1};
  mlx_array rhs = mlx_array_new_data(rhs_data, shape, 2, MLX_FLOAT32);
  mlx_vecdot(&res, arr, rhs, 1, stream);
  print_array("vecdot (axis=1)", res);

  mlx_array lbool = mlx_array_new_bool(true);
  mlx_array rbool = mlx_array_new_bool(false);
  mlx_logical_xor(&res, lbool, rbool, stream);
  print_array("logical_xor", res);

  mlx_vector_array parts = mlx_vector_array_new();
  mlx_unstack_axis(&parts, arr, 0, stream);
  printf("unstack (axis=0): %zu arrays\n", mlx_vector_array_size(parts));

  float square_data[] = {4, 3, 6, 3};
  int square_shape[] = {2, 2};
  mlx_array square =
      mlx_array_new_data(square_data, square_shape, 2, MLX_FLOAT32);
  mlx_linalg_det(&res, square, stream);
  print_array("linalg_det", res);

  mlx_array sign = mlx_array_new();
  mlx_array logabsdet = mlx_array_new();
  mlx_linalg_slogdet(&sign, &logabsdet, square, stream);
  print_array("slogdet sign", sign);
  print_array("slogdet logabsdet", logabsdet);

  mlx_array_free(arr);
  mlx_array_free(rhs);
  mlx_array_free(res);
  mlx_array_free(lbool);
  mlx_array_free(rbool);
  mlx_array_free(square);
  mlx_array_free(sign);
  mlx_array_free(logabsdet);
  mlx_vector_array_free(parts);
  mlx_stream_free(stream);

  return 0;
}
