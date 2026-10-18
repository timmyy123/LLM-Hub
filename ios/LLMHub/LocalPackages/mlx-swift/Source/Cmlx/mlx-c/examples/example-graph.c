/* Copyright © 2023-2024 Apple Inc. */

#include <stdio.h>
#include "mlx/c/mlx.h"

int main(void) {
  mlx_stream stream = mlx_default_cpu_stream_new();
  float data[] = {1, 2, 3, 4, 5, 6};
  int shape[] = {2, 3};
  mlx_array res = mlx_array_new();
  mlx_array val = mlx_array_new_data(data, shape, 2, MLX_FLOAT32);
  mlx_array two = mlx_array_new_int(2);
  mlx_divide(&res, val, two, stream);
  mlx_log(&res, res, stream);

  mlx_node_namer namer = mlx_node_namer_new();
  mlx_node_namer_set_name(namer, val, "inputs");
  mlx_node_namer_set_name(namer, res, "result");
  mlx_vector_array vec = mlx_vector_array_new();
  mlx_vector_array_append_value(vec, val);
  mlx_vector_array_append_value(vec, two);
  mlx_vector_array_append_value(vec, res);

  mlx_export_to_dot(stdout, namer, vec);

  mlx_array_free(val);
  mlx_array_free(two);
  mlx_array_free(res);
  mlx_vector_array_free(vec);
  mlx_node_namer_free(namer);
  mlx_stream_free(stream);

  return 0;
}
