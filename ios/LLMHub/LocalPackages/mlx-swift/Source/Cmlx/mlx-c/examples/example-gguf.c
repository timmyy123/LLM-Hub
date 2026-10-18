/* Copyright © 2023-2024 Apple Inc. */

#include <stdio.h>
#include "mlx/c/mlx.h"

char default_filename[] = "test.gguf";
const char* dtype_strs[] = {
    "MLX_BOOL",
    "MLX_UINT8",
    "MLX_UINT16",
    "MLX_UINT32",
    "MLX_UINT64",
    "MLX_INT8",
    "MLX_INT16",
    "MLX_INT32",
    "MLX_INT64",
    "MLX_FLOAT16",
    "MLX_FLOAT32",
    "MLX_FLOAT64",
    "MLX_BFLOAT16",
    "MLX_COMPLEX64",
};

void add_zero_array_to_gguf(
    mlx_io_gguf gguf,
    const char* key,
    const int* shape,
    int ndim,
    mlx_dtype dtype,
    const char* metadata,
    mlx_stream s) {
  mlx_array arr = mlx_array_new();
  mlx_zeros(&arr, shape, ndim, dtype, s);
  mlx_io_gguf_set_array(gguf, key, arr);
  if (metadata) {
    mlx_io_gguf_set_metadata_string(gguf, key, metadata);
  }
  mlx_array_free(arr);
}

int main(int argc, char* argv[]) {
  mlx_stream stream = mlx_default_cpu_stream_new();

  char* filename = default_filename;
  if (argc > 1) {
    filename = argv[1];
  } else {
    mlx_io_gguf gguf = mlx_io_gguf_new();
    add_zero_array_to_gguf(
        gguf, "array3D", (int[]){3, 4, 5}, 3, MLX_FLOAT16, NULL, stream);
    add_zero_array_to_gguf(
        gguf,
        "array2D",
        (int[]){6, 7},
        2,
        MLX_FLOAT32,
        "a 6x7 zero array",
        stream);
    mlx_save_gguf(filename, gguf);
    mlx_io_gguf_free(gguf);
  }

  mlx_io_gguf gguf = mlx_io_gguf_new();
  mlx_load_gguf(&gguf, filename, stream);
  mlx_vector_string keys = mlx_vector_string_new();
  mlx_io_gguf_get_keys(&keys, gguf);

  char* key;
  bool flag;
  mlx_array value = mlx_array_new();
  for (int i = 0; i < mlx_vector_string_size(keys); i++) {
    mlx_vector_string_get(&key, keys, i);
    printf("%s ", key);
    mlx_io_gguf_get_array(&value, gguf, key);
    const int* shape = mlx_array_shape(value);
    for (int d = 0; d < mlx_array_ndim(value); d++) {
      if (d != 0) {
        printf("x");
      }
      printf("%d", shape[d]);
    }
    printf(" %s", dtype_strs[mlx_array_dtype(value)]);

    mlx_io_gguf_has_metadata_array(&flag, gguf, key);
    if (flag) {
      printf(" [array]");
    }
    mlx_io_gguf_has_metadata_string(&flag, gguf, key);
    if (flag) {
      printf(" [string]");
    }
    mlx_io_gguf_has_metadata_vector_string(&flag, gguf, key);
    if (flag) {
      printf(" [vector string]");
    }

    printf("\n");
  }

  mlx_array_free(value);
  mlx_vector_string_free(keys);
  mlx_io_gguf_free(gguf);
  mlx_stream_free(stream);

  return 0;
}
