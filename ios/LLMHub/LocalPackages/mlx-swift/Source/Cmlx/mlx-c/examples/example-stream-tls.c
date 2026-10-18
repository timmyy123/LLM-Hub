/* Copyright © 2026 Apple Inc. */
/* Exercises the thread-local / vector-stream C bindings. */

#include <stdio.h>

#include "mlx/c/mlx.h"

int main(void) {
  mlx_device dev = mlx_device_new();
  mlx_get_default_device(&dev);

  mlx_stream_thread_local tls = mlx_stream_thread_local_new(dev);
  if (!tls.ctx)
    mlx_error("mlx_stream_thread_local_new returned empty");
  printf("thread-local stream created\n");

  mlx_stream cur = mlx_stream_new();
  if (mlx_stream_from_thread_local(&cur, tls))
    mlx_error("mlx_stream_from_thread_local failed");
  printf("stream_from_thread_local ok\n");

  mlx_vector_stream streams = mlx_vector_stream_new();
  if (mlx_get_streams(&streams))
    mlx_error("mlx_get_streams failed");

  size_t n = mlx_vector_stream_size(streams);
  printf("get_streams returned %zu streams\n", n);
  if (n == 0)
    mlx_error("expected at least one stream");

  mlx_stream first = mlx_stream_new();
  if (mlx_vector_stream_get(&first, streams, 0))
    mlx_error("mlx_vector_stream_get failed");
  mlx_vector_stream_free(streams);

  if (mlx_synchronize_default())
    mlx_error("mlx_synchronize_default failed");
  if (mlx_synchronize_thread_local(tls))
    mlx_error("mlx_synchronize_thread_local failed");

  mlx_stream_thread_local tls2 = mlx_stream_thread_local_new(dev);
  if (mlx_stream_thread_local_set(&tls2, tls))
    mlx_error("mlx_stream_thread_local_set failed");
  if (mlx_stream_thread_local_free(tls2))
    mlx_error("mlx_stream_thread_local_free failed");
  mlx_stream_thread_local_free(tls);

  if (mlx_clear_streams())
    mlx_error("mlx_clear_streams failed");

  mlx_stream_free(cur);
  mlx_stream_free(first);
  mlx_device_free(dev);
  printf("ok\n");
  return 0;
}
