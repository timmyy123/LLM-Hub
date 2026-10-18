/* Copyright © 2025 Apple Inc. */

/* Streams are thread affine: a stream created on one thread cannot be used to
 * evaluate on another. `mlx_stream_new_thread_unsafe` opts out of that, which
 * is what language bindings that do their own locking need.
 *
 * This creates a stream on the main thread and evaluates on it from a worker
 * thread. Note the mutex: MLX applies no synchronization to these streams, so
 * serializing the work is the caller's job.
 */

#include <pthread.h>
#include <stdio.h>

#include "mlx/c/mlx.h"

static pthread_mutex_t eval_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
  mlx_stream stream;
  float result;
  int status;
} work;

static void* run(void* arg) {
  work* w = (work*)arg;

  pthread_mutex_lock(&eval_mutex);

  mlx_array a = mlx_array_new_float(2.0);
  mlx_array b = mlx_array_new_float(3.0);
  mlx_array sum = mlx_array_new();

  w->status = mlx_add(&sum, a, b, w->stream);
  if (w->status == 0) {
    w->status = mlx_array_item_float32(&w->result, sum);
  }

  mlx_array_free(a);
  mlx_array_free(b);
  mlx_array_free(sum);

  pthread_mutex_unlock(&eval_mutex);
  return NULL;
}

int main(void) {
  mlx_device dev = mlx_device_new();
  mlx_get_default_device(&dev);

  /* created here, used on the worker thread below */
  work w = {mlx_stream_new_thread_unsafe(dev), 0.0f, 0};

  pthread_t thread;
  if (pthread_create(&thread, NULL, run, &w) != 0) {
    printf("failed to spawn thread\n");
    return 1;
  }
  pthread_join(thread, NULL);

  if (w.status != 0) {
    /* the default error handler has already reported the reason */
    printf("evaluation failed\n");
    return 1;
  }

  printf("2 + 3 evaluated on another thread = %g\n", w.result);

  mlx_stream_free(w.stream);
  mlx_device_free(dev);

  return 0;
}
