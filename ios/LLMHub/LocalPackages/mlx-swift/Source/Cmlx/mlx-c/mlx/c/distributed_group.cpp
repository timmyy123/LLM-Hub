/* Copyright © 2023-2024 Apple Inc. */

#include <cstring>

#include "mlx/c/distributed_group.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" mlx_distributed_group mlx_distributed_group_new(void) {
  try {
    return mlx_distributed_group_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return {nullptr};
  }
}

extern "C" int mlx_distributed_group_free(mlx_distributed_group group) {
  try {
    mlx_distributed_group_free_(group);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int
mlx_distributed_init(mlx_distributed_group* res, bool strict, const char* bk) {
  try {
    if (bk) {
      mlx_distributed_group_set_(
          *res, mlx::core::distributed::init(strict, bk));
    } else {
      mlx_distributed_group_set_(*res, mlx::core::distributed::init(strict));
    }
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_distributed_group_rank(mlx_distributed_group group) {
  try {
    return mlx_distributed_group_get_(group).rank();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}

extern "C" int mlx_distributed_group_size(mlx_distributed_group group) {
  try {
    return mlx_distributed_group_get_(group).size();
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 0;
  }
}

extern "C" int mlx_distributed_group_split(
    mlx_distributed_group* res,
    mlx_distributed_group group,
    int color,
    int key) {
  try {
    mlx_distributed_group_set_(
        *res, mlx_distributed_group_get_(group).split(color, key));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" bool mlx_distributed_is_available(const char* bk) {
  try {
    if (bk) {
      return mlx::core::distributed::is_available(bk);
    } else {
      return mlx::core::distributed::is_available();
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return false;
  }
}
