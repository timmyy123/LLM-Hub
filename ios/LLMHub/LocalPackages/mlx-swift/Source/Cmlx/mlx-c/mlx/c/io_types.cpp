/* Copyright © 2023-2024 Apple Inc. */

#include <cstring>

#include "mlx/c/device.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"

extern "C" mlx_io_reader mlx_io_reader_new(void* desc, mlx_io_vtable vtable) {
  try {
    return mlx_io_reader_new_(desc, vtable);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_io_reader({nullptr});
  }
}

extern "C" int mlx_io_reader_free(mlx_io_reader io) {
  try {
    mlx_io_reader_free_(io);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_io_reader_descriptor(void** desc_, mlx_io_reader io) {
  try {
    *desc_ = mlx_io_reader_get_(io)->desc;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_io_reader_tostring(mlx_string* str_, mlx_io_reader io) {
  try {
    mlx_string_set_(*str_, mlx_io_reader_get_(io)->label());
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" mlx_io_writer mlx_io_writer_new(void* desc, mlx_io_vtable vtable) {
  try {
    return mlx_io_writer_new_(desc, vtable);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_io_writer({nullptr});
  }
}

extern "C" int mlx_io_writer_free(mlx_io_writer io) {
  try {
    mlx_io_writer_free_(io);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_io_writer_descriptor(void** desc_, mlx_io_writer io) {
  try {
    *desc_ = mlx_io_writer_get_(io)->desc;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_io_writer_tostring(mlx_string* str_, mlx_io_writer io) {
  try {
    mlx_string_set_(*str_, mlx_io_writer_get_(io)->label());
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" mlx_io_gguf mlx_io_gguf_new(void) {
  try {
    return mlx_io_gguf_new_(mlx::core::GGUFLoad());
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_io_gguf({nullptr});
  }
}

extern "C" int mlx_io_gguf_free(mlx_io_gguf io) {
  try {
    mlx_io_gguf_free_(io);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_io_gguf_get_keys(mlx_vector_string* keys, mlx_io_gguf io) {
  try {
    auto& cpp_map = mlx_io_gguf_get_(io).first;
    std::vector<std::string> cpp_keys;
    cpp_keys.reserve(cpp_map.size());
    for (const auto& [key, value] : cpp_map) {
      cpp_keys.push_back(key);
    }
    mlx_vector_string_set_(*keys, cpp_keys);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int
mlx_io_gguf_get_array(mlx_array* arr, mlx_io_gguf io, const char* key) {
  try {
    auto& cpp_map = mlx_io_gguf_get_(io).first;
    auto it = cpp_map.find(key);
    if (it != cpp_map.end()) {
      mlx_array_set_(*arr, it->second);
    } else {
      return 2; // not found
    }
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

#define IMPLEMENT_GGUF_GET_METADATA(CNAME, CPPTYPE)                \
  extern "C" int mlx_io_gguf_get_metadata_##CNAME(                 \
      mlx_##CNAME* res, mlx_io_gguf io, const char* key) {         \
    try {                                                          \
      auto& cpp_map = mlx_io_gguf_get_(io).second;                 \
      auto it = cpp_map.find(key);                                 \
      if (it != cpp_map.end()) {                                   \
        if (std::holds_alternative<CPPTYPE>(it->second)) {         \
          mlx_##CNAME##_set_(*res, std::get<CPPTYPE>(it->second)); \
        } else {                                                   \
          return 3; /* found key, wrong value type */              \
        }                                                          \
      } else {                                                     \
        return 2; /* not found */                                  \
      }                                                            \
      return 0;                                                    \
    } catch (std::exception & e) {                                 \
      mlx_error(e.what());                                         \
      return 1;                                                    \
    }                                                              \
  }

IMPLEMENT_GGUF_GET_METADATA(array, mlx::core::array);
IMPLEMENT_GGUF_GET_METADATA(string, std::string);
IMPLEMENT_GGUF_GET_METADATA(vector_string, std::vector<std::string>);

#define IMPLEMENT_GGUF_HAS_METADATA(CNAME, CPPTYPE)        \
  extern "C" int mlx_io_gguf_has_metadata_##CNAME(         \
      bool* flag, mlx_io_gguf io, const char* key) {       \
    try {                                                  \
      auto& cpp_map = mlx_io_gguf_get_(io).second;         \
      auto it = cpp_map.find(key);                         \
      if (it != cpp_map.end()) {                           \
        if (std::holds_alternative<CPPTYPE>(it->second)) { \
          *flag = true;                                    \
        } else {                                           \
          *flag = false;                                   \
        }                                                  \
        return 0;                                          \
      } else {                                             \
        *flag = false;                                     \
        return 2; /* not found */                          \
      }                                                    \
      return 0;                                            \
    } catch (std::exception & e) {                         \
      *flag = false;                                       \
      mlx_error(e.what());                                 \
      return 1;                                            \
    }                                                      \
  }

IMPLEMENT_GGUF_HAS_METADATA(array, mlx::core::array);
IMPLEMENT_GGUF_HAS_METADATA(string, std::string);
IMPLEMENT_GGUF_HAS_METADATA(vector_string, std::vector<std::string>);

extern "C" int
mlx_io_gguf_set_array(mlx_io_gguf io, const char* key, const mlx_array arr) {
  try {
    auto& cpp_map = mlx_io_gguf_get_(io).first;
    cpp_map.insert(std::make_pair(std::string(key), mlx_array_get_(arr)));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_io_gguf_set_metadata_array(
    mlx_io_gguf io,
    const char* key,
    const mlx_array marr) {
  try {
    auto& cpp_map = mlx_io_gguf_get_(io).second;
    cpp_map.insert(std::make_pair(std::string(key), mlx_array_get_(marr)));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_io_gguf_set_metadata_string(
    mlx_io_gguf io,
    const char* key,
    const char* mstr) {
  try {
    auto& cpp_map = mlx_io_gguf_get_(io).second;
    cpp_map.insert(std::make_pair(std::string(key), std::string(mstr)));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_io_gguf_set_metadata_vector_string(
    mlx_io_gguf io,
    const char* key,
    const mlx_vector_string mvstr) {
  try {
    auto& cpp_map = mlx_io_gguf_get_(io).second;
    cpp_map.insert(
        std::make_pair(std::string(key), mlx_vector_string_get_(mvstr)));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}
