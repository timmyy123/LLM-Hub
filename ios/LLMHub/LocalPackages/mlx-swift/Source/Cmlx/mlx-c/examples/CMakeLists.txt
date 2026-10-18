add_executable(example ${CMAKE_CURRENT_LIST_DIR}/example.c)
target_link_libraries(example PUBLIC mlxc)

add_executable(example-float64 ${CMAKE_CURRENT_LIST_DIR}/example-float64.c)
target_link_libraries(example-float64 PUBLIC mlxc)

add_executable(example-grad ${CMAKE_CURRENT_LIST_DIR}/example-grad.c)
target_link_libraries(example-grad PUBLIC mlxc)

add_executable(example-safe-tensors
               ${CMAKE_CURRENT_LIST_DIR}/example-safe-tensors.c)
target_link_libraries(example-safe-tensors PUBLIC mlxc)

add_executable(example-metal-kernel
               ${CMAKE_CURRENT_LIST_DIR}/example-metal-kernel.c)
target_link_libraries(example-metal-kernel PUBLIC mlxc)

add_executable(example-closure ${CMAKE_CURRENT_LIST_DIR}/example-closure.c)
target_link_libraries(example-closure PUBLIC mlxc)

add_executable(example-export ${CMAKE_CURRENT_LIST_DIR}/example-export.c)
target_link_libraries(example-export PUBLIC mlxc)

add_executable(example-gguf ${CMAKE_CURRENT_LIST_DIR}/example-gguf.c)
target_link_libraries(example-gguf PUBLIC mlxc)

add_executable(example-graph ${CMAKE_CURRENT_LIST_DIR}/example-graph.c)
target_link_libraries(example-graph PUBLIC mlxc)

add_executable(example-thread-stream
               ${CMAKE_CURRENT_LIST_DIR}/example-thread-stream.c)
target_link_libraries(example-thread-stream PUBLIC mlxc)
find_package(Threads REQUIRED)
target_link_libraries(example-thread-stream PUBLIC Threads::Threads)

add_executable(example-ops ${CMAKE_CURRENT_LIST_DIR}/example-ops.c)
target_link_libraries(example-ops PUBLIC mlxc)

add_executable(example-stream-tls
              ${CMAKE_CURRENT_LIST_DIR}/example-stream-tls.c)
target_link_libraries(example-stream-tls PUBLIC mlxc)
