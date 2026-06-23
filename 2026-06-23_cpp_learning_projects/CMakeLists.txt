cmake_minimum_required(VERSION 3.16)

project(mini_rpc_protobuf_async
    VERSION 0.1.0
    DESCRIPTION "A learning-oriented mini RPC project using protobuf-compatible frames and async C++ sessions"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(minirpc
    src/client.cpp
    src/event_loop.cpp
    src/message.cpp
    src/protobuf_wire.cpp
    src/server.cpp
    src/session.cpp
)

target_include_directories(minirpc
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(minirpc
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(minirpc_server examples/server_main.cpp)
target_link_libraries(minirpc_server PRIVATE minirpc pthread)

add_executable(minirpc_client examples/client_main.cpp)
target_link_libraries(minirpc_client PRIVATE minirpc pthread)

enable_testing()

add_executable(minirpc_protocol_tests tests/protocol_tests.cpp)
target_link_libraries(minirpc_protocol_tests PRIVATE minirpc)
add_test(NAME minirpc_protocol_tests COMMAND minirpc_protocol_tests)
