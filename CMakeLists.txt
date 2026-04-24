cmake_minimum_required(VERSION 3.16)
project(tpu_event_model CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(tpu_sim
    src/engine/engine.cpp
    src/schedule/dummy_schedule.cpp
)
target_include_directories(tpu_sim PUBLIC src)

add_executable(tpu_sim_main src/main.cpp)
target_link_libraries(tpu_sim_main PRIVATE tpu_sim)

enable_testing()
add_executable(test_engine tests/test_engine.cpp)
target_link_libraries(test_engine PRIVATE tpu_sim)
add_test(NAME test_engine COMMAND test_engine)
