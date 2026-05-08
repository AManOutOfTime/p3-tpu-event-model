cmake_minimum_required(VERSION 3.20)
project(tpu_event_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    add_compile_options(/W4 /permissive-)
    # yaml-cpp pulls in some warnings on MSVC; don't fail our build on them.
    add_compile_options(/wd4251)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/FetchDeps.cmake)

enable_testing()

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(tests)
