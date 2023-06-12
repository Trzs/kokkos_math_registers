cmake_minimum_required(VERSION 3.14)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0028 NEW)

# set the project name
project(kokkos_math_registers)

SET(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include(FetchContent)
FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
)
FetchContent_MakeAvailable(kokkos)

# add_subdirectory(kokkos)

# add the executable
add_executable(kokkos_math_registers main.cpp)

target_link_libraries(kokkos_math_registers Kokkos::kokkos)
target_compile_options(kokkos_math_registers PRIVATE -Wpedantic -lineinfo --resource-usage)
#target_compile_options(kokkos_math_registers PRIVATE -Wpedantic -Rpass-analysis=kernel-resource-usage)
