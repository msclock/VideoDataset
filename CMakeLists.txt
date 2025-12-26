cmake_minimum_required(VERSION 3.25)
message(STATUS "CMAKE VERSION:${CMAKE_VERSION}")

set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES cmake/configure/ConanProvider.cmake)

project(
  videodataset
  LANGUAGES CXX C CUDA
  HOMEPAGE_URL "https://github.com/AgiBot-World/VideoDataset")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ProjectDefault)
include(ConfigureVersion)

# add -pg to profile build
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -g -O2 -fno-omit-frame-pointer")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g -fno-omit-frame-pointer -DNDEBUG")
endif()

add_subdirectory(csrc)
