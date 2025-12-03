cmake_minimum_required(VERSION 3.25)
message(STATUS "CMAKE VERSION:${CMAKE_VERSION}")

set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES cmake/configure/ConanProvider.cmake)

project(
  fast-context-queue
  LANGUAGES CXX C
  HOMEPAGE_URL "https://code.agibot.com/ai-platform/fast-context-queue")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ProjectDefault)
include(ConfigureVersion)

add_subdirectory(csrc)
