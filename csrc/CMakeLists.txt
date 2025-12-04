set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

find_package(Boost REQUIRED)

execute_process(
  COMMAND ${Python_EXECUTABLE} -c
          "import site;print(';'.join(site.getsitepackages()))"
  OUTPUT_VARIABLE _site_dirs
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _site_cmd_retval
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
if(NOT ${_site_cmd_retval} STREQUAL "0")
  message(FATAL_ERROR "Failed to get site-packages directory")
endif()

message(
  STATUS "${Python_EXECUTABLE} found site-packages directory: ${_site_dirs}")
if(NOT CMAKE_PREFIX_PATH MATCHES "${_site_dirs}")
  set(CMAKE_PREFIX_PATH "${_site_dirs};${CMAKE_PREFIX_PATH}")
endif()

set(target_name _core)
set(target_name_internal ${target_name}_internal)

file(GLOB_RECURSE ${target_name_internal}_srcs "src/*.cpp")
file(GLOB_RECURSE ${target_name_internal}_hdrs "include/*.hpp" "include/*.h")

add_library(${target_name_internal} OBJECT)
warn_target(${target_name_internal})
harden_target(${target_name_internal})
sanitize_target(${target_name_internal})

generate_git_header(VERSION_NAMESPACE_PREFIX ${target_name})

target_sources(${target_name_internal} PRIVATE ${${target_name_internal}_srcs}
                                               ${${target_name_internal}_hdrs})

target_include_interface_directories(
  ${target_name_internal} ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/git_version)

target_link_libraries(
  ${target_name_internal} PUBLIC pybind11::pybind11  boost::boost)

pybind11_add_module(${target_name} MODULE)
warn_target(${target_name})
harden_target(${target_name})
sanitize_target(${target_name})

target_link_libraries(${target_name} PRIVATE ${target_name_internal}
                                             pybind11::module)

string(REPLACE "-" "_" MODULE_NAME ${CMAKE_PROJECT_NAME})
find_program(_STUBGEN stubgen)
if(_STUBGEN)
  add_custom_command(
    TARGET ${target_name}
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E env
      "PYTHONPATH=$<TARGET_FILE_DIR:${target_name}>:$ENV{PYTHONPATH}"
      ${_STUBGEN} -o "${CMAKE_SOURCE_DIR}/src/${MODULE_NAME}" --include-docstrings
      -m ${target_name}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating .pyi stubs for _core")
endif()

install(TARGETS ${target_name} DESTINATION ${MODULE_NAME})

message(
  STATUS
    "Exclude post regexes: x86_64-linux-gnu;/opt;/tmp;/lib64;/usr/local;${_site_dirs}"
)
install_dependency(
  TARGETS
  ${target_name}
  POST_EXCLUDE_REGEXES
  "x86_64-linux-gnu"
  "/opt"
  "/tmp"
  "/lib64"
  "/usr/local"
  ${_site_dirs}
  DEPENDS_DESTINATION
  ${MODULE_NAME})

if(BUILD_TESTING)
  message(STATUS "Enabling testing")
  add_test_subdirectory(tests)
endif()
