#include "_core.hpp"
#include "catch2/catch_all.hpp"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <pybind11/embed.h>
#include <sstream>

TEST_METHOD("VideoDecoder.decode", "[VideoDecoder]") {
  pybind11::scoped_interpreter guard{};
}
