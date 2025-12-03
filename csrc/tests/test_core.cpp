#include <pybind11/embed.h>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <sstream>

#include "_core.hpp"

TEST_CASE("VideoDecoder.decode", "[VideoDecoder]") {
    pybind11::scoped_interpreter guard{};
}
