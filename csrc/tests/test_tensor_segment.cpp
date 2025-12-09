#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include "tensor_segment.hpp"

#ifdef __unix__
#include <sys/types.h>
#include <sys/wait.h>
#endif

TEST_CASE("TensorSegment Test ", "[TestTensorSegment]") {
    TensorSegment sg;
    auto t = torch::rand({1, 1280, 720});
    auto h = sg.tensor_to_handle(t);
    auto t2 = sg.handle_to_tensor(h);
    REQUIRE(t.equal(t2));
}
