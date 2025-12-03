#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "_core.hpp"

TEST_CASE("CircleByteBuffer Test", "[circular_byte_buffer]") {
    CircularBuffer buffer(1000 * 1000);
}
