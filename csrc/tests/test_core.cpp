#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <thread>

#ifdef __unix__
#include <sys/types.h>
#include <sys/wait.h>
#endif

#include "_core.hpp"

TEST_CASE("CircularBuffer Test - Write and Read", "[circular_buffer]") {
    pybind11::scoped_interpreter guard{};
    CircularBuffer buffer;
    auto msgs = py::bytes("Hello");
    REQUIRE(buffer.write(msgs) == Q_SUCCESS);

    auto results = buffer.read();
    REQUIRE(results[1].cast<int>() == Q_SUCCESS);
    REQUIRE(results[0].cast<std::string>() == "Hello");

    auto empty_status = buffer.read();
    REQUIRE(empty_status[1].cast<int>() == Q_EMPTY);
    REQUIRE(empty_status[0].is_none());
}

TEST_CASE("CircularBuffer Test - Queue Full", "[circular_buffer]") {
    pybind11::scoped_interpreter guard{};
    CircularBuffer buffer(20, 1);
    auto msgs = py::bytes("1234567890");
    REQUIRE(buffer.write(msgs) == Q_SUCCESS);
    REQUIRE(buffer.write(msgs) == Q_FULL);
}

TEST_CASE("CircularBuffer Test - Multi Thread", "[circular_buffer]") {
    pybind11::scoped_interpreter guard{};
    CircularBuffer buffer;
    std::vector<std::string> messages;
    messages.emplace_back("ChildMessage1");
    messages.emplace_back("ChildMessage2");
    pybind11::gil_scoped_release release; // Release GIL to allow threads to run concurrently
    // Create a thread for writing messages
    std::thread writer([&buffer, &messages]() {
        pybind11::gil_scoped_acquire acquire; // Acquire GIL to allow Python code to run again
        for (const auto& msg : messages) {
            REQUIRE(buffer.write(py::bytes(msg.c_str())) == Q_SUCCESS);
        }
    });

    // Create a thread for reading messages
    std::thread reader([&buffer, &messages]() {
        pybind11::gil_scoped_acquire acquire; // Acquire GIL to allow Python code to run again
        for (size_t i = 0; i < messages.size(); ++i) {
            auto read_msg = buffer.read();
            if (read_msg[1].cast<int>() == Q_SUCCESS)
                REQUIRE(read_msg[0].cast<std::string>() == messages[i]);
            if (read_msg[1].cast<int>() == Q_EMPTY)
                continue;
        }
    });
    // Join the threads with the main thread
    writer.join();
    reader.join();
}

#ifdef __unix__
TEST_CASE("CircularBuffer Test - Multi Process", "[circular_buffer]") {
    pybind11::scoped_interpreter guard{};
    CircularBuffer buffer;
    const char* msg1 = "ChildMessage";

    // Fork a new process
    pid_t pid = fork();
    if (pid == 0) {
        // Child process
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Wait for parent process to initialize
        REQUIRE(buffer.write(py::bytes(msg1), true, 1) == Q_SUCCESS);
    }
    else if (pid > 0) {
        // Parent process
        // Wait for child process to end
        waitpid(pid, nullptr, 0);

        auto read_msg = buffer.read();
        REQUIRE(read_msg[1].cast<int>() == Q_SUCCESS);
        REQUIRE(read_msg[0].cast<std::string>() == "ChildMessage");
    }
    else {
        // Fork failed
        REQUIRE(false);
    }
}
#endif
