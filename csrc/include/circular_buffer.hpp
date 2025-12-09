#pragma once

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <sys/types.h>

#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

namespace bip = boost::interprocess;
namespace py = pybind11;

constexpr int Q_SUCCESS = 0, Q_EMPTY = -1, Q_FULL = -2;

// Logs the error message to stderr and in debug mode triggers assert if the condition is false.
#define LOG_ASSERT(cond, msg)                                     \
    if (!(cond)) {                                                \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, (msg)); \
        assert((cond));                                           \
    }

struct CircularBufferState {
    explicit CircularBufferState(size_t max_buffer_size, size_t maxsize)
        : max_buffer_size(max_buffer_size),
          maxsize(maxsize) {}

    ~CircularBufferState() = default;

    size_t get_max_buffer_size() const { return max_buffer_size; }

    size_t get_max_size() const { return maxsize; }

    bool is_fit(size_t data_size, size_t additional_size) const {
        const bool cond_size = size + data_size <= max_buffer_size;
        const bool cond_num = num_elem + additional_size <= maxsize;

        return cond_size && cond_num;
    }

    void buffer_write(uint8_t *buffer, const uint8_t *data, const size_t data_size) {
        if (tail + data_size < max_buffer_size) {
            memcpy(buffer + tail, data, data_size);
            tail += data_size;
        }
        else {
            const auto before_wrap = max_buffer_size - tail, after_wrap = data_size - before_wrap;
            memcpy(buffer + tail, data, before_wrap);
            memcpy(buffer, data + before_wrap, after_wrap);
            tail = after_wrap;
        }

        size += data_size;

        LOG_ASSERT(size <= max_buffer_size, "Combined message size exceeds the size of the queue");
        LOG_ASSERT(tail < max_buffer_size, "Tail pointer points past the buffer boundary");
    }

    void buffer_read(uint8_t *buffer, uint8_t *data, size_t read_size, bool pop_message) {
        size_t new_head;

        if (head + read_size < max_buffer_size) {
            memcpy(data, buffer + head, read_size);
            new_head = head + read_size;
        }
        else {
            const auto before_wrap = max_buffer_size - head, after_wrap = read_size - before_wrap;
            memcpy(data, buffer + head, before_wrap);
            memcpy(data + before_wrap, buffer, after_wrap);
            new_head = after_wrap;
        }

        const auto new_size = size - read_size;

        LOG_ASSERT(new_head < max_buffer_size, "Circular buffer head pointer is incorrect");
        LOG_ASSERT(new_size >= 0 && new_size <= max_buffer_size, "New size is incorrect after reading from buffer");

        if (pop_message) {
            head = new_head;
            size = new_size;
        }
    }

public:
    static const size_t MIN_MSG_SIZE = sizeof(size_t) + 1;
    size_t max_buffer_size;
    size_t maxsize;
    size_t head = 0, tail = 0, size = 0;
    size_t num_elem = 0;

    int not_empty_n_waiters = 0, not_full_n_waiters = 0;
    boost::interprocess::interprocess_mutex mutex;
    boost::interprocess::interprocess_condition not_empty;
    boost::interprocess::interprocess_condition not_full;
};

class CircularBuffer {
public:
    explicit CircularBuffer(size_t max_buffer_size = 10'000'000,
                            size_t maxsize = 1'000'000'000,
                            std::string name = "",
                            bool auto_unlink = true)
        : name_(name.empty() ? _safe_base("CircularBuffer") : std::move(name)),
          auto_unlink_(auto_unlink) {
        std::string state_name = name_ + "S_";
        std::string buf_name = name_ + "B_";

        constexpr size_t queue_size = sizeof(CircularBufferState);

        // try {
        state_mem_ =
            std::make_shared<bip::shared_memory_object>(bip::open_or_create, state_name.c_str(), bip::read_write);
        state_mem_->truncate(queue_size);
        state_region_ = bip::mapped_region(*state_mem_, bip::read_write);
        state_ = static_cast<CircularBufferState *>(state_region_.get_address());
        new (state_) CircularBufferState(max_buffer_size, maxsize);
        // }
        // catch (const bip::interprocess_exception &ex) {
        //     state_mem_ =
        //         std::make_shared<bip::shared_memory_object>(bip::open_only, state_name.c_str(), bip::read_write);
        //     state_region_ = bip::mapped_region(*state_mem_, bip::read_write);
        //     state_ = static_cast<CircularBufferState *>(state_region_.get_address());
        //     new (state_) CircularBufferState(max_buffer_size, maxsize);
        // }

        // try {
        buf_mem_ = std::make_shared<bip::shared_memory_object>(bip::open_or_create, buf_name.c_str(), bip::read_write);
        buf_mem_->truncate(static_cast<boost::interprocess::offset_t>(max_buffer_size));
        buf_region_ = bip::mapped_region(*buf_mem_, bip::read_write);
        buf_ = static_cast<uint8_t *>(buf_region_.get_address());
        // std::fill_n(buf_, max_buffer_size, 0);
        // }
        // catch (const bip::interprocess_exception &ex) {
        //     buf_mem_ = std::make_shared<bip::shared_memory_object>(bip::open_only, buf_name.c_str(),
        //     bip::read_write); buf_region_ = bip::mapped_region(*buf_mem_, bip::read_write); buf_ =
        //     static_cast<uint8_t *>(buf_region_.get_address());
        // }
    }

    ~CircularBuffer() {
        if (auto_unlink_) {
            if (state_mem_) {
                state_mem_->remove((name_ + "S_").c_str());
                state_mem_ = nullptr;
            }
            if (buf_mem_) {
                buf_mem_->remove((name_ + "B_").c_str());
                buf_mem_ = nullptr;
            }
        }
    }

    CircularBuffer(const CircularBuffer &o)
        : CircularBuffer(o.get_max_buffer_size(), o.get_max_size(), o.name_, o.auto_unlink_) {}

    int write(const py::bytes &msg, const int block = true, float timeout = 0.2f) {
        boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(state_->mutex);
        const std::string_view msg_view = msg.cast<std::string_view>();
        const auto msg_size = sizeof(size_t) + msg_view.size() * sizeof(uint8_t);

        auto wait_remaining = float_seconds_to_chrono(timeout);
        while (!state_->is_fit(msg_size, 1)) {
            if (!block || wait_remaining.count() <= 0)
                return Q_FULL;

            // If there are any consumers waiting, wake them up!
            if (state_->not_empty_n_waiters > 0)
                state_->not_empty.notify_one();

            wait_remaining = wait(wait_remaining, &state_->not_full, &lock, &state_->not_full_n_waiters);
        }

        auto msg_len = msg_view.size();
        state_->buffer_write(this->buf_, reinterpret_cast<const uint8_t *>(&msg_len), sizeof(size_t));
        state_->buffer_write(this->buf_, reinterpret_cast<const uint8_t *>(msg_view.data()), msg_view.size());
        ++state_->num_elem;

        if (state_->not_empty_n_waiters > 0)
            state_->not_empty.notify_one();
        else if (state_->not_full_n_waiters && state_->is_fit(CircularBufferState::MIN_MSG_SIZE, 1))
            state_->not_full.notify_one();

        return Q_SUCCESS;
    }

    py::typing::Tuple<py::bytes, int> read(bool block = true, float timeout = 2.0f) {
        boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(state_->mutex);
        auto wait_remaining = float_seconds_to_chrono(timeout);
        int wait_count = 0;
        while (state_->size <= 0) {
            if (!block || wait_remaining.count() <= 0)
                return py::make_tuple(py::none(), Q_EMPTY);
            wait_count++;
            wait_remaining = wait(wait_remaining, &state_->not_empty, &lock, &state_->not_empty_n_waiters);
        }

        size_t msg_size;
        state_->buffer_read(this->buf_, reinterpret_cast<uint8_t *>(&msg_size), sizeof(size_t), false);

        LOG_ASSERT(state_->size >= sizeof(size_t) + msg_size, "Queue size is less than message size!");

        const auto read_num_bytes = sizeof(size_t) + msg_size;
        std::vector<uint8_t> msg_buffer(msg_size + 10);
        state_->buffer_read(this->buf_, msg_buffer.data(), read_num_bytes, true);
        auto msg = py::bytes(reinterpret_cast<const char *>(msg_buffer.data() + sizeof(size_t)), msg_size);
        --state_->num_elem;

        if (state_->not_full_n_waiters > 0)
            state_->not_full.notify_one();
        else if (state_->size > 0 && state_->not_empty_n_waiters > 0)
            state_->not_empty.notify_one();

        return py::make_tuple(msg, Q_SUCCESS);
    }

    size_t get_queue_size() { return state_->num_elem; }

    size_t get_data_size() { return state_->size; }

    bool is_queue_full() {
        constexpr size_t min_message_size = 1;
        constexpr size_t min_messages_count = 1;
        return !state_->is_fit(min_message_size + sizeof(min_message_size), min_messages_count);
    }

    std::string get_name() const noexcept { return name_; }

    size_t get_max_buffer_size() const noexcept { return state_->get_max_buffer_size(); }

    size_t get_max_size() const noexcept { return state_->get_max_size(); }

    bool get_auto_unlink() const noexcept { return auto_unlink_; }

private:
    static std::string _safe_base(const std::string &prefix) { return prefix + std::to_string(std::rand()); }

    static std::chrono::microseconds float_seconds_to_chrono(float seconds) {
        return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::duration<float>(seconds));
    }

    static std::chrono::microseconds wait(
        std::chrono::microseconds wait_time,
        boost::interprocess::interprocess_condition *cond,
        boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> *lock,
        int *waiter_count) {
        auto tp = std::chrono::steady_clock::now() + wait_time;

        ++(*waiter_count);
        cond->wait_until(*lock, tp);
        --(*waiter_count);
        auto remaining_us =
            std::chrono::duration_cast<std::chrono::microseconds>(tp - std::chrono::steady_clock::now());
        return remaining_us > std::chrono::microseconds::zero() ? remaining_us : std::chrono::microseconds::zero();
    }

    std::string name_;
    bool auto_unlink_;

    std::shared_ptr<bip::shared_memory_object> state_mem_;
    bip::mapped_region state_region_;
    CircularBufferState *state_ = nullptr;

    std::shared_ptr<bip::shared_memory_object> buf_mem_;
    bip::mapped_region buf_region_;
    uint8_t *buf_ = nullptr;
};
