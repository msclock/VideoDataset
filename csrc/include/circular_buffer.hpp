#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <utility>

#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

namespace bip = boost::interprocess;

constexpr int Q_SUCCESS = 0, Q_EMPTY = -1, Q_FULL = -2, Q_MSG_BUFFER_TOO_SMALL = -3;

// Logs the error message to stderr and in debug mode triggers assert if the condition is false.
#define LOG_ASSERT(cond, msg)                                     \
    if (!(cond)) {                                                \
        fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, (msg)); \
        assert(((msg), (cond)));                                  \
    }

struct CircularBufferState {
    explicit CircularBufferState(size_t max_size_bytes, size_t maxsize)
        : max_size_bytes(max_size_bytes),
          maxsize(maxsize) {}

    ~CircularBufferState() = default;

    [[nodiscard]] size_t get_max_buffer_size() const { return max_size_bytes; }

    [[nodiscard]] size_t get_max_size() const { return maxsize; }

    [[nodiscard]] bool can_fit(size_t data_size, size_t additional_size) const {
        const bool cond_size = size + data_size <= max_size_bytes;
        const bool cond_num = num_elem + additional_size <= maxsize;

        return cond_size && cond_num;
    }

    void circular_buffer_write(uint8_t *buffer, const uint8_t *data, const size_t data_size) {
        if (tail + data_size < max_size_bytes) {
            memcpy(buffer + tail, data, data_size);
            tail += data_size;
        }
        else {
            const auto before_wrap = max_size_bytes - tail, after_wrap = data_size - before_wrap;
            memcpy(buffer + tail, data, before_wrap);
            memcpy(buffer, data + before_wrap, after_wrap);
            tail = after_wrap;
        }

        size += data_size;

        LOG_ASSERT(size <= max_size_bytes, "Combined message size exceeds the size of the queue");
        LOG_ASSERT(tail < max_size_bytes, "Tail pointer points past the buffer boundary");
    }

    void circular_buffer_read(uint8_t *buffer, uint8_t *data, size_t read_size, bool pop_message) {
        size_t new_head;

        if (head + read_size < max_size_bytes) {
            memcpy(data, buffer + head, read_size);
            new_head = head + read_size;
        }
        else {
            const auto before_wrap = max_size_bytes - head, after_wrap = read_size - before_wrap;
            memcpy(data, buffer + head, before_wrap);
            memcpy(data + before_wrap, buffer, after_wrap);
            new_head = after_wrap;
        }

        const auto new_size = size - read_size;

        LOG_ASSERT(new_head < max_size_bytes, "Circular buffer head pointer is incorrect");
        LOG_ASSERT(new_size >= 0 && new_size <= max_size_bytes, "New size is incorrect after reading from buffer");

        if (pop_message) {
            head = new_head;
            size = new_size;
        }
    }

public:
    static const size_t MIN_MSG_SIZE = sizeof(size_t) + 1;
    size_t max_size_bytes;
    size_t maxsize;
    size_t head = 0, tail = 0, size = 0;
    size_t num_elem = 0;

    boost::interprocess::interprocess_mutex mutex;
    int not_empty_n_waiters = 0, not_full_n_waiters = 0;
    boost::interprocess::interprocess_condition not_empty;
    boost::interprocess::interprocess_condition not_full;
};

inline std::chrono::microseconds float_seconds_to_chrono(float seconds) {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::duration<float>(seconds));
}

inline std::chrono::microseconds wait(std::chrono::microseconds wait_time,
                                      boost::interprocess::interprocess_condition *cond,
                                      boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> *lock,
                                      int *waiter_count) {
    auto tp = std::chrono::steady_clock::now() + wait_time;

    ++(*waiter_count);
    cond->wait_until(*lock, tp);
    --(*waiter_count);
    auto remaining_us = std::chrono::duration_cast<std::chrono::microseconds>(tp - std::chrono::steady_clock::now());
    return remaining_us > std::chrono::microseconds::zero() ? remaining_us : std::chrono::microseconds::zero();
}

class CircularBuffer {
public:
    explicit CircularBuffer(size_t max_size_bytes = 10'000'000,
                            size_t maxsize = 1'000'000'000,
                            std::string name = "",
                            bool create = true,
                            bool auto_unlink = true)
        : name_(name.empty() ? _safe_base("CircularBuffer") : std::move(name)),
          auto_unlink_(auto_unlink) {
        std::string state_name = name_ + "S_";
        std::string buf_name = name_ + "B_";

        constexpr size_t queue_size = sizeof(CircularBufferState);

        if (create) {
            try {
                state_mem_ =
                    std::make_shared<bip::shared_memory_object>(bip::create_only, state_name.c_str(), bip::read_write);
                state_mem_->truncate(queue_size);
                state_region_ = bip::mapped_region(*state_mem_, bip::read_write);
                state_ = static_cast<CircularBufferState *>(state_region_.get_address());
                new (state_) CircularBufferState(max_size_bytes, maxsize);
            }
            catch (const bip::interprocess_exception &ex) {
                state_mem_ =
                    std::make_shared<bip::shared_memory_object>(bip::open_only, state_name.c_str(), bip::read_write);
                state_region_ = bip::mapped_region(*state_mem_, bip::read_write);
                state_ = static_cast<CircularBufferState *>(state_region_.get_address());
                new (state_) CircularBufferState(max_size_bytes, maxsize);
            }
        }
        else {
            state_mem_ =
                std::make_shared<bip::shared_memory_object>(bip::open_only, state_name.c_str(), bip::read_write);
            state_region_ = bip::mapped_region(*state_mem_, bip::read_write);
            state_ = static_cast<CircularBufferState *>(state_region_.get_address());
        }

        if (create) {
            try {
                buf_mem_ =
                    std::make_shared<bip::shared_memory_object>(bip::create_only, buf_name.c_str(), bip::read_write);
                buf_mem_->truncate(static_cast<boost::interprocess::offset_t>(max_size_bytes));
                buf_region_ = bip::mapped_region(*buf_mem_, bip::read_write);
                buf_ = static_cast<char *>(buf_region_.get_address());
                std::fill_n(buf_, max_size_bytes, 0);
            }
            catch (const bip::interprocess_exception &ex) {
                buf_mem_ =
                    std::make_shared<bip::shared_memory_object>(bip::open_only, buf_name.c_str(), bip::read_write);
                buf_region_ = bip::mapped_region(*buf_mem_, bip::read_write);
                buf_ = static_cast<char *>(buf_region_.get_address());
            }
        }
        else {
            buf_mem_ = std::make_shared<bip::shared_memory_object>(bip::open_only, buf_name.c_str(), bip::read_write);
            buf_region_ = bip::mapped_region(*buf_mem_, bip::read_write);
            buf_ = static_cast<char *>(buf_region_.get_address());
        }
    }

    ~CircularBuffer() {
        if (auto_unlink_) {
            if (state_mem_) {
                state_mem_->remove((name_ + "S_").c_str());
            }
            if (buf_mem_) {
                buf_mem_->remove((name_ + "B_").c_str());
            }
        }
    }

    int queue_put(const void **msgs_data,
                  const size_t *msg_sizes,
                  const size_t num_msgs,
                  const int block,
                  const float timeout) {
        boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(state_->mutex);

        size_t total_size = num_msgs * sizeof(size_t);
        for (size_t i = 0; i < num_msgs; ++i)
            total_size += msg_sizes[i];

        auto wait_remaining = float_seconds_to_chrono(timeout);
        while (!state_->can_fit(total_size, num_msgs)) {
            if (!block || wait_remaining.count() <= 0)
                return Q_FULL;

            // If there are any consumers waiting, wake them up!
            if (state_->not_empty_n_waiters > 0)
                state_->not_empty.notify_one();

            wait_remaining = wait(wait_remaining, &state_->not_full, &lock, &state_->not_full_n_waiters);
        }

        for (size_t i = 0; i < num_msgs; ++i) {
            state_->circular_buffer_write((uint8_t *)this->buf_, (const uint8_t *)(msg_sizes + i), sizeof(size_t));
            state_->circular_buffer_write((uint8_t *)this->buf_, (const uint8_t *)(msgs_data[i]), msg_sizes[i]);
            ++state_->num_elem;
        }

        if (state_->not_empty_n_waiters > 0)
            state_->not_empty.notify_one();
        else if (state_->not_full_n_waiters && state_->can_fit(CircularBufferState::MIN_MSG_SIZE, 1))
            state_->not_full.notify_one();

        return Q_SUCCESS;
    }

    int queue_get(void *msg_buffer,
                  size_t msg_buffer_size,
                  size_t max_messages_to_get,
                  size_t max_bytes_to_get,
                  size_t *messages_read,
                  size_t *bytes_read,
                  size_t *messages_size,
                  int block,
                  float timeout) {
        *messages_read = *bytes_read = *messages_size = 0;
        boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> lock(state_->mutex);

        auto wait_remaining = float_seconds_to_chrono(timeout);
        while (state_->size <= 0) {
            if (!block || wait_remaining.count() <= 0)
                return Q_EMPTY;

            wait_remaining = wait(wait_remaining, &state_->not_empty, &lock, &state_->not_empty_n_waiters);
        }

        auto status = Q_SUCCESS;
        while (*messages_read < max_messages_to_get && *bytes_read < max_bytes_to_get) {
            size_t msg_size;
            state_->circular_buffer_read((uint8_t *)this->buf_, (uint8_t *)&msg_size, sizeof(msg_size), false);
            *messages_size += sizeof(msg_size) + msg_size;

            if (msg_buffer_size < *messages_size) {
                status = Q_MSG_BUFFER_TOO_SMALL;
                break;
            }

            LOG_ASSERT(state_->size >= sizeof(msg_size) + msg_size, "Queue size is less than message size!");

            const auto read_num_bytes = sizeof(msg_size) + msg_size;
            state_->circular_buffer_read((uint8_t *)this->buf_,
                                         (uint8_t *)msg_buffer + *bytes_read,
                                         read_num_bytes,
                                         true);

            *bytes_read += read_num_bytes;
            *messages_read += 1;
            --state_->num_elem;

            if (state_->size <= 0) {
                break;
            }
        }

        if (*messages_read > 0 && state_->not_full_n_waiters > 0)
            state_->not_full.notify_one();
        else if (state_->size > 0 && state_->not_empty_n_waiters > 0)
            state_->not_empty.notify_one();

        return status;
    }

    size_t get_queue_size() { return state_->num_elem; }

    size_t get_data_size() { return state_->size; }

    bool is_queue_full() {
        constexpr size_t min_message_size = 1;
        constexpr size_t min_messages_count = 1;
        return !state_->can_fit(min_message_size + sizeof(min_message_size), min_messages_count);
    }

private:
    static std::string _safe_base(const std::string &prefix) { return prefix + std::to_string(std::rand()); }

    std::string name_;
    bool auto_unlink_;

    std::shared_ptr<bip::shared_memory_object> state_mem_;
    bip::mapped_region state_region_;
    CircularBufferState *state_ = nullptr;

    std::shared_ptr<bip::shared_memory_object> buf_mem_;
    bip::mapped_region buf_region_;
    char *buf_ = nullptr;
};
