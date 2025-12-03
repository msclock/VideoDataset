#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "_core.hpp"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
      Pybind11 _core plugin
      -----------------------
      .. currentmodule:: _core
    )pbdoc";

    py::class_<CircularBuffer>(m, "CircularBuffer", R"pydoc(A circular buffer for process communication.)pydoc")
        .def(py::init<size_t, size_t, std::string, bool, bool>(),
             py::arg("max_byte_size") = 10'000'000,
             py::arg("max_size") = 1'000'000'000,
             py::arg("name") = "",
             py::arg("create") = true,
             py::arg("auto_unlink") = true,
             py::doc(R"(Create a circular buffer.

Args:
    max_byte_size (int): Maximum size of the buffer in bytes.
    max_size (int): Maximum number of elements in the buffer.
    name (str): shared memory name.
    create (bool): whether to create first.
    auto_unlink (bool): whether to unlink the shared memory when the buffer is destroyed.)"))
        .def(
            "queue_put",
            [](CircularBuffer &self,
               py::buffer msgs_data,
               py::buffer msg_sizes,
               size_t num_msgs,
               int block,
               float timeout) {
                py::buffer_info msgs_data_info = msgs_data.request();
                py::buffer_info msg_sizes_info = msg_sizes.request();
                self.queue_put(static_cast<const void **>(msgs_data_info.ptr),
                               static_cast<const size_t *>(msg_sizes_info.ptr),
                               num_msgs,
                               block,
                               timeout);
            },
            py::arg("msgs_data"),
            py::arg("msg_sizes"),
            py::arg("num_msgs"),
            py::arg("block"),
            py::arg("timeout"),
            py::doc(R"(Put messages into the buffer.)"))
        .def(
            "queue_get",
            [](CircularBuffer &self,
               py::buffer msg_buffer,
               size_t msg_buffer_size,
               size_t max_messages_to_get,
               size_t max_bytes_to_get,
               py::buffer message_read,
               py::buffer bytes_read,
               py::buffer messages_size,
               int block,
               float timeout) {
                return self.queue_get(static_cast<void *>(msg_buffer.request().ptr),
                                      msg_buffer_size,
                                      max_messages_to_get,
                                      max_bytes_to_get,
                                      static_cast<size_t *>(message_read.request().ptr),
                                      static_cast<size_t *>(bytes_read.request().ptr),
                                      static_cast<size_t *>(messages_size.request().ptr),
                                      block,
                                      timeout);
            },
            py::arg("msg_buffer"),
            py::arg("msg_buffer_size"),
            py::arg("max_messages_to_get"),
            py::arg("max_bytes_to_get"),
            py::arg("message_read"),
            py::arg("bytes_read"),
            py::arg("messages_size"),
            py::arg("block"),
            py::arg("timeout"),
            py::doc(R"(Get messages from the buffer.)"))
        .def("get_queue_size", &CircularBuffer::get_queue_size)
        .def("get_data_size", &CircularBuffer::get_data_size)
        .def("is_queue_full", &CircularBuffer::is_queue_full);

    m.attr("Q_SUCCESS") = py::cast(Q_SUCCESS);
    m.attr("Q_EMPTY") = py::cast(Q_EMPTY);
    m.attr("Q_FULL") = py::cast(Q_FULL);
    m.attr("Q_MSG_BUFFER_TOO_SMALL") = py::cast(Q_MSG_BUFFER_TOO_SMALL);
}
