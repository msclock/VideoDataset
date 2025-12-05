#include <pybind11/pytypes.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <unordered_map>
#include <vector>

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
        .def(py::init<size_t, size_t, std::string, bool>(),
             py::arg("max_buffer_size") = 10'000'000,
             py::arg("max_size") = 1'000'000'000,
             py::arg("name") = "",
             py::arg("auto_unlink") = true,
             py::doc(R"(Create a circular buffer.

Args:
    max_buffer_size (int): Maximum size of the buffer in bytes.
    max_size (int): Maximum number of elements in the buffer.
    name (str): shared memory name.
    auto_unlink (bool): whether to unlink the shared memory when the buffer is destroyed.)"))
        .def("write",
             &CircularBuffer::write,
             py::arg("msg"),
             py::arg("block") = py::bool_(true),
             py::arg("timeout") = 0.2f,
             py::doc(R"(Put messages into the buffer.

Args:
    msgs (bytes): a message to put into the buffer.
    block (bool): whether to block if the buffer is full.
    timeout (float): timeout in seconds. Default to 0.2 seconds.)"))
        .def("read",
             &CircularBuffer::read,
             py::arg("block") = py::bool_(true),
             py::arg("timeout") = 2.0f,
             py::doc(R"(Get messages from the buffer.

Args:
    block (bool): whether to block if there are no messages in the buffer.
    timeout (float): timeout in seconds. Default to 2.0 seconds.)"))
        .def("get_queue_size", &CircularBuffer::get_queue_size, py::doc("Get the number of elements in the buffer."))
        .def("get_data_size", &CircularBuffer::get_data_size, py::doc("Get the size of the buffer in bytes."))
        .def("is_queue_full", &CircularBuffer::is_queue_full, py::doc("Check if the buffer is full."))
        .def("get_name", &CircularBuffer::get_name, py::doc("Get the name of the buffer."))
        .def("get_max_buffer_size",
             &CircularBuffer::get_max_buffer_size,
             py::doc("Get the maximum size of the buffer in bytes."))
        .def("get_max_size",
             &CircularBuffer::get_max_size,
             py::doc("Get the maximum number of elements in the buffer."))
        .def("get_auto_unlink", &CircularBuffer::get_auto_unlink, py::doc("Get the auto_unlink flag."))
        .def(py::pickle(
            [](const CircularBuffer& p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p.get_max_buffer_size(), p.get_max_size(), p.get_name(), p.get_auto_unlink());
            },
            [](py::tuple& t) { // __setstate__
                if (t.size() != 4)
                    throw std::runtime_error("Invalid state!");

                return CircularBuffer(t[0].cast<size_t>(),
                                      t[1].cast<size_t>(),
                                      t[2].cast<std::string>(),
                                      t[3].cast<bool>());
            }));

    m.attr("Q_SUCCESS") = py::cast(Q_SUCCESS);
    m.attr("Q_EMPTY") = py::cast(Q_EMPTY);
    m.attr("Q_FULL") = py::cast(Q_FULL);
}
