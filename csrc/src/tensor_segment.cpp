#include <cmath>
#include <cstddef>
#include <cstring>

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <torch/expanding_array.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "tensor_segment.hpp"

namespace py = pybind11;
using namespace py::literals;

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
      Pybind11 _core plugin
      -----------------------
      .. currentmodule:: _core
    )pbdoc";

    py::class_<TensorSegment::handle_t>(m, "handle_t")
        .def(py::init<int>())
        .def(py::pickle(
            [](const TensorSegment::handle_t& h) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(static_cast<uint64_t>(h.handle));
            },
            [](py::tuple& t) { // __setstate__
                if (t.size() != 1)
                    throw std::runtime_error("Invalid state!");
                return TensorSegment::handle_t(bip::managed_shared_memory::handle_t(t[0].cast<uint64_t>()));
            }));

    py::class_<TensorSegment>(m,
                              "TorchSegment",
                              R"pydoc(a helper class to save and restore tensors from shared memory.)pydoc")
        .def(py::init<size_t, std::string>(),
             py::arg("pool_size") = POOL_SIZE_DEFAULT,
             py::arg("name") = "",
             py::doc(R"(Create a tensor segment.

Args:
    pool_size (int): the size of the shared memory pool.
    name (str): the name of the shared memory.)"))
        .def(py::pickle(
            [](const TensorSegment& p) { // __getstate__
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p.pool_size(), p.name());
            },
            [](py::tuple& t) { // __setstate__
                if (t.size() != 2)
                    throw std::runtime_error("Invalid state!");

                return TensorSegment(t[0].cast<size_t>(), t[1].cast<std::string>());
            }))
        .def("save_tensor",
             &TensorSegment::tensor_to_handle,
             py::arg("tensor"),
             py::doc(R"(Save a tensor to shared memory.)"))
        .def("restore_tensor",
             &TensorSegment::handle_to_tensor,
             py::keep_alive<0, 1>(),
             py::arg("handle"),
             py::doc(R"(Restore a tensor from shared memory.)"));

    m.attr("POOL_SIZE_DEFAULT") = py::cast(POOL_SIZE_DEFAULT);
}
