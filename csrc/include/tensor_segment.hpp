#pragma once

#include <ATen/StorageUtils.h>
#include <ATen/core/TensorBody.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/creation_tags.hpp>
#include <boost/interprocess/interprocess_fwd.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/mapped_region.hpp>

namespace bip = boost::interprocess;

constexpr size_t POOL_SIZE_DEFAULT = 1024 * 1024 * 100; // 100M

// TensorSegment is a helper class to save and restore tensors from shared memory.
class TensorSegment {
public:
    struct TensorMeta {
        uint32_t dtype;
        uint32_t ndim;
        int64_t shape[16];
        int64_t stride[16];
    };

    struct handle_t {
        bip::managed_shared_memory::handle_t handle;
        explicit handle_t(bip::managed_shared_memory::handle_t h) : handle(h){};
    };

    explicit TensorSegment(const size_t pool_size = POOL_SIZE_DEFAULT, std::string name = "")
        : pool_size_(pool_size),
          name_(name.empty() ? _safe_base("CircularBuffer") : std::move(name)) {
        segment_ = bip::managed_shared_memory(bip::open_or_create, name_.c_str(), pool_size_);
    };

    TensorSegment(const TensorSegment& o) : TensorSegment(o.pool_size_, o.name_){};

    ~TensorSegment() { bip::shared_memory_object::remove(name_.c_str()); }

    std::string name() const { return name_; }

    size_t pool_size() const { return pool_size_; }

    handle_t tensor_to_handle(const torch::Tensor& t) {
        TORCH_CHECK(t.device().is_cpu(), "Only CPU tensor supported");
        t.contiguous();

        void* buf;
        try {
            buf = segment_.allocate_aligned(sizeof(TensorMeta) + t.nbytes(), 128);
        }
        catch (bip::interprocess_exception& e) {
            TORCH_CHECK(false, "Shared memory pool is full, please increase the pool size");
        }
        auto* meta = new (buf) TensorMeta{};
        meta->dtype = static_cast<uint32_t>(t.scalar_type());
        meta->ndim = static_cast<uint32_t>(t.dim());
        std::memcpy(meta->shape, t.sizes().data(), static_cast<size_t>(t.dim()) * sizeof(int64_t));
        std::memcpy(meta->stride, t.strides().data(), static_cast<size_t>(t.dim()) * sizeof(int64_t));
        std::memcpy(static_cast<char*>(buf) + sizeof(TensorMeta), t.data_ptr(), t.nbytes());
        return handle_t(segment_.get_handle_from_address(buf));
    }

    torch::Tensor handle_to_tensor(handle_t h) {
        void* buf = segment_.get_address_from_handle(h.handle);
        auto* meta = static_cast<TensorMeta*>(buf);
        auto options = torch::TensorOptions()
                           .dtype(static_cast<torch::ScalarType>(meta->dtype))
                           .device(torch::kCPU)
                           .layout(torch::kStrided);
        std::vector<int64_t> sizes(meta->shape, meta->shape + meta->ndim);
        std::vector<int64_t> strides(meta->stride, meta->stride + meta->ndim);
        return torch::from_blob(
                   static_cast<char*>(buf) + sizeof(TensorMeta),
                   sizes,
                   strides,
                   [this, buf](void*) { this->segment_.deallocate(buf); },
                   options)
            .clone();
    }

    static std::string _safe_base(const std::string& prefix) { return prefix + std::to_string(std::rand()); }

private:
    size_t pool_size_;
    std::string name_;
    bip::managed_shared_memory segment_;
};
