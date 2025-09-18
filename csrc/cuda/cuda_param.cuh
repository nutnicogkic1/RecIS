#pragma once
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>

#include <iostream>

#include "cuda/utils.cuh"
#include "utils/host_pinmem_vector.h"

namespace recis {
namespace cuda {

template <typename T>
class CudaVecParam {
 public:
  using value_type = T;

  CudaVecParam(size_t size, cudaStream_t stream)
      : size_(size), stream_(stream) {
    data_.resize(size);
  }

  ~CudaVecParam() {
    if (cuda_data_) {
      cudaStreamSynchronize(stream_);
      delete_cuda_ptr(cuda_data_);
    }
  }

  // Lazy update: copy to device only when needed
  T* data() {
    if (modify_) {
      if (!cuda_data_) {
        cuda_data_ = cuda_malloc_and_copy<T>(data_.data(), size_, stream_);
      } else {
        C10_CUDA_CHECK(cudaMemcpyAsync(cuda_data_, data_.data(),
                                       size_ * sizeof(T),
                                       cudaMemcpyHostToDevice, stream_));
      }
      modify_ = false;
    }
    return cuda_data_;
  }

  const T* data() const {
    if (modify_) {
      std::cerr << "Device data is stale in const data() access." << std::endl;
      exit(EXIT_FAILURE);
    }
    return cuda_data_;
  }

  T& operator[](size_t pos) {
    modify_ = true;
    return data_[pos];
  }

  const T& operator[](size_t pos) const { return data_[pos]; }

  void resize(size_t new_size) {
    data_.resize(new_size);
    if (cuda_data_) {
      delete_cuda_ptr(cuda_data_);
      cuda_data_ = nullptr;
    }
    size_ = new_size;
    modify_ = true;
  }

  size_t size() const { return size_; }

  CudaVecParam& operator=(const CudaVecParam& other) {
    if (this != &other) {
      this->resize(other.size_);
      std::copy(other.data_.begin(), other.data_.end(), this->data_.begin());
      this->modify_ = true;
    }
    return *this;
  }

  CudaVecParam(const CudaVecParam& other) : size_(0), stream_(other.stream_) {
    *this = other;
  }

 private:
  size_t size_;
  recis::utils::PinMemVector<T> data_;
  T* cuda_data_ = nullptr;
  cudaStream_t stream_;
  bool modify_ = true;
};
}  // namespace cuda
}  // namespace recis
