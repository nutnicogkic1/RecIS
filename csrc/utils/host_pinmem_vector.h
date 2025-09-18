#include <ATen/cuda/CachingHostAllocator.h>
#include <torch/torch.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
namespace recis {
namespace utils {

template <typename T>
class PinMemVector {
 private:
  c10::DataPtr data_ptr_;
  T* data_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;
  torch::Allocator* allocator_;

  void reallocate(size_t new_capacity) {
    if (new_capacity <= capacity_ && new_capacity >= size_) {
      if (new_capacity < size_) {
        for (size_t i = new_capacity; i < size_; ++i) {
          (data_ + i)->~T();
        }
        size_ = new_capacity;
      }

      if (new_capacity > capacity_) {
      } else {
        return;
      }
    }

    if (new_capacity < size_) {
      for (size_t i = new_capacity; i < size_; ++i) {
        (data_ + i)->~T();
      }
    }

    c10::DataPtr new_data_ptr = allocator_->allocate(new_capacity * sizeof(T));
    T* new_data = static_cast<T*>(new_data_ptr.get());

    if (data_) {
      if constexpr (std::is_nothrow_move_constructible_v<T>) {
        for (size_t i = 0; i < size_; ++i) {
          new (new_data + i) T(std::move(data_[i]));
          (data_ + i)->~T();
        }
      } else {
        for (size_t i = 0; i < size_; ++i) {
          new (new_data + i) T(data_[i]);
          (data_ + i)->~T();
        }
      }
    }

    data_ptr_ = std::move(new_data_ptr);
    data_ = new_data;
    capacity_ = new_capacity;
  }

  void destroy_elements() {
    if (data_) {
      for (size_t i = 0; i < size_; ++i) {
        (data_ + i)->~T();
      }
    }
  }

 public:
  PinMemVector() : allocator_(at::cuda::getCachingHostAllocator()) {}

  explicit PinMemVector(size_t count)
      : allocator_(at::cuda::getCachingHostAllocator()),
        size_(count),
        capacity_(count) {
    if (count > 0) {
      data_ptr_ = allocator_->allocate(capacity_ * sizeof(T));
      data_ = static_cast<T*>(data_ptr_.get());
      for (size_t i = 0; i < size_; ++i) {
        new (data_ + i) T();
      }
    }
  }

  PinMemVector(size_t count, const T& value)
      : allocator_(at::cuda::getCachingHostAllocator()),
        size_(count),
        capacity_(count) {
    if (count > 0) {
      data_ptr_ = allocator_->allocate(capacity_ * sizeof(T));
      data_ = static_cast<T*>(data_ptr_.get());
      for (size_t i = 0; i < size_; ++i) {
        new (data_ + i) T(value);
      }
    }
  }

  PinMemVector(std::initializer_list<T> init)
      : allocator_(at::cuda::getCachingHostAllocator()),
        size_(init.size()),
        capacity_(init.size()) {
    if (capacity_ > 0) {
      data_ptr_ = allocator_->allocate(capacity_ * sizeof(T));
      data_ = static_cast<T*>(data_ptr_.get());
      size_t i = 0;
      for (const auto& item : init) {
        new (data_ + i) T(item);
        i++;
      }
    }
  }

  // Copy constructor
  PinMemVector(const PinMemVector& other)
      : allocator_(at::cuda::getCachingHostAllocator()),
        size_(other.size_),
        capacity_(other.capacity_) {
    if (capacity_ > 0) {
      data_ptr_ = allocator_->allocate(capacity_ * sizeof(T));
      data_ = static_cast<T*>(data_ptr_.get());
      for (size_t i = 0; i < size_; ++i) {
        new (data_ + i) T(other.data_[i]);
      }
    }
  }

  // Move constructor
  PinMemVector(PinMemVector&& other) noexcept
      : data_ptr_(std::move(other.data_ptr_)),
        data_(other.data_),
        size_(other.size_),
        capacity_(other.capacity_),
        allocator_(other.allocator_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  ~PinMemVector() { destroy_elements(); }

  PinMemVector& operator=(const PinMemVector& other) {
    if (this == &other) return *this;

    destroy_elements();

    if (capacity_ < other.size_) {
      data_ptr_ = allocator_->allocate(other.capacity_ * sizeof(T));
      data_ = static_cast<T*>(data_ptr_.get());
      capacity_ = other.capacity_;
    }

    size_ = other.size_;
    for (size_t i = 0; i < size_; ++i) {
      new (data_ + i) T(other.data_[i]);
    }
    return *this;
  }

  PinMemVector& operator=(PinMemVector&& other) noexcept {
    if (this == &other) return *this;

    destroy_elements();

    data_ptr_ = std::move(other.data_ptr_);
    data_ = other.data_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    allocator_ = other.allocator_;

    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
    return *this;
  }

  T& operator[](size_t pos) { return data_[pos]; }
  const T& operator[](size_t pos) const { return data_[pos]; }
  T& at(size_t pos) {
    if (pos >= size_) {
      throw std::out_of_range("PinMemVector::at");
    }
    return data_[pos];
  }
  const T& at(size_t pos) const {
    if (pos >= size_) {
      throw std::out_of_range("PinMemVector::at");
    }
    return data_[pos];
  }
  T* data() noexcept { return data_; }
  const T* data() const noexcept { return data_; }

  T* begin() noexcept { return data_; }
  const T* begin() const noexcept { return data_; }
  const T* cbegin() const noexcept { return data_; }

  T* end() noexcept { return data_ + size_; }
  const T* end() const noexcept { return data_ + size_; }
  const T* cend() const noexcept { return data_ + size_; }

  bool empty() const noexcept { return size_ == 0; }
  size_t size() const noexcept { return size_; }
  size_t capacity() const noexcept { return capacity_; }

  void reserve(size_t new_cap) {
    if (new_cap > capacity_) {
      reallocate(new_cap);
    }
  }

  void clear() noexcept {
    destroy_elements();
    size_ = 0;
  }

  void push_back(const T& value) {
    if (size_ == capacity_) {
      reserve(capacity_ == 0 ? 1 : capacity_ * 2);
    }
    new (data_ + size_) T(value);
    size_++;
  }

  void push_back(T&& value) {
    if (size_ == capacity_) {
      reserve(capacity_ == 0 ? 1 : capacity_ * 2);
    }
    new (data_ + size_) T(std::move(value));
    size_++;
  }

  template <typename... Args>
  T& emplace_back(Args&&... args) {
    if (size_ == capacity_) {
      reserve(capacity_ == 0 ? 1 : capacity_ * 2);
    }
    new (data_ + size_) T(std::forward<Args>(args)...);
    return data_[size_++];
  }

  void pop_back() {
    if (size_ > 0) {
      size_--;
      (data_ + size_)->~T();
    }
  }

  void resize(size_t count) {
    if (count < size_) {
      for (size_t i = count; i < size_; ++i) {
        (data_ + i)->~T();
      }
    } else if (count > size_) {
      if (count > capacity_) {
        reserve(count);
      }
      for (size_t i = size_; i < count; ++i) {
        new (data_ + i) T();
      }
    }
    size_ = count;
  }

  void resize(size_t count, const T& value) {
    if (count < size_) {
      for (size_t i = count; i < size_; ++i) {
        (data_ + i)->~T();
      }
    } else if (count > size_) {
      if (count > capacity_) {
        reserve(count);
      }
      for (size_t i = size_; i < count; ++i) {
        new (data_ + i) T(value);
      }
    }
    size_ = count;
  }
};

}  // namespace utils

}  // namespace recis
