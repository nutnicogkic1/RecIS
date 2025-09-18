/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 */

#pragma once

#include <cuco/detail/__config>
#include <cuco/detail/bitwise_compare.cuh>

#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <cstddef>

#if defined(CUCO_DISABLE_KERNEL_VISIBILITY_WARNING_SUPPRESSION)
#define CUCO_SUPPRESS_KERNEL_WARNINGS
#elif defined(__NVCC__) && (defined(__GNUC__) || defined(__clang__))
// handle when nvcc is the CUDA compiler and gcc or clang is host
#define CUCO_SUPPRESS_KERNEL_WARNINGS _Pragma("nv_diag_suppress 1407")
_Pragma("GCC diagnostic ignored \"-Wattributes\"")
#elif defined(__clang__)
// handle when clang is the CUDA compiler
#define CUCO_SUPPRESS_KERNEL_WARNINGS _Pragma("clang diagnostic ignored \"-Wattributes\"")
#elif defined(__NVCOMPILER)
#define CUCO_SUPPRESS_KERNEL_WARNINGS #pragma diag_suppress attribute_requires_external_linkage
#endif

#ifndef CUCO_KERNEL
#define CUCO_KERNEL __attribute__((visibility("hidden"))) __global__
#endif


#include <cuco/detail/error.hpp>

namespace cuco {
/**
 * @brief A device allocator using `cudaMalloc`/`cudaFree` to satisfy (de)allocations.
 *
 * @tparam T The allocator's value type
 */
template <typename T>
class cuda_allocator {
 public:
  using value_type = T;  ///< Allocator's value type

  cuda_allocator() = default;

  /**
   * @brief Copy constructor.
   */
  template <class U>
  cuda_allocator(cuda_allocator<U> const&) noexcept
  {
  }

  /**
   * @brief Allocates storage for `n` objects of type `T` using `cudaMalloc`.
   *
   * @param n The number of objects to allocate storage for
   * @return Pointer to the allocated storage
   */
  value_type* allocate(std::size_t n)
  {
    value_type* p;
    CUCO_CUDA_TRY(cudaMalloc(&p, sizeof(value_type) * n));
    return p;
  }

  /**
   * @brief Deallocates storage pointed to by `p`.
   *
   * @param p Pointer to memory to deallocate
   */
  void deallocate(value_type* p, std::size_t) { CUCO_CUDA_TRY(cudaFree(p)); }
};

/**
 * @brief Equality comparison operator.
 *
 * @tparam T Value type of LHS object
 * @tparam U Value type of RHS object
 *
 * @return `true` iff given arguments are equal
 */
template <typename T, typename U>
bool operator==(cuda_allocator<T> const&, cuda_allocator<U> const&) noexcept
{
  return true;
}

/**
 * @brief Inequality comparison operator.
 *
 * @tparam T Value type of LHS object
 * @tparam U Value type of RHS object
 *
 * @param lhs Left-hand side object to compare
 * @param rhs Right-hand side object to compare
 *
 * @return `true` iff given arguments are not equal
 */
template <typename T, typename U>
bool operator!=(cuda_allocator<T> const& lhs, cuda_allocator<U> const& rhs) noexcept
{
  return not(lhs == rhs);
}

}  // namespace cuco


namespace cuco {
namespace detail {

/**
 * @brief Converts pair to `cuda::std::tuple` to allow assigning to a zip iterator.
 *
 * @tparam Key The slot key type
 * @tparam Value The slot value type
 */
template <typename Key, typename Value>
struct slot_to_tuple {
  /**
   * @brief Converts a pair to a `cuda::std::tuple`.
   *
   * @tparam S The slot type
   *
   * @param s The slot to convert
   * @return A cuda::std::tuple containing `s.first` and `s.second`
   */
  template <typename S>
  __device__ cuda::std::tuple<Key, Value> operator()(S const& s)
  {
    return cuda::std::tuple<Key, Value>(s.first, s.second);
  }
};

/**
 * @brief Device functor returning whether the input slot `s` is filled.
 *
 * @tparam Key The slot key type
 */
template <typename Key>
struct slot_is_filled {
  Key empty_key_sentinel_;  ///< The value of the empty key sentinel

  /**
   * @brief Indicates if the target slot `s` is filled.
   *
   * @tparam S The slot type
   *
   * @param s The slot to query
   * @return `true` if slot `s` is filled
   */
  template <typename S>
  __device__ bool operator()(S const& s)
  {
    return not cuco::detail::bitwise_compare(thrust::get<0>(s), empty_key_sentinel_);
  }
};

}  // namespace detail
}  // namespace cuco
