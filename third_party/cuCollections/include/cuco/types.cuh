/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
 * limitations under the License.
 */

#pragma once


namespace cuco::detail {

  /**
   * @brief A strong type wrapper
   *
   * @tparam T Type of the value
   *
   */
  template <class T>
  struct strong_type {
    /**
     * @brief Constructs a strong type
     *
     * @param v Value to be wrapped as a strong type
     */
    __host__ __device__ explicit constexpr strong_type(T v) : value{v} {}
  
    /**
     * @brief Implicit conversion operator to the underlying value.
     *
     * @return Underlying value
     */
    __host__ __device__ constexpr operator T() const noexcept { return value; }
  
    T value;  ///< Underlying data value
  };
  
  }  // namespace cuco::detail
  
  /**
   * @brief Convenience wrapper for defining a strong type
   */
  #define CUCO_DEFINE_STRONG_TYPE(Name, Type)                 \
    struct Name : public cuco::detail::strong_type<Type> {    \
      __host__ __device__ explicit constexpr Name(Type value) \
        : cuco::detail::strong_type<Type>(value)              \
      {                                                       \
      }                                                       \
    };
  
  /**
   * @brief Convenience wrapper for defining a templated strong type
   */
  #define CUCO_DEFINE_TEMPLATE_STRONG_TYPE(Name)                                                    \
    template <typename T>                                                                           \
    struct Name : public cuco::detail::strong_type<T> {                                             \
      __host__ __device__ explicit constexpr Name(T value) : cuco::detail::strong_type<T>(value) {} \
    };
  

/**
 * @brief Defines various strong type wrappers used across this library.
 *
 * @note Each strong type inherits from `cuco::detail::strong_type<T>`. `CUCO_DEFINE_STRONG_TYPE`
 * and `CUCO_DEFINE_TEMPLATE_STRONG_TYPE` are convenience macros used to define a named type in a
 * single line, e.g., `CUCO_DEFINE_STRONG_TYPE(foo, double)` defines `struct foo : public
 * cuco::detail::strong_type<double> {...};`, where `cuco::foo{42.0}` is implicitly convertible to
 * `double{42.0}`.
 */

namespace cuco {
/**
 * @brief A strong type wrapper `cuco::empty_key<Key>` used to denote the empty key sentinel.
 */
CUCO_DEFINE_TEMPLATE_STRONG_TYPE(empty_key);

/**
 * @brief A strong type wrapper `cuco::empty_value<T>` used to denote the empty value sentinel.
 */
CUCO_DEFINE_TEMPLATE_STRONG_TYPE(empty_value);

/**
 * @brief A strong type wrapper `cuco::erased_key<Key>` used to denote the erased key sentinel.
 */
CUCO_DEFINE_TEMPLATE_STRONG_TYPE(erased_key);

/**
 * @brief A strong type wrapper `cuco::sketch_size_kb` for specifying the upper-bound sketch size of
 * `cuco::hyperloglog(_ref)` in KB.
 *
 * @note Values can also be specified as literals, e.g., 64.3_KB.
 */
CUCO_DEFINE_STRONG_TYPE(sketch_size_kb, double);

/**
 * @brief A strong type wrapper `cuco::standard_deviation` for specifying the desired standard
 * deviation for the cardinality estimate of `cuco::hyperloglog(_ref)`.
 */
CUCO_DEFINE_STRONG_TYPE(standard_deviation, double);

}  // namespace cuco

// User-defined literal operators for `cuco::sketch_size_KB`
__host__ __device__ constexpr cuco::sketch_size_kb operator""_KB(long double value)
{
  return cuco::sketch_size_kb{static_cast<double>(value)};
}

__host__ __device__ constexpr cuco::sketch_size_kb operator""_KB(unsigned long long int value)
{
  return cuco::sketch_size_kb{static_cast<double>(value)};
}
