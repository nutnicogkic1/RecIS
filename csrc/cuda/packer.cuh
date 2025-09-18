#pragma once
#include <cuda_runtime.h>

#include <type_traits>
namespace recis {
namespace cuda {
#define CUDA_PACK_SIZE 16
template <typename A, typename B>
struct select_pack_size {
  static constexpr int value =
      CUDA_PACK_SIZE / sizeof(A) < CUDA_PACK_SIZE / sizeof(B)
          ? CUDA_PACK_SIZE / sizeof(A)
          : CUDA_PACK_SIZE / sizeof(B);
};

template <typename T, int pack_size>
struct Packer {
  using type = T;
  static constexpr int vec_size = 1;

  __device__ static void load(const T* ptr, T& val) { val = *ptr; }
  __device__ static void store(T* ptr, const T& val) { *ptr = val; }

  __device__ static T get_element(const T& v, int idx) { return v; }
  __device__ static void set_element(T& v, int idx, T val) { v = val; }
};

#define PACKER_TEMPLATE(C_TYPE, CUDA_VEC_TYPE, PACK_SIZE)                   \
  template <>                                                               \
  struct Packer<C_TYPE, PACK_SIZE> {                                        \
    using type = CUDA_VEC_TYPE;                                             \
    static constexpr int vec_size = PACK_SIZE;                              \
                                                                            \
    __device__ static void load(const C_TYPE* ptr, CUDA_VEC_TYPE& v) {      \
      v = *(const CUDA_VEC_TYPE*)ptr;                                       \
    }                                                                       \
                                                                            \
    __device__ static void store(C_TYPE* ptr, const CUDA_VEC_TYPE& v) {     \
      *(CUDA_VEC_TYPE*)ptr = v;                                             \
    }                                                                       \
                                                                            \
    __device__ static C_TYPE get_element(const CUDA_VEC_TYPE& v, int idx) { \
      return (&v.x)[idx];                                                   \
    }                                                                       \
                                                                            \
    __device__ static void set_element(CUDA_VEC_TYPE& v, int idx,           \
                                       C_TYPE val) {                        \
      (&v.x)[idx] = val;                                                    \
    }                                                                       \
  };

PACKER_TEMPLATE(float, float4, 4)
PACKER_TEMPLATE(float, float2, 2)
PACKER_TEMPLATE(int, int2, 2)
PACKER_TEMPLATE(int, int4, 4)
PACKER_TEMPLATE(int64_t, longlong2, 2)
#undef PACKER_TEMPLATE

}  // namespace cuda
}  // namespace recis
