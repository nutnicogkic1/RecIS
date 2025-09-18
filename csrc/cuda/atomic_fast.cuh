#ifndef RECIS_FUNCTIONAL_ATOMIC_FAST_CUH_
#define RECIS_FUNCTIONAL_ATOMIC_FAST_CUH_
#include <assert.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
namespace recis {
namespace functional {
// all atomic_add no return
// atomic_add_custom
template <typename T>
__device__ __forceinline__ void atomic_add_custom(T* address, const T val) {
  atomicAdd(address, val);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
template <>
__device__ __forceinline__ void atomic_add_custom<c10::Half>(
    c10::Half* address, const c10::Half val) {
  atomicAdd(reinterpret_cast<__half*>(address), __half(val));
}
#else
template <>
__device__ __forceinline__ void atomic_add_custom<c10::Half>(
    c10::Half* address, const c10::Half val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  at::Half hsum;
  do {
    assumed = old;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    hsum = hsum + val;
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <>
__device__ __forceinline__ void atomic_add_custom<c10::BFloat16>(
    c10::BFloat16* address, const c10::BFloat16 val) {
  atomicAdd(reinterpret_cast<__nv_bfloat16*>(address), __nv_bfloat16(val));
}
#else
template <>
__device__ __forceinline__ void atomic_add_custom<c10::BFloat16>(
    c10::BFloat16* address, const c10::BFloat16 val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  at::BFloat16 bsum;
  do {
    assumed = old;
    bsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    bsum = bsum + val;
    old = (size_t)address & 2 ? (old & 0xffff) | (bsum.x << 16)
                              : (old & 0xffff0000) | bsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}
#endif

// atomic_add_vec
template <typename T>
__device__ inline void atomic_add_vec(T* dst, T value) {
  atomic_add_custom<T>(dst, value);
};

template <>
__device__ inline void atomic_add_vec<float4>(float4* dst, float4 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  atomicAdd(dst, value);
#else
  float* dst_f = reinterpret_cast<float*>(dst);
  atomicAdd(dst_f, value.x);
  atomicAdd(dst_f + 1, value.y);
  atomicAdd(dst_f + 2, value.z);
  atomicAdd(dst_f + 3, value.w);
#endif
}  // atomic_add_vec

template <>
__device__ inline void atomic_add_vec<float2>(float2* dst, float2 value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  atomicAdd(dst, value);
#else
  float* dst_f = reinterpret_cast<float*>(dst);
  atomicAdd(dst_f, value.x);
  atomicAdd(dst_f + 1, value.y);
#endif
}  // atomic_add_vec

// atomic_add_fast
template <typename T>
__device__ inline void atomic_add_fast(T* dst, T value) {
  atomic_add_vec<T>(dst, value);
};

template <>
__device__ inline void atomic_add_fast<float>(float* dst, float value) {
//__match_any_sync requires SM70+; shuffle requires SM50+
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  uint lane = threadIdx.x & 31;
  auto mask_act = __activemask();
  auto mask_sm = __match_any_sync(mask_act, (int64_t)dst);
  int leader = __ffsll(mask_sm) - 1;
  float cur_val = (float)0;
  auto mask_min = mask_sm >> (leader);
  int cur_idx = leader;
  union punner {
    int l;
    float s;
  };
  punner pnr = {};
  pnr.s = value;
  while (mask_min != 0) {
    if (mask_min & 1) {
      punner add_val = {};
      add_val.l = __shfl_sync(mask_act, pnr.l, cur_idx);  // 0xFFFFFFFF
      cur_val += add_val.s;
    };
    cur_idx++;
    mask_min = mask_min >> 1;
  };
  if (lane == leader) {
    atomicAdd(dst, cur_val);
  }
#else
  atomicAdd(dst, value);
#endif
}

template <>
__device__ inline void atomic_add_fast<float2>(float2* dst, float2 value) {
//__match_any_sync requires SM70+; shuffle requires SM50+
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  uint lane = threadIdx.x & 31;
  auto mask_act = __activemask();
  auto mask_sm = __match_any_sync(mask_act, (int64_t)dst);
  int leader = __ffsll(mask_sm) - 1;
  float2 cur_val = make_float2(0.0f, 0.0f);
  auto mask_min = mask_sm >> (leader);
  int cur_idx = leader;
  union punner {
    long long l;
    float2 s;
  };
  punner pnr = {};
  pnr.s = value;
  while (mask_min != 0) {
    if (mask_min & 1) {
      punner add_val = {};
      add_val.l = __shfl_sync(mask_act, pnr.l, cur_idx);  // 0xFFFFFFFF
      cur_val.x += add_val.s.x;
      cur_val.y += add_val.s.y;
    };
    cur_idx++;
    mask_min = mask_min >> 1;
  };
  if (lane == leader) {
    atomic_add_vec<float2>(dst, cur_val);
  }
#else
  atomic_add_vec<float2>(dst, value);
#endif
}  // atomic_add_fast

}  // namespace functional
}  // namespace recis
#endif  // RECIS_FUNCTIONAL_ATOMIC_FAST_CUH_
