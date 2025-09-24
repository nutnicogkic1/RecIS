#pragma once
#include <cuda_runtime.h>

#include <functional>

#include "cuda/packer.cuh"
#include "cuda/utils.cuh"

namespace recis {
namespace cuda {

#define KWRAP_SIZE 32
#define KBLOCK_SIZE 256

template <typename A, typename B, typename C, typename Factory>
__global__ void fused_element_wise_kernel_packed(const A** a, const B* b, C** c,
                                                 int64_t N, int64_t* sizes,
                                                 Factory factory) {
  int64_t vec_id = blockIdx.y;
  int64_t size_local = sizes[vec_id];
  int64_t threads_num = blockDim.x * gridDim.x;
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  static constexpr int pack_size = select_pack_size<A, C>::value;
  using AP = Packer<A, pack_size>;
  using CP = Packer<C, pack_size>;

  for (int64_t index = tid; index * pack_size < size_local;
       index += threads_num) {
    int64_t idx = index * pack_size;

    if (idx + pack_size < size_local) {
      typename AP::type a_vec;
      typename CP::type c_vec;

      AP::load(a[vec_id] + idx, a_vec);

#pragma unroll
      for (int64_t j = 0; j < pack_size; ++j) {
        auto a_val = AP::get_element(a_vec, j);
        auto result = factory(a_val, b[vec_id]);
        CP::set_element(c_vec, j, result);
      }

      CP::store(c[vec_id] + idx, c_vec);
    } else {
      for (int64_t i = idx; i < size_local; i++) {
        c[vec_id][i] = factory(a[vec_id][i], b[vec_id]);
      }
    }
  }
}

template <typename A, typename B, typename C, typename Factory>
__global__ void fused_element_wise_kernel(const A** a, const B* b, C** c,
                                          int64_t N, int64_t* sizes,
                                          Factory factory) {
  int64_t vec_id = blockIdx.y;
  int64_t size_local = sizes[vec_id];
  int64_t threads_num = blockDim.x * gridDim.x;
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int64_t index = tid; index < size_local; index += threads_num) {
    c[vec_id][index] = factory(a[vec_id][index], b[vec_id]);
  }
}

template <typename A, typename B, typename C, typename Factory>
void fused_element_wise_launcher(const A** a, const B* b, C** c, int64_t* sizes,
                                 int64_t N, Factory factor, bool with_pack,
                                 cudaStream_t stream) {
  int64_t sm_count = get_sm_count();
  int64_t max_size = 0;
  std::vector<int64_t> offsets(N + 1, 0);
  for (int64_t i = 0; i < N; ++i) {
    max_size = std::max(max_size, sizes[i]);
  }
  if (max_size == 0) return;
  int64_t block_num =
      min(sm_count * 8, (max_size + KBLOCK_SIZE - 1) / KBLOCK_SIZE);
  dim3 grid(block_num, N);
  dim3 block(KBLOCK_SIZE);
  int64_t* d_sizes = cuda_malloc_and_copy<int64_t>(sizes, N, stream);
  if (with_pack) {
    fused_element_wise_kernel_packed<A, B, C, Factory>
        <<<grid, block, 0, stream>>>(a, b, c, N, d_sizes, factor);
  } else {
    fused_element_wise_kernel<A, B, C, Factory>
        <<<grid, block, 0, stream>>>(a, b, c, N, d_sizes, factor);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  delete_cuda_ptr(d_sizes);
}

}  // namespace cuda
}  // namespace recis
