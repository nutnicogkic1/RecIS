#pragma once
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>

#include <iostream>
namespace recis {
namespace cuda {

__inline__ void checkCudaError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << file
              << ":" << line << std::endl;
    exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA_ERROR(err) checkCudaError(err, __FILE__, __LINE__)

__inline__ int get_sm_count() {
  int device;
  cudaGetDevice(&device);
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  return sm_count;
}

template <typename T>
__inline__ T* cuda_malloc(size_t bytes, cudaStream_t stream = 0) {
  if (bytes == 0) {
    return nullptr;
  }
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  T* dst = reinterpret_cast<T*>(allocator->raw_allocate(bytes));
  return dst;
}

template <typename T>
T* cuda_malloc_and_copy(T* src, int size, cudaStream_t stream = 0,
                        bool async = true) {
  size_t total_bytes = size * sizeof(T);
  T* dst = cuda_malloc<T>(total_bytes, stream);
  cudaMemcpyAsync(dst, src, total_bytes, cudaMemcpyHostToDevice, stream);
  if (!async) {
    cudaStreamSynchronize(stream);
  }
  return dst;
}

template <typename T>
T* cuda_malloc_and_memset(unsigned char byte, size_t size,
                          cudaStream_t stream = 0, bool async = true) {
  size_t total_bytes = size * sizeof(T);
  T* dst = cuda_malloc<T>(total_bytes, stream);
  cudaMemsetAsync(dst, byte, total_bytes, stream);
  if (!async) {
    cudaStreamSynchronize(stream);
  }
  return dst;
}

__inline__ void delete_cuda_ptr(void* ptr) {
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  allocator->raw_delete(ptr);
}

template <typename scalar_t>
void cal_pack_sizes(int64_t num_ids, int64_t embedding_dim,
                    int64_t& emb_tile_size, int64_t& emb_thread_size,
                    int64_t& id_tile_size, int64_t& id_blocks,
                    int64_t& real_pack_size) {
  emb_tile_size = 16 / sizeof(scalar_t);
  emb_tile_size = std::gcd(emb_tile_size, embedding_dim);
  emb_thread_size = std::log2((embedding_dim - 1) / emb_tile_size);
  emb_thread_size = std::pow(2, std::max((emb_thread_size + 1), 0l));
  id_tile_size =
      std::max(emb_thread_size, 128l) / emb_thread_size;  // BLOCK_SIZE=128
  id_blocks = (num_ids + id_tile_size - 1) / id_tile_size;
  real_pack_size = emb_tile_size * sizeof(scalar_t);
}

}  // namespace cuda
}  // namespace recis
