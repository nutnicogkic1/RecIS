#include <torch/extension.h>

#include <vector>
namespace recis {
namespace functional {
#define _MAX_BIT_SIZE 64
#define _MAX_ENCODE_SIZE 12
#define _MASK (1LL << (_MAX_BIT_SIZE - _MAX_ENCODE_SIZE)) - 1
torch::Tensor ids_encode_cpu(std::vector<torch::Tensor> inputs,
                             torch::Tensor table_ids);

torch::Tensor ids_encode_cuda(std::vector<torch::Tensor> inputs,
                              torch::Tensor table_ids);

torch::Tensor ids_encode(std::vector<torch::Tensor> inputs,
                         torch::Tensor table_ids);

}  // namespace functional
}  // namespace recis
