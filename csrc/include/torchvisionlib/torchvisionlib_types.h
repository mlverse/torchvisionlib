#include <torch/torch.h>

using tensor_pair = std::tuple<torch::Tensor,torch::Tensor>;

namespace make_raw {
void* TensorPair (const tensor_pair& x);
}

namespace from_raw {
tensor_pair& TensorPair (void* x);
}
