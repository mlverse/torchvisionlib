#include <torch/torch.h>
#include "torchvisionlib/torchvisionlib_types.h"
#include <torchvisionlib/torchvisionlib.h>
#include <lantern/types.h>

namespace make_raw {

void* TensorPair (const tensor_pair& x) {
  return make_ptr<tensor_pair>(x);
}
}

namespace from_raw {
tensor_pair& TensorPair (void* x) {
  return *reinterpret_cast<tensor_pair*>(x);
}
}

// [[torch::export]]
void delete_tensor_pair(void* x) {
  delete reinterpret_cast<tensor_pair*>(x);
}

// [[torch::export]]
torch::Tensor tensor_pair_get_first(tensor_pair x) {
  return std::get<0>(x);
}

// [[torch::export]]
torch::Tensor tensor_pair_get_second(tensor_pair x) {
  return std::get<1>(x);
}
