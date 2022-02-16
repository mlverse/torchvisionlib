#include <Rcpp.h>
#include "torchvisionlib_types.h"
#include "exports.h"

namespace torchvisionlib {

void* tensor_pair::get() {
  return ptr.get();
}

tensor_pair::operator SEXP () const {
  Rcpp::List out;
  out.push_back(rcpp_tensor_pair_get_first(*this));
  out.push_back(rcpp_tensor_pair_get_second(*this));
  return out;
}

tensor_pair::tensor_pair (void* x) : ptr(x, rcpp_delete_tensor_pair) {};

}
