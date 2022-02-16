#pragma once
#include <torch.h>

namespace torchvisionlib {

class tensor_pair {
public:
  // this is the slot to hold the void*
  std::shared_ptr<void> ptr;
  // the constructor from a void*
  tensor_pair (void* x);
  // casting operator Rcpp->SEXP
  operator SEXP () const;
  // returns the void* from the type.
  void* get ();
};

}
