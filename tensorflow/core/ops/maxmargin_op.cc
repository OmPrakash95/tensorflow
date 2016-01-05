// Copyright 2015 William Chan <williamchan@cmu.edu>.

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/util/padding.h"
namespace tensorflow {
REGISTER_OP("SoftmaxMaxMarginWithLogits")
    .Input("features: float")
    .Input("labels: float")
    .Output("loss: float")
    .Output("backprop: float")
    .Doc(R"doc(
HUNGRY HUNGRY HIPPO!
)doc");
}  // namespace tensorflow
