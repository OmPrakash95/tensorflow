// Copyright 2015 William Chan <williamchan@cmu.edu>.

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("NBestListDecoding")
    .Attr("state_len: int")
    .Input("state: state_len * float")
    .Input("prefix_logprob: float")
    .Input("token_logprob: float")
    .Output("path_state: state_len * float")
    .Output("path: int32")
    .Output("token: int32")
    .Output("logprob: float")
    .Doc(R"doc(
NBestListDecoding TensorOp.
)doc");

}  // namespace tensorflow
