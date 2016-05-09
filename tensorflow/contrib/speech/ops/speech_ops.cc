#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("ParseUtterance")
    .Attr("feats_dim: int")
    .Attr("feats_len_max: int")
    .Attr("tokens_len_max: int")
    .Attr("token_model: string")
    .Input("serialized: string")
    .Output("feats: feats_len_max * float")
    .Output("feats_len: int64")
    .Output("tokens: tokens_len_max * int64")
    .Output("tokens_len: int64")
    .Output("tokens_weight: tokens_len_max * float")
    .Output("text: string")
    .Output("uttid: string")
    .Doc(R"doc(
)doc");

}  // end namespace tensorflow
