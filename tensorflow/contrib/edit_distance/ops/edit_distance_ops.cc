#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("EditDistanceList")
    .Attr("eos_token: int")
    .Attr("sequence_len_max: int")
    .Attr("T: {int32,int64}")
    .Input("ref: sequence_len_max * T")
    .Input("hyp: sequence_len_max * T")
    .Output("edit_distance: int64")
    .Output("ref_length: int64")
    .Doc(R"doc(
)doc");

}  // end namespace tensorflow
