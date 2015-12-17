// Copyright 2015 William Chan <williamchan@cmu.edu>.

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("Gru")
    .Attr("cell_size: int")
    .Attr("sequence_len_max: int")
    .Input("sequence_len: int64")
    .Input("wxr: float")
    .Input("whr: float")
    .Input("wxz: float")
    .Input("whz: float")
    .Input("wxh: float")
    .Input("whh: float")
    .Input("xs: sequence_len_max * float")
    .Output("rs: sequence_len_max * float")
    .Output("zs: sequence_len_max * float")
    .Output("rhs: sequence_len_max * float")
    .Output("gs: sequence_len_max * float")
    .Output("hs: sequence_len_max * float")
    .Doc(R"doc(
GRU
)doc");

REGISTER_OP("GruGrad")
    .Attr("cell_size: int")
    .Attr("sequence_len_max: int")
    .Input("sequence_len: int64")
    .Input("wxr: float")
    .Input("whr: float")
    .Input("wxz: float")
    .Input("whz: float")
    .Input("wxh: float")
    .Input("whh: float")
    .Input("xs: sequence_len_max * float")
    .Input("rs: sequence_len_max * float")
    .Input("zs: sequence_len_max * float")
    .Input("rhs: sequence_len_max * float")
    .Input("gs: sequence_len_max * float")
    .Input("hs: sequence_len_max * float")
    .Input("drs: sequence_len_max * float")
    .Input("dzs: sequence_len_max * float")
    .Input("drhs: sequence_len_max * float")
    .Input("dgs: sequence_len_max * float")
    .Input("dhs: sequence_len_max * float")
    .Output("dwxr: float")
    .Output("dwhr: float")
    .Output("dwxz: float")
    .Output("dwhz: float")
    .Output("dwxh: float")
    .Output("dwhh: float")
    .Output("dxs: sequence_len_max * float")
    .Doc(R"doc(
GRU
)doc");

}  // end namespace tensorflow
