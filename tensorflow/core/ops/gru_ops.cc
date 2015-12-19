// Copyright 2015 William Chan <williamchan@cmu.edu>.

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("GruCell")
    .Attr("cell_size: int")
    .Input("sequence_len: int64")
    .Input("wxr: float")
    .Input("whr: float")
    .Input("wxz: float")
    .Input("whz: float")
    .Input("wxh: float")
    .Input("whh: float")
    .Input("h_prev: float")
    .Input("x: float")
    .Output("r: float")
    .Output("z: float")
    .Output("rh: float")
    .Output("g: float")
    .Output("h: float")
    .Doc(R"doc(
GRU Cell
)doc");

REGISTER_OP("GruCellGrad")
    .Attr("cell_size: int")
    .Input("sequence_len: int64")
    .Input("wxr: float")
    .Input("whr: float")
    .Input("wxz: float")
    .Input("whz: float")
    .Input("wxh: float")
    .Input("whh: float")
    .Input("h_prev: float")
    .Input("x: float")
    .Input("r: float")
    .Input("z: float")
    .Input("rh: float")
    .Input("hh: float")
    .Input("h: float")
    .Input("dh: float")
    .Output("dwxr: float")
    .Output("dwhr: float")
    .Output("dwxz: float")
    .Output("dwhz: float")
    .Output("dwxh: float")
    .Output("dwhh: float")
    .Output("dh_prev: float")
    .Output("dx: float")
    .Doc(R"doc(
GRU Cell Gradient
)doc");

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
GRU Gradient
)doc");

REGISTER_OP("Sink")
    .Attr("sinks: int")
    .Input("input: sinks * float")
    .Doc(R"doc(
Sink NoOp
)doc");

}  // end namespace tensorflow
