// Copyright 2015 William Chan <williamchan@cmu.edu>.

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("TokenSample")
    .Attr("sample_prob: float")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Input("ground_truth: int32")
    .Input("token_distribution: float")
    .Output("token: int32")
    .Doc(R"doc(
AttentionMask
)doc");

REGISTER_OP("GruCell")
    .Attr("cell_size: int")
    .Attr("time_idx: int = -1")
    .Input("sequence_len: int64")
    .Input("wxr: float")
    .Input("whr: float")
    .Input("wxz: float")
    .Input("whz: float")
    .Input("wxh: float")
    .Input("whh: float")
    .Input("br: float")
    .Input("bz: float")
    .Input("bh: float")
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
    .Attr("time_idx: int = -1")
    .Input("sequence_len: int64")
    .Input("wxr: float")
    .Input("whr: float")
    .Input("wxz: float")
    .Input("whz: float")
    .Input("wxh: float")
    .Input("whh: float")
    .Input("br: float")
    .Input("bz: float")
    .Input("bh: float")
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
    .Output("dbr: float")
    .Output("dbz: float")
    .Output("dbh: float")
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
    .Input("br: float")
    .Input("bz: float")
    .Input("bh: float")
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
    .Input("br: float")
    .Input("bz: float")
    .Input("bh: float")
    .Input("xs: sequence_len_max * float")
    .Input("rs: sequence_len_max * float")
    .Input("zs: sequence_len_max * float")
    .Input("rhs: sequence_len_max * float")
    .Input("gs: sequence_len_max * float")
    .Input("hs: sequence_len_max * float")
    .Input("dhs: sequence_len_max * float")
    .Output("dwxr: float")
    .Output("dwhr: float")
    .Output("dwxz: float")
    .Output("dwhz: float")
    .Output("dwxh: float")
    .Output("dwhh: float")
    .Output("dbr: float")
    .Output("dbz: float")
    .Output("dbh: float")
    .Output("dxs: sequence_len_max * float")
    .Doc(R"doc(
GRU Gradient
)doc");

REGISTER_OP("GruFused")
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
    .Output("rzs: sequence_len_max * float")
    .Output("rhs: sequence_len_max * float")
    .Output("gs: sequence_len_max * float")
    .Output("hs: sequence_len_max * float")
    .Doc(R"doc(
GRU
)doc");

REGISTER_OP("GruFusedGrad")
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
    .Input("rzs: sequence_len_max * float")
    .Input("rhs: sequence_len_max * float")
    .Input("gs: sequence_len_max * float")
    .Input("hs: sequence_len_max * float")
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

REGISTER_OP("CCTCBootstrapAlignment")
    .Attr("blank_token: int = 4")
    .Attr("lpad: int = 10")
    .Attr("rpad: int = 2")
    .Attr("features_len_max: int")
    .Attr("tokens_len_max: int")
    .Input("tokens: tokens_len_max * int32")
    .Input("tokens_len: int64")
    .Input("features_len: int64")
    .Output("tokens_aligned: features_len_max * int32")
    .Output("tokens_aligned_weight: features_len_max * float")
    .Doc(R"doc(
)doc");

REGISTER_OP("CCTCEditDistance")
    .Attr("blank_token: int = 4")
    .Attr("features_len_max: int")
    .Attr("tokens_len_max: int")
    .Input("hyp: features_len_max * int32")
    .Input("hyp_logits: features_len_max * float")
    .Input("hyp_probs: features_len_max * float")
    .Input("hyp_baseline: features_len_max * float")
    .Input("ref: tokens_len_max * int32")
    .Input("ref_len: int64")
    .Output("edit_distance: int64")
    .Doc(R"doc(
)doc");

REGISTER_OP("CCTCEditDistanceReinforceGrad")
    .Attr("blank_token: int = 4")
    .Attr("features_len_max: int")
    .Attr("tokens_len_max: int")
    .Attr("discount_factor: float = 1.0")
    .Input("hyp: features_len_max * int32")
    .Input("hyp_probs: features_len_max * float")
    .Input("hyp_baseline: features_len_max * float")
    .Input("ref: tokens_len_max * int32")
    .Input("ref_len: int64")
    .Output("hyp_logits_backprop: features_len_max * float")
    .Output("hyp_baseline_backprop: features_len_max * float")
    .Doc(R"doc(
)doc");

}  // end namespace tensorflow
