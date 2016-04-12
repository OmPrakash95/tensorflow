#ifndef TENSORFLOW_KERNELS_LSTM_OPS_H_
#define TENSORFLOW_KERNELS_LSTM_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"


namespace tensorflow {
namespace functor {

template <typename Device>
struct LSTMCellFprop {
  void operator()(const Device& d, const int batch_size, const int input_size,
                  const int cell_size, const float forget_bias,
                  typename TTypes<float>::ConstMatrix x,
                  typename TTypes<float>::Matrix xh,
                  typename TTypes<float>::ConstMatrix states_prev,
                  typename TTypes<float>::ConstMatrix w,
                  typename TTypes<float>::ConstVec b,
                  typename TTypes<float>::Matrix h,
                  typename TTypes<float>::Matrix states) {
    Eigen::array<int, 2> i_offsets  = {0, cell_size * 0};
    Eigen::array<int, 2> f_offsets  = {0, cell_size * 1};
    Eigen::array<int, 2> o_offsets  = {0, cell_size * 2};
    Eigen::array<int, 2> cs_offsets = {0, cell_size * 3};
    Eigen::array<int, 2> ci_offsets = {0, cell_size * 4};
    Eigen::array<int, 2> co_offsets = {0, cell_size * 5};
    Eigen::array<int, 2> h_offsets  = {0, cell_size * 6};

    Eigen::array<int, 2> cell_extents = {batch_size, cell_size};

    // xh = [x, h_prev]
    Eigen::array<int, 2> xh_x_offsets = {0, 0};
    Eigen::array<int, 2> xh_x_extents = {batch_size, input_size};
    Eigen::array<int, 2> xh_h_offsets = {0, input_size};
    Eigen::array<int, 2> xh_h_extents = {batch_size, cell_size};
    xh.slice(xh_x_offsets, xh_x_extents).device(d) = x;
    xh.slice(xh_h_offsets, xh_h_extents).device(d) =
        states_prev.slice(h_offsets, cell_extents);

    // contract_pairs is GEMM w/o any transpose.
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
    contract_pairs[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);

    Eigen::array<int, 2> states_offsets = {0, 0};
    Eigen::array<int, 2> states_extents = {batch_size, cell_size * 4};

    // states = xh * W + b
    states.slice(states_offsets, states_extents).device(d) =
        xh.contract(w, contract_pairs) +
        b.broadcast(Eigen::array<int, 2>({batch_size, 1}));

    // Forget gate bias.
    states.slice(f_offsets, cell_extents).device(d) +=
        states.slice(f_offsets, cell_extents).constant(forget_bias);

    // Apply activation functions to input, forget, output and cell input.
    // i = sigmoid(states[0])
    // f = sigmoid(states[1])
    // o = sigmoid(states[2])
    Eigen::array<int, 2> ifo_offsets = {0, 0};
    Eigen::array<int, 2> ifo_extents = {batch_size, cell_size * 3};
    states.slice(ifo_offsets, ifo_extents).device(d) =
        states.slice(ifo_offsets, ifo_extents).sigmoid();

    // ci = tanh(states[3])
    states.slice(ci_offsets, cell_extents).device(d) =
        states.slice(cs_offsets, cell_extents).tanh();
    
    // cs = ci .* i + f .* cs_prev
    states.slice(cs_offsets, cell_extents).device(d) =
        states.slice(i_offsets, cell_extents) * states.slice(ci_offsets, cell_extents) +
        states.slice(f_offsets, cell_extents) * states_prev.slice(cs_offsets, cell_extents);
    // states = [i, f, o, cs]

    // co = tanh(cs)
    states.slice(co_offsets, cell_extents).device(d) =
        states.slice(cs_offsets, cell_extents).tanh();

    // h = o .* co
    states.slice(h_offsets, cell_extents).device(d) =
        states.slice(o_offsets, cell_extents) *
        states.slice(co_offsets, cell_extents);

    h.device(d) = states.slice(h_offsets, cell_extents);
  }
};

template <typename Device>
struct LSTMCellBprop {
  void operator()(const Device& d, const int batch_size, const int input_size,
                  const int cell_size,
                  typename TTypes<float>::ConstMatrix x,
                  typename TTypes<float>::Matrix xh,
                  typename TTypes<float>::ConstMatrix states_prev,
                  typename TTypes<float>::ConstMatrix w,
                  typename TTypes<float>::ConstVec b,
                  typename TTypes<float>::Matrix h,
                  typename TTypes<float>::Matrix states) {
}

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_LSTM_OPS_H_
