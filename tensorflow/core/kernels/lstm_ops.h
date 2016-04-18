#ifndef TENSORFLOW_KERNELS_LSTM_OPS_H_
#define TENSORFLOW_KERNELS_LSTM_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/stream.h"

namespace tensorflow {

#if GOOGLE_CUDA

namespace {
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace

#endif  // GOOGLE_CUDA

namespace functor {

template <typename Device, bool USE_CUBLAS>
struct LSTMCellBlockFprop {
  void operator()(
      perftools::gputools::Stream* stream, const Device& d,
      const int batch_size, const int input_size, const int cell_size, const float forget_bias,
      typename TTypes<float>::ConstMatrix x,
      typename TTypes<float>::Matrix xh,
      typename TTypes<float>::ConstMatrix states_prev,
      typename TTypes<float>::ConstMatrix w,
      typename TTypes<float>::ConstVec b,
      typename TTypes<float>::Matrix h,
      typename TTypes<float>::Matrix states) {
    // Pointer offsets.
    Eigen::array<int, 2> i_offsets  = {0, cell_size * 0};
    Eigen::array<int, 2> f_offsets  = {0, cell_size * 1};
    Eigen::array<int, 2> o_offsets  = {0, cell_size * 2};
    Eigen::array<int, 2> cs_offsets = {0, cell_size * 3};
    Eigen::array<int, 2> ci_offsets = {0, cell_size * 4};
    Eigen::array<int, 2> co_offsets = {0, cell_size * 5};
    Eigen::array<int, 2> h_offsets  = {0, cell_size * 6};

    Eigen::array<int, 2> cell_extents = {batch_size, cell_size};

    // xh = [x, h_prev].
    Eigen::array<int, 2> xh_x_offsets = {0, 0};
    Eigen::array<int, 2> xh_x_extents = {batch_size, input_size};
    Eigen::array<int, 2> xh_h_offsets = {0, input_size};
    Eigen::array<int, 2> xh_h_extents = {batch_size, cell_size};

    auto h_prev =
        states_prev.slice(h_offsets, cell_extents);

    xh.slice(xh_x_offsets, xh_x_extents).device(d) = x;
    xh.slice(xh_h_offsets, xh_h_extents).device(d) = h_prev;

    Eigen::array<int, 2> states_offsets = {0, 0};
    Eigen::array<int, 2> states_extents = {batch_size, cell_size * 4};

    // states = xh * w + b
    if (USE_CUBLAS) {
      states.slice(states_offsets, states_extents).device(d) =
          b.broadcast(Eigen::array<int, 2>({batch_size, 1}));

      auto kNoTranspose =
          perftools::gputools::blas::Transpose::kNoTranspose;

      const uint64 m = batch_size;
      const uint64 k = input_size + cell_size;
      const uint64 n = cell_size * 4;

      auto a_ptr = AsDeviceMemory(xh.data());
      auto b_ptr = AsDeviceMemory(w.data());
      auto c_ptr = AsDeviceMemory(states.data());

      bool blas_launch_status = stream->ThenBlasGemm(
          kNoTranspose, kNoTranspose, n, m, k, 1.0f, b_ptr, n, a_ptr, k, 1.0f,
          &c_ptr, cell_size * 7).ok();
      CHECK(blas_launch_status);
    } else {
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
      contract_pairs[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);

      states.slice(states_offsets, states_extents).device(d) =
          xh.contract(w, contract_pairs) +
          b.broadcast(Eigen::array<int, 2>({batch_size, 1}));
    }

    // Forget gate bias.
    auto f = states.slice(f_offsets, cell_extents);
    f.device(d) += f.constant(forget_bias);

    // Apply activation functions to input, forget, output and cell input.
    // i = sigmoid(states[0])
    // f = sigmoid(states[1])
    // o = sigmoid(states[2])
    Eigen::array<int, 2> ifo_offsets = {0, 0};
    Eigen::array<int, 2> ifo_extents = {batch_size, cell_size * 3};
    states.slice(ifo_offsets, ifo_extents).device(d) =
        states.slice(ifo_offsets, ifo_extents).sigmoid();

    // ci = tanh(states[3])
    auto ci = states.slice(ci_offsets, cell_extents);
    ci.device(d) =
        states.slice(cs_offsets, cell_extents).tanh();

    // cs = ci .* i + f .* cs_prev
    auto i = states.slice(i_offsets, cell_extents);
    auto cs_prev = states_prev.slice(cs_offsets, cell_extents);
    auto cs = states.slice(cs_offsets, cell_extents);
    cs.device(d) = i * ci + f * cs_prev;
    // states = [i, f, o, cs]

    // co = tanh(cs)
    auto co = states.slice(co_offsets, cell_extents);
    co.device(d) = cs.tanh();

    // h = o .* co
    auto o = states.slice(o_offsets, cell_extents);
    states.slice(h_offsets, cell_extents).device(d) = o * co;

    h.device(d) = states.slice(h_offsets, cell_extents);
  }
};

template <typename Device, bool USE_CUBLAS>
struct LSTMCellBlockBprop {
  void operator()(
      perftools::gputools::Stream* stream, const Device& d,
      const int batch_size, const int input_size, const int cell_size,
      typename TTypes<float>::ConstMatrix x,
      typename TTypes<float>::Matrix xh,
      typename TTypes<float>::ConstMatrix states_prev,
      typename TTypes<float>::ConstMatrix w,
      typename TTypes<float>::ConstVec b,
      typename TTypes<float>::ConstMatrix states,
      typename TTypes<float>::ConstMatrix h_grad,
      typename TTypes<float>::ConstMatrix states_grad,
      typename TTypes<float>::Matrix xh_grad,
      typename TTypes<float>::Matrix x_grad,
      typename TTypes<float>::Matrix states_prev_grad,
      typename TTypes<float>::Matrix w_grad,
      typename TTypes<float>::Vec b_grad) {
    // Pointer offsets.
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
    auto h_prev = states_prev.slice(h_offsets, cell_extents);
    xh.slice(xh_h_offsets, xh_h_extents).device(d) = h_prev;

    // dh.
    auto dh = states_prev_grad.slice(h_offsets, cell_extents);
    dh.device(d) =
        h_grad + states_grad.slice(h_offsets, cell_extents);

    // do[t] = sigm'(o[t]) .* dh[t] .* co[t]
    auto co = states.slice(co_offsets, cell_extents);
    auto o = states.slice(o_offsets, cell_extents);
    states_prev_grad.slice(o_offsets, cell_extents).device(d) =
        o * (o.constant(1.0f) - o) * dh * co;

    // dcs[t] += tanh'(cs[t]) .* dh[t] .* o[t]
    auto dcs = states_prev_grad.slice(co_offsets, cell_extents);
    dcs.device(d) =
        (co.constant(1.0f) - co * co) * dh * o +
        states_grad.slice(ci_offsets, cell_extents);

    // dci[t] = tanh'(ci[t]) dcs[t] i[t]
    auto ci = states.slice(ci_offsets, cell_extents);
    auto i = states.slice(i_offsets, cell_extents);
    states_prev_grad.slice(cs_offsets, cell_extents).device(d) =
        (ci.constant(1.0f) - ci * ci) * dcs * i;

    // df[t] = sigm'(f[t]) dcs[t] cs[t - 1]
    auto f = states.slice(f_offsets, cell_extents);
    auto cs_prev = states_prev.slice(cs_offsets, cell_extents);
    states_prev_grad.slice(f_offsets, cell_extents).device(d) =
        f * (f.constant(1.0f) - f) * dcs * cs_prev;

    // di[t] = sigm'(i[t]) dcs[t] ci[t]
    states_prev_grad.slice(i_offsets, cell_extents).device(d) =
        i * (i.constant(1.0f) - i) * dcs * ci;

    // xh_grad.
    Eigen::array<int, 2> states_offsets = {0, 0};
    Eigen::array<int, 2> states_extents = {batch_size, cell_size * 4};

    auto dstate4 = states_prev_grad.slice(states_offsets, states_extents);

    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> xh_grad_contract_pairs;
    xh_grad_contract_pairs[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 1);

    // xh_grad = dstate4 * w^T
    if (USE_CUBLAS) {
      auto kTranspose =
          perftools::gputools::blas::Transpose::kTranspose;
      auto kNoTranspose =
          perftools::gputools::blas::Transpose::kNoTranspose;

      const uint64 m = batch_size;
      const uint64 k = cell_size * 4;
      const uint64 n = input_size + cell_size;

      auto a_ptr = AsDeviceMemory(states_prev_grad.data());
      auto b_ptr = AsDeviceMemory(w.data());
      auto c_ptr = AsDeviceMemory(xh_grad.data());

      bool blas_launch_status = CHECK_NOTNULL(stream)->ThenBlasGemm(
          kTranspose, kNoTranspose, n, m, k, 1.0f, b_ptr,
          k, a_ptr, cell_size * 7, 0.0f, &c_ptr, n).ok();
      CHECK(blas_launch_status);
    } else {
      xh_grad.device(d) =
          dstate4.contract(w, xh_grad_contract_pairs);
    }

    x_grad.device(d) = xh_grad.slice(xh_x_offsets, xh_x_extents);
    states_prev_grad.slice(h_offsets, cell_extents).device(d) =
        xh_grad.slice(xh_h_offsets, xh_h_extents);

    // dcs.
    states_prev_grad.slice(ci_offsets, cell_extents).device(d) = dcs * f;

    // w_grad, b_grad.
    if (USE_CUBLAS) {
      auto kTranspose =
          perftools::gputools::blas::Transpose::kTranspose;
      auto kNoTranspose =
          perftools::gputools::blas::Transpose::kNoTranspose;

      const uint64 m = input_size + cell_size;
      const uint64 k = batch_size;
      const uint64 n = cell_size * 4;

      auto a_ptr = AsDeviceMemory(xh.data());
      auto b_ptr = AsDeviceMemory(states_prev_grad.data());
      auto c_ptr = AsDeviceMemory(w_grad.data());

      bool blas_launch_status = CHECK_NOTNULL(stream)->ThenBlasGemm(
          kNoTranspose, kTranspose, n, m, k, 1.0f, b_ptr,
          cell_size * 7, a_ptr, m, 1.0f, &c_ptr, n).ok();
      CHECK(blas_launch_status);
    } else {
      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> w_grad_contract_pairs;
      w_grad_contract_pairs[0] = Eigen::IndexPair<Eigen::DenseIndex>(0, 0);

      w_grad.device(d) += xh.contract(dstate4, w_grad_contract_pairs);
    }
    b_grad.device(d) += dstate4.sum(Eigen::array<int, 1>(0));
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_LSTM_OPS_H_
