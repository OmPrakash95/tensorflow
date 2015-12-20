// Copyright 2015 William Chan <williamchan@cmu.edu>.

#ifndef TENSORFLOW_KERNELS_GRU_OP_H_
#define TENSORFLOW_KERNELS_GRU_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// Helpers to define tensor<T> needed by GruMatMul op.
template <typename T>
struct GruMatMulTypes {
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>, Eigen::Aligned>
      out_type;
  typedef Eigen::TensorMap<Eigen::Tensor<const T, 2, Eigen::RowMajor>,
                           Eigen::Aligned> in_type;
};

template <typename Device, typename In0, typename In1, typename Out,
          typename DimPair>
void GruMatMul(const Device& d, Out out, In0 in0, In1 in1,
            const DimPair& dim_pair, float beta) {
  if (beta == 0.0f) {
    out.device(d) = in0.contract(in1, dim_pair);
  } else {
    out.device(d) = out.constant(beta) * out + in0.contract(in1, dim_pair);
  }
}

template <typename Device>
struct GruMatMulFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, typename GruMatMulTypes<float>::out_type out,
      typename GruMatMulTypes<float>::in_type in0,
      typename GruMatMulTypes<float>::in_type in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      float beta);
};

}  // end namespace functor

namespace generator {
class AttentionMaskGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  AttentionMaskGenerator(
      float fill_value, TTypes<int64>::ConstVec sequence_len,
      TTypes<float, 2>::ConstTensor input)
    : fill_value_(fill_value), sequence_len_(sequence_len), input_(input) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  float operator()(const Eigen::array<Eigen::DenseIndex, 2>& coords) const {
    if (coords[1] < sequence_len_(coords[0])) {
      return input_(coords);
    } else {
      return fill_value_;
    }
  }

 private:
  float fill_value_;
  TTypes<int64>::ConstVec sequence_len_;
  TTypes<float, 2>::ConstTensor input_;
};
}  // end namespace generator

template <typename Device>
struct GruDeviceSynchronize {
  void operator()(const Device& d);
};

template <typename Device>
struct GruSetZero {
  void operator()(const Device& d, Tensor* x);
};

template <typename Device>
struct AttentionMask {
  void operator()(
      const Device& d, float fill_value, const Tensor& sequence_len,
      const Tensor& input, Tensor* output);
};

template <typename Device>
struct GruPadTime {
  void operator()(
      const Device& d, const Tensor& sequence_len, const int64 sequence_idx,
      float value, Tensor* x);
};

template <typename Device>
struct GruActivationSigmoid {
  void operator()(const Device& d, Tensor* x);
};

template <typename Device>
struct GruActivationTanh {
  void operator()(const Device& d, Tensor* x);
};

template <typename Device>
struct GruActivationSigmoidGradient {
  void operator()(const Device& d, const Tensor& x, Tensor* dx);
};

template <typename Device>
struct GruActivationTanhGradient {
  void operator()(const Device& d, const Tensor& x, Tensor* dx);
};

template <typename Device>
struct GruCWiseMult {
  void operator()(const GPUDevice& d, const Tensor& a, const Tensor& b,
      float beta, Tensor* c);
};

template <typename Device>
struct GruH {
  void operator()(
      const Device& d, const Tensor& z, const Tensor* h_prev, const Tensor& g,
      Tensor* h);
};

template <typename Device>
struct GruDz {
  void operator()(
      const Device& d, const Tensor& dh, const Tensor* h_prev, const Tensor& g,
      Tensor* dz);
};

template <typename Device>
struct GruDg {
  void operator()(
      const Device& d, const Tensor& dh, const Tensor& z, Tensor* dg);
};

void GruSetZeroGPU(const GPUDevice& d, Tensor* x);
void AttentionMaskGPU(
    const GPUDevice& d, float fill_value, const Tensor& sequence_len,
    const Tensor& input, Tensor* output);
void GruPadTimeGPU(
    const GPUDevice& d, const Tensor& sequence_len, const int64 sequence_idx,
    float value, Tensor* x);
void GruActivationSigmoidGPU(const GPUDevice& d, Tensor* x);
void GruActivationTanhGPU(const GPUDevice& d, Tensor* x);
void GruActivationSigmoidGradientGPU(
    const GPUDevice& d, const Tensor& x, Tensor* dx);
void GruActivationTanhGradientGPU(
    const GPUDevice& d, const Tensor& x, Tensor* dx);
void GruCWiseMultGPU(
    const GPUDevice& d, const Tensor& a, const Tensor& b, float beta, Tensor* c);

void GruHGPU(
    const GPUDevice& d, const Tensor& z, const Tensor* h_prev, const Tensor& g,
    Tensor* h);
void GruDzGPU(
    const GPUDevice& d, const Tensor& dh, const Tensor* h_prev, const Tensor& g,
    Tensor* dz);
void GruDgGPU(
    const GPUDevice& d, const Tensor& dh, const Tensor& z, Tensor* dg);

}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_GRU_OP_H_
