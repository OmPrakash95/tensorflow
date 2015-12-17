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

}  // namespace functor

template <typename Device>
struct GruActivationSigmoid {
  void operator()(const Device& d, Tensor* x);
};

template <typename Device>
struct GruActivationTanh {
  void operator()(const Device& d, Tensor* x);
};

template <typename Device>
struct GruCWiseMult {
  void operator()(const GPUDevice& d, const Tensor& a, const Tensor& b, Tensor* c);
};

template <typename Device>
struct GruH {
  void operator()(
      const Device& d, const Tensor& z, const Tensor& h_prev, const Tensor& g, Tensor* h);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_KERNELS_GRU_OP_H_
