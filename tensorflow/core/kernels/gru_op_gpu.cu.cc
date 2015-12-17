#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/gru_op.h"

namespace tensorflow {
// typedef Eigen::GpuDevice GPUDevice;
// 
// template <>
// struct GruMatMulFunctor<GPUDevice> {
//   void operator()(
//       const GPUDevice& d, typename GruMatMulTypes<float>::out_type out,
//       typename GruMatMulTypes<float>::in_type in0,
//       typename GruMatMulTypes<float>::in_type in1,
//       const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
//       float beta) {
//     GruMatMul<GPUDevice>(d, To32Bit(out), To32Bit(in0), To32Bit(in1), dim_pair, beta);
//   }
// };
// 
// template struct GruMatMulFunctor<GPUDevice>;

template <>
struct GruActivationSigmoid<GPUDevice> {
  void operator()(const GPUDevice& d, Tensor* x) {
    x->vec<float>().device(d) = x->vec<float>().sigmoid();
  }
};

template <>
struct GruActivationTanh<GPUDevice> {
  void operator()(const GPUDevice& d, Tensor* x) {
    x->vec<float>().device(d) = x->vec<float>().tanh();
  }
};

template <>
struct GruCWiseMult<GPUDevice> {
  void operator()(const GPUDevice& d, const Tensor& a, const Tensor& b, Tensor* c) {
    c->vec<float>().device(d) = a.vec<float>() * b.vec<float>();
  }
};

template <>
struct GruH<GPUDevice> {
  void operator()(
      const GPUDevice& d, const Tensor& z, const Tensor& h_prev, const Tensor& g, Tensor* h) {
    h->vec<float>().device(d) =
        z.vec<float>() * h_prev.vec<float>() +
        (z.vec<float>().constant(1.0f) - z.vec<float>()) * g.vec<float>();
  }
};

template struct GruActivationSigmoid<GPUDevice>;
template struct GruActivationTanh<GPUDevice>;
template struct GruCWiseMult<GPUDevice>;
template struct GruH<GPUDevice>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
