#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/gru_op.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Partial specialization GruMatMulTensorFunctor<Device=GPUDevice, T>
template <>
struct GruMatMulFunctor<GPUDevice> {
  void operator()(
      const GPUDevice& d, typename GruMatMulTypes<float>::out_type out,
      typename GruMatMulTypes<float>::in_type in0,
      typename GruMatMulTypes<float>::in_type in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      float beta) {
    GruMatMul<GPUDevice>(d, To32Bit(out), To32Bit(in0), To32Bit(in1), dim_pair, beta);
  }
};

template struct GruMatMulFunctor<GPUDevice>;

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
