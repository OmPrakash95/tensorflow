#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/user_kernels/gru_op.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Partial specialization MatMulTensorFunctor<Device=GPUDevice, T>
template <>
struct MatMulFunctor<GPUDevice> {
  void operator()(
      const GPUDevice& d, typename MatMulTypes<float>::out_type out,
      typename MatMulTypes<float>::in_type in0,
      typename MatMulTypes<float>::in_type in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      float beta) {
    MatMul<GPUDevice>(d, To32Bit(out), To32Bit(in0), To32Bit(in1), dim_pair, beta);
  }
};

template struct MatMulFunctor<GPUDevice>;

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
