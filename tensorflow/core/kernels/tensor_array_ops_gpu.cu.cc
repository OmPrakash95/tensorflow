#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/tensor_array_ops.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
template struct functor::TensorArraySubsample<GPUDevice, float>;
template struct functor::TensorArraySubsampleGrad<GPUDevice, float>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
