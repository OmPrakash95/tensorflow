#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/lstm_ops.h"

namespace tensorflow {

namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template struct LSTMCellBlockFprop<GPUDevice, true>;
template struct LSTMCellBlockBprop<GPUDevice, true>;

}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA
