// Copyright 2015 William Chan <williamchan@cmu.edu>.

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/user_kernels/s4_mask_attention_energies_op.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template struct functor::S4MaskAttentionEnergies<GPUDevice>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
