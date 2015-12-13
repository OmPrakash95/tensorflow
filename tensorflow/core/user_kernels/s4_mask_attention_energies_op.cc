// Copyright 2015 William Chan <williamchan@cmu.edu>.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/user_kernels/s4_mask_attention_energies_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
class S4MaskAttentionEnergiesOp : public OpKernel {
 public:
  explicit S4MaskAttentionEnergiesOp(OpKernelConstruction* ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("median_window_left", &median_window_left_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("median_window_right", &median_window_right_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("mode", &mode_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* energies = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("energies", &energies));

    const Tensor* energies_len = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("energies_len", &energies));

    auto seq_lens_t = energies_len->vec<int64>();

    Tensor* masked_energies = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, energies->shape(), &masked_energies));

    functor::S4MaskAttentionEnergies<Device>::ComputeDefault(
        context->eigen_device<Device>(), energies->tensor<float, 2>(), seq_lens_t,
        masked_energies->tensor<float, 2>());
  }

 protected:
  int64 median_window_left_;
  int64 median_window_right_;
  std::string mode_;
};

REGISTER_KERNEL_BUILDER(Name("S4MaskAttentionEnergies").Device(DEVICE_CPU), S4MaskAttentionEnergiesOp<CPUDevice>);

#if GOOGLE_CUDA

namespace functor {
  template <>
  void S4MaskAttentionEnergies<GPUDevice>::ComputeDefault(
      const GPUDevice& d, typename TTypes<float, 2>::ConstTensor energies,
      TTypes<int64>::ConstVec energies_len,
      TTypes<float, 2>::Tensor masked_energies);

  template <>
  void S4MaskAttentionEnergies<GPUDevice>::ComputeMedian(
      const GPUDevice& d, typename TTypes<float, 2>::ConstTensor energies,
      const GPUDevice& d, typename TTypes<float, 2>::ConstTensor energies_prev,
      int64 median_window_left, int64 median_window_right,
      TTypes<int64>::ConstVec energies_len,
      TTypes<float, 2>::Tensor masked_energies);

  extern template struct S4MaskAttentionEnergies<GPUDevice>;
}  // namespace functor

REGISTER_KERNEL_BUILDER(Name("S4MaskAttentionEnergies").Device(DEVICE_GPU), S4MaskAttentionEnergiesOp<GPUDevice>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
