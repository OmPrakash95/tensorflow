// Copyright 2015 William Chan <williamchan@cmu.edu>.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/maxmargin_op.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
class SoftmaxMaxMarginWithLogitsOp : public OpKernel {
 public:
  explicit SoftmaxMaxMarginWithLogitsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    const Tensor& labels_in = context->input(1);

    Tensor* loss_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({logits_in.dim_size(0)}), &loss_out));
    Tensor* back_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, logits_in.shape(), &back_out));
  }
};

namespace functor {
template <typename T>
struct SoftmaxMaxMarginFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::ConstMatrix labels,
                  typename TTypes<T>::Matrix scratch,
                  typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop) {
    SoftmaxMaxMarginEigenImpl<CPUDevice, T>::Compute(d, logits, labels, scratch, loss,
                                         backprop);
  }
};
}  // namespace functor

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("SoftmaxMaxMarginWithLogits")
                            .Device(DEVICE_GPU),
                        SoftmaxMaxMarginWithLogitsOp<GPUDevice>);
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
