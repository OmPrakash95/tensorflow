#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/lstm_ops.h"

#include <memory>
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"


namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
class LSTMCellBlockOp : public OpKernel {
 public:
  explicit LSTMCellBlockOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* states_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("states_prev", &states_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    const int64 batch_size = x_tensor->dim_size(0);
    const int64 input_size = x_tensor->dim_size(1);

    // Sanity checks for our input shapes.
    CHECK_EQ(states_prev_tensor->dim_size(0), batch_size);
    CHECK_EQ(states_prev_tensor->dim_size(1), cell_size_ * 7);

    CHECK_EQ(w_tensor->dim_size(0), input_size + cell_size_);
    CHECK_EQ(w_tensor->dim_size(1), cell_size_ * 4);

    CHECK_EQ(b_tensor->dim_size(0), cell_size_ * 4);

    Tensor* h_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("h",
          TensorShape({batch_size, cell_size_}), &h_tensor));

    // Allocate our output matrices.
    Tensor* states_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("states",
        TensorShape({batch_size, cell_size_ * 7}), &states_tensor));

    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, input_size + cell_size_}), &xh_tensor));

    functor::LSTMCellBlockFprop<Device>()(
        ctx->eigen_device<Device>(), batch_size, input_size, cell_size_,
        forget_bias_, x_tensor->matrix<float>(), xh_tensor.matrix<float>(),
        states_prev_tensor->matrix<float>(),
        w_tensor->matrix<float>(), b_tensor->vec<float>(),
        h_tensor->matrix<float>(), states_tensor->matrix<float>());
  }

 private:
  int64 cell_size_;
  float forget_bias_;
};

REGISTER_KERNEL_BUILDER(Name("LSTMCellBlock")    \
                            .Device(DEVICE_CPU),
                        LSTMCellBlockOp<CPUDevice>);

#if GOOGLE_CUDA
namespace functor {
  template <>
  void LSTMCellBlockFprop<GPUDevice>::operator()(
      const GPUDevice& d, const int batch_size, const int input_size,
      const int cell_size, const float forget_bias,
      typename TTypes<float>::ConstMatrix x,
      typename TTypes<float>::Matrix xh,
      typename TTypes<float>::ConstMatrix states_prev,
      typename TTypes<float>::ConstMatrix w,
      typename TTypes<float>::ConstVec b,
      typename TTypes<float>::Matrix h,
      typename TTypes<float>::Matrix states);
  extern template struct LSTMCellBlockFprop<GPUDevice>;
}  // namespace functor

REGISTER_KERNEL_BUILDER(Name("LSTMCellBlock")    \
                            .Device(DEVICE_GPU),
                        LSTMCellBlockOp<GPUDevice>);
#endif  // GOOGLE_CUDA

template <typename Device>
class LSTMCellBlockGradOp : public OpKernel {
 public:
  explicit LSTMCellBlockGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* states_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("states_prev", &states_prev_tensor));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    const Tensor* h_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h", &h_tensor));

    const Tensor* states_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("states", &states_tensor));

    const Tensor* h_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("h_grad", &h_grad_tensor));

    const Tensor* states_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("states_grad", &states_grad_tensor));

    const int64 batch_size = x_tensor->dim_size(0);
    const int64 input_size = x_tensor->dim_size(1);

    // Sanity checks for our input shapes.
    CHECK_EQ(states_prev_tensor->dim_size(0), batch_size);
    CHECK_EQ(states_prev_tensor->dim_size(1), cell_size_ * 7);

    CHECK_EQ(w_tensor->dim_size(0), input_size + cell_size_);
    CHECK_EQ(w_tensor->dim_size(1), cell_size_ * 4);

    CHECK_EQ(b_tensor->dim_size(0), cell_size_ * 4);

    CHECK_EQ(h_tensor->dim_size(0), batch_size);
    CHECK_EQ(h_tensor->dim_size(1), cell_size_);

    CHECK_EQ(states_tensor->dim_size(0), batch_size);
    CHECK_EQ(states_tensor->dim_size(1), cell_size_ * 7);

    CHECK_EQ(h_grad_tensor->dim_size(0), batch_size);
    CHECK_EQ(h_grad_tensor->dim_size(1), cell_size_);

    CHECK_EQ(states_grad_tensor->dim_size(0), batch_size);
    CHECK_EQ(states_grad_tensor->dim_size(1), cell_size_ * 7);

    Tensor* x_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("x_grad",
          TensorShape({batch_size, input_size}), &x_grad_tensor));

    Tensor* states_prev_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("states_prev_grad",
          TensorShape({batch_size, cell_size_ * 7}), &states_prev_grad_tensor));

    Tensor* w_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("w_grad",
          TensorShape({input_size + cell_size_, cell_size_ * 4}),
          &w_grad_tensor));

    Tensor* b_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("b_grad",
          TensorShape({cell_size_ * 4}), &b_grad_tensor));

    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, input_size + cell_size_}), &xh_tensor));

    Tensor xh_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, input_size + cell_size_}), &xh_grad_tensor));

    functor::LSTMCellBlockBprop<Device>()(
        ctx->eigen_device<Device>(), batch_size, input_size, cell_size_,
        x_tensor->matrix<float>(), xh_tensor.matrix<float>(),
        states_prev_tensor->matrix<float>(), w_tensor->matrix<float>(),
        b_tensor->vec<float>(), h_tensor->matrix<float>(),
        states_tensor->matrix<float>(), h_grad_tensor->matrix<float>(),
        states_grad_tensor->matrix<float>(), xh_grad_tensor.matrix<float>(),
        x_grad_tensor->matrix<float>(), states_prev_grad_tensor->matrix<float>(),
        w_grad_tensor->matrix<float>(), b_grad_tensor->vec<float>());
  }

 protected:
  int64 cell_size_;
};

REGISTER_KERNEL_BUILDER(Name("LSTMCellBlockGrad")    \
                            .Device(DEVICE_CPU),
                        LSTMCellBlockGradOp<CPUDevice>);

#if GOOGLE_CUDA
namespace functor {
  template <>
  void LSTMCellBlockBprop<GPUDevice>::operator()(
      const GPUDevice& d, const int batch_size, const int input_size,
      const int cell_size,
      typename TTypes<float>::ConstMatrix x,
      typename TTypes<float>::Matrix xh,
      typename TTypes<float>::ConstMatrix states_prev,
      typename TTypes<float>::ConstMatrix w,
      typename TTypes<float>::ConstVec b,
      typename TTypes<float>::ConstMatrix h,
      typename TTypes<float>::ConstMatrix states,
      typename TTypes<float>::ConstMatrix h_grad,
      typename TTypes<float>::ConstMatrix states_grad,
      typename TTypes<float>::Matrix xh_grad,
      typename TTypes<float>::Matrix x_grad,
      typename TTypes<float>::Matrix states_prev_grad,
      typename TTypes<float>::Matrix w_grad,
      typename TTypes<float>::Vec b_grad);
  extern template struct LSTMCellBlockBprop<GPUDevice>;
}  // namespace functor

REGISTER_KERNEL_BUILDER(Name("LSTMCellBlockGrad")    \
                            .Device(DEVICE_GPU),
                        LSTMCellBlockGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
