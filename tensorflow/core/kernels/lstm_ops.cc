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
class LstmCellOp : public OpKernel {
 public:
  explicit LstmCellOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* x_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("x", &x_tensor));

    const Tensor* states_prev_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("stats_prev", &states_prev_tensor));

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

    functor::LSTMCellFprop<Device>()(
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
                        LstmCellOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("LSTMCellBlock")    \
                            .Device(DEVICE_GPU),
                        LstmCellOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
