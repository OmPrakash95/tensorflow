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

#if GOOGLE_CUDA

namespace {
perftools::gputools::DeviceMemory<float> AsDeviceMemory(const float* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<float*>(cuda_memory));
  perftools::gputools::DeviceMemory<float> typed(wrapped);
  return typed;
}

void TensorMemZero(Tensor* tensor, perftools::gputools::Stream* stream) {
  auto ptr = AsDeviceMemory(tensor->flat<float>().data());
  if (stream) {
    stream->ThenMemZero(&ptr, tensor->TotalBytes());
  } else {
    std::memset(tensor->flat<float>().data(), 0, tensor->TotalBytes());
  }
}
}  // namespace

#endif  // GOOGLE_CUDA

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, bool USE_CUBLAS>
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

    perftools::gputools::Stream* stream =
        ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

    // Sanity checks for our input shapes.
    CHECK_EQ(states_prev_tensor->dim_size(0), batch_size);
    CHECK_EQ(states_prev_tensor->dim_size(1), cell_size_ * 7);

    // CHECK_EQ(w_tensor->dim_size(0), input_size + cell_size_);
    // CHECK_EQ(w_tensor->dim_size(1), cell_size_ * 4);

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

    functor::LSTMCellBlockFprop<Device, USE_CUBLAS>()(
        stream, ctx->eigen_device<Device>(),
        batch_size, input_size, cell_size_, forget_bias_,
        x_tensor->matrix<float>(),
        xh_tensor.matrix<float>(), states_prev_tensor->matrix<float>(),
        w_tensor->matrix<float>(), b_tensor->vec<float>(),
        h_tensor->matrix<float>(), states_tensor->matrix<float>());
  }

 private:
  int64 cell_size_;
  float forget_bias_;
};

REGISTER_KERNEL_BUILDER(Name("LSTMCellBlock")    \
                            .Device(DEVICE_CPU),
                        LSTMCellBlockOp<CPUDevice, false>);

#if GOOGLE_CUDA
namespace functor {
  template <>
  void LSTMCellBlockFprop<GPUDevice, true>::operator()(
      perftools::gputools::Stream* stream, const GPUDevice& d, const int batch_size, const int input_size,
      const int cell_size, const float forget_bias,
      typename TTypes<float>::ConstMatrix x,
      typename TTypes<float>::Matrix xh,
      typename TTypes<float>::ConstMatrix states_prev,
      typename TTypes<float>::ConstMatrix w,
      typename TTypes<float>::ConstVec b,
      typename TTypes<float>::Matrix h,
      typename TTypes<float>::Matrix states);
  extern template struct LSTMCellBlockFprop<GPUDevice, true>;
}  // end namespace functor

REGISTER_KERNEL_BUILDER(Name("LSTMCellBlock")     \
                            .Device(DEVICE_GPU),  \
                        LSTMCellBlockOp<GPUDevice, true>);
#endif  // GOOGLE_CUDA

template <typename Device, bool USE_CUBLAS>
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

    perftools::gputools::Stream* stream =
        ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

    // Sanity checks for our input shapes.
    CHECK_EQ(states_prev_tensor->dim_size(0), batch_size);
    CHECK_EQ(states_prev_tensor->dim_size(1), cell_size_ * 7);

    // CHECK_EQ(w_tensor->dim_size(0), input_size + cell_size_);
    // CHECK_EQ(w_tensor->dim_size(1), cell_size_ * 4);

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
    TensorMemZero(w_grad_tensor, stream);

    Tensor* b_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("b_grad",
          TensorShape({cell_size_ * 4}), &b_grad_tensor));
    TensorMemZero(b_grad_tensor, stream);

    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, input_size + cell_size_}), &xh_tensor));

    Tensor xh_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, input_size + cell_size_}), &xh_grad_tensor));

    functor::LSTMCellBlockBprop<Device, USE_CUBLAS>()(
        stream, ctx->eigen_device<Device>(),
        batch_size, input_size, cell_size_,
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
                        LSTMCellBlockGradOp<CPUDevice, false>);

#if GOOGLE_CUDA
namespace functor {
  template <>
  void LSTMCellBlockBprop<GPUDevice, true>::operator()(
      perftools::gputools::Stream* stream, const GPUDevice& d,
      const int batch_size, const int input_size, const int cell_size,
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
  extern template struct LSTMCellBlockBprop<GPUDevice, true>;
}  // namespace functor

REGISTER_KERNEL_BUILDER(Name("LSTMCellBlockGrad")  \
                            .Device(DEVICE_GPU),   \
                        LSTMCellBlockGradOp<GPUDevice, true>);
#endif  // GOOGLE_CUDA

template <typename Device, bool USE_CUBLAS>
class LSTMBlockOp : public OpKernel {
 public:
  explicit LSTMBlockOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sequence_len_max", &sequence_len_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("forget_bias", &forget_bias_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* sequence_len_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sequence_len", &sequence_len_tensor));

    OpInputList x_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("x", &x_tensors));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    OpOutputList h_tensors;
    OP_REQUIRES_OK(ctx, ctx->output_list("h", &h_tensors));

    OpOutputList states_tensors;
    OP_REQUIRES_OK(ctx, ctx->output_list("states", &states_tensors));

    auto sequence_len_t = sequence_len_tensor->vec<int64>();
    std::vector<int64> seq_lens_vector(sequence_len_t.size());
    ctx->eigen_device<Device>().memcpyDeviceToHost(
        seq_lens_vector.data(), sequence_len_t.data(),
        sizeof(int64) * sequence_len_t.size());

    const int64 batch_size = x_tensors[0].dim_size(0);
    const int64 input_size = x_tensors[0].dim_size(1);
    const int64 sequence_len_max =
        *std::max_element(seq_lens_vector.begin(), seq_lens_vector.end());

    perftools::gputools::Stream* stream =
        ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

    for (int64 t = 0; t < sequence_len_max_; ++t ) {
      Tensor* states_tensor = nullptr;
      states_tensors.allocate(
          t, TensorShape({batch_size, cell_size_ * 7}), &states_tensor);
      TensorMemZero(states_tensor, stream);

      Tensor* h_tensor = nullptr;
      h_tensors.allocate(
          t, TensorShape({batch_size, cell_size_}), &h_tensor);
      TensorMemZero(h_tensor, stream);
    }

    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
          DT_FLOAT, TensorShape({batch_size, input_size + cell_size_}),
          &xh_tensor));
    
    for (int64 t = 0; t < sequence_len_max; ++t) {
      const Tensor x_tensor = x_tensors[t];
      const Tensor* states_prev_tensor =
          t <= 0 ? states_tensors[0] : states_tensors[t - 1];

      Tensor* states_tensor = states_tensors[t];
      Tensor* h_tensor = h_tensors[t];

      functor::LSTMCellBlockFprop<Device, USE_CUBLAS>()(
        stream, ctx->eigen_device<Device>(),
        batch_size, input_size, cell_size_, forget_bias_,
        x_tensor.matrix<float>(),
        xh_tensor.matrix<float>(),
        states_prev_tensor->matrix<float>(),
        w_tensor->matrix<float>(),
        b_tensor->vec<float>(),
        h_tensor->matrix<float>(),
        states_tensor->matrix<float>());
    }
  }

 private:
  int64 sequence_len_max_;
  int64 cell_size_;
  float forget_bias_;
};

REGISTER_KERNEL_BUILDER(Name("LSTMBlock")         \
                            .Device(DEVICE_CPU),  \
                        LSTMBlockOp<CPUDevice, false>);

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("LSTMBlock")         \
                            .Device(DEVICE_GPU),  \
                        LSTMBlockOp<GPUDevice, true>);
#endif  // GOOGLE_CUDA

template <typename Device, bool USE_CUBLAS>
class LSTMBlockGradOp : public OpKernel {
 public:
  explicit LSTMBlockGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sequence_len_max", &sequence_len_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* sequence_len_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sequence_len", &sequence_len_tensor));

    OpInputList x_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("x", &x_tensors));

    const Tensor* w_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("w", &w_tensor));

    const Tensor* b_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("b", &b_tensor));

    OpInputList h_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("h", &h_tensors));

    OpInputList states_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("states", &states_tensors));

    OpInputList h_grad_tensors;
    OP_REQUIRES_OK(ctx, ctx->input_list("h_grad", &h_grad_tensors));

    auto sequence_len_t = sequence_len_tensor->vec<int64>();
    std::vector<int64> seq_lens_vector(sequence_len_t.size());
    ctx->eigen_device<Device>().memcpyDeviceToHost(
        seq_lens_vector.data(), sequence_len_t.data(),
        sizeof(int64) * sequence_len_t.size());

    const int64 batch_size = x_tensors[0].dim_size(0);
    const int64 input_size = x_tensors[0].dim_size(1);
    const int64 sequence_len_max =
        *std::max_element(seq_lens_vector.begin(), seq_lens_vector.end());

    perftools::gputools::Stream* stream =
        ctx->op_device_context() ? ctx->op_device_context()->stream() : nullptr;

    OpOutputList x_grad_tensors;
    OP_REQUIRES_OK(ctx, ctx->output_list("x_grad", &x_grad_tensors));

    Tensor* w_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("w_grad",
          TensorShape({input_size + cell_size_, cell_size_ * 4}),
          &w_grad_tensor));
    TensorMemZero(w_grad_tensor, stream);

    Tensor* b_grad_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("b_grad",
          TensorShape({cell_size_ * 4}), &b_grad_tensor));
    TensorMemZero(b_grad_tensor, stream);

    for (int64 t = 0; t < sequence_len_max_; ++t) {
      Tensor* x_grad_tensor = nullptr;
      x_grad_tensors.allocate(
          t, TensorShape({batch_size, input_size}), &x_grad_tensor);
      TensorMemZero(x_grad_tensor, stream);
    }

    Tensor xh_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(
          DT_FLOAT, TensorShape({batch_size, input_size + cell_size_}),
          &xh_tensor));

    Tensor xh_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, input_size + cell_size_}), &xh_grad_tensor));

    Tensor states_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_ * 7}), &states_grad_tensor));
    TensorMemZero(&states_grad_tensor, stream);

    Tensor states_prev_grad_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_ * 7}), &states_prev_grad_tensor));

    Tensor states_zero_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT,
        TensorShape({batch_size, cell_size_ * 7}), &states_zero_tensor));
    TensorMemZero(&states_zero_tensor, stream);

    for (int64 t = sequence_len_max - 1; t >= 0; --t) {
      const Tensor x_tensor = x_tensors[t];
      const Tensor states_prev_tensor =
          t <= 0 ? states_zero_tensor : states_tensors[t - 1];
      const Tensor states_tensor = states_tensors[t];
      const Tensor h_tensor = h_tensors[t];
      const Tensor h_grad_tensor = h_grad_tensors[t];

      Tensor* x_grad_tensor = x_grad_tensors[t];
      const Tensor states_grad_const_tensor = states_grad_tensor;

      functor::LSTMCellBlockBprop<Device, USE_CUBLAS>()(
          stream, ctx->eigen_device<Device>(),
          batch_size, input_size, cell_size_,
          x_tensor.matrix<float>(),
          xh_tensor.matrix<float>(),
          states_prev_tensor.matrix<float>(),
          w_tensor->matrix<float>(),
          b_tensor->vec<float>(),
          h_tensor.matrix<float>(),
          states_tensor.matrix<float>(),
          h_grad_tensor.matrix<float>(),
          states_grad_const_tensor.matrix<float>(),
          xh_grad_tensor.matrix<float>(),
          x_grad_tensor->matrix<float>(),
          states_prev_grad_tensor.matrix<float>(),
          w_grad_tensor->matrix<float>(),
          b_grad_tensor->vec<float>()
      );

      if (stream) {
        auto states_grad_ptr =
            AsDeviceMemory(states_grad_tensor.flat<float>().data());
        auto states_prev_grad_ptr =
            AsDeviceMemory(states_prev_grad_tensor.flat<float>().data());
        stream->ThenMemcpy(
            &states_grad_ptr, states_prev_grad_ptr,
            states_grad_tensor.TotalBytes());
      } else {
        std::memcpy(states_grad_tensor.flat<float>().data(),
                    states_prev_grad_tensor.flat<float>().data(),
                    states_grad_tensor.TotalBytes());
      }
    }
  }

 private:
  int64 sequence_len_max_;
  int64 cell_size_;
};

REGISTER_KERNEL_BUILDER(Name("LSTMBlockGrad")     \
                            .Device(DEVICE_CPU),  \
                        LSTMBlockGradOp<CPUDevice, false>);

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("LSTMBlockGrad")     \
                            .Device(DEVICE_GPU),  \
                        LSTMBlockGradOp<GPUDevice, true>);
#endif  // GOOGLE_CUDA
}  // end namespace tensorflow
