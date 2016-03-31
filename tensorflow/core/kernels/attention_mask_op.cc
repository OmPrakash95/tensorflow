#define EIGEN_USE_THREADS

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/attention_mask_op.h"

#include <random>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/guarded_philox_random.h"

namespace tensorflow {

template <>
struct AttentionMask<CPUDevice> {
  void operator()(
      const CPUDevice& d, float fill_value, const Tensor& sequence_len,
      const Tensor& input, Tensor* output) {
    generator::AttentionMaskGenerator generator(
        fill_value, sequence_len.vec<int64>(), input.matrix<float>());
    output->matrix<float>().device(d) = input.matrix<float>().generate(generator);
  }
};

template <>
struct AttentionMask<GPUDevice> {
  void operator()(
      const GPUDevice& d, float fill_value, const Tensor& sequence_len,
      const Tensor& input, Tensor* output) {
    AttentionMaskGPU(d, fill_value, sequence_len, input, output);
  }
};

template <>
struct AttentionMaskMedian<GPUDevice> {
  void operator()(
      const GPUDevice& d, float fill_value, int64 window_l, int64 window_r,
      const Tensor& sequence_len, const Tensor& input, const Tensor& median,
      Tensor* output) {
    AttentionMaskMedianGPU(
        d, fill_value, window_l, window_r, sequence_len, input, median, output);
  }
};

template <>
struct AttentionMaskWindow<GPUDevice> {
  void operator()(
      const GPUDevice& d, float fill_value, int64 s_min, int64 s_max,
      float v_min, float v_max, int64 index, const Tensor& sequence_len,
      const Tensor& input, Tensor* output) {
    AttentionMaskWindowGPU(
        d, fill_value, s_min, s_max, v_min, v_max, index, sequence_len, input,
        output);
  }
};

template <>
struct ComputeMedian<GPUDevice> {
  void operator()(
      const GPUDevice& d, const Tensor& input, Tensor* median) {
    ComputeMedianGPU(d, input, median);
  }
};

template <>
struct SetZero<CPUDevice> {
  void operator()(const CPUDevice& d, Tensor* x) {
    x->matrix<float>().device(d) = x->matrix<float>().constant(0.0f);
  }
};

template <>
struct SetZero<GPUDevice> {
  void operator()(const GPUDevice& d, Tensor* x) {
    SetZeroGPU(d, x);
  }
};

template <typename Device>
class AttentionMaskOp : public OpKernel {
 public:
  explicit AttentionMaskOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_value", &fill_value_));
  }

  void Compute(OpKernelContext* ctx) override {
    #define INPUT_TENSOR(T)                                                    \
      const Tensor* T = nullptr;                                               \
      OP_REQUIRES_OK(ctx, ctx->input(#T, &T));

    INPUT_TENSOR(attention_states_sequence_len);
    INPUT_TENSOR(input);

    #define OUTPUT_TENSOR(NAME, SHAPE)                                         \
    Tensor* NAME = nullptr;                                                    \
    ctx->allocate_output(#NAME, SHAPE, &NAME);                                 \
    SetZero<Device>()(ctx->eigen_device<Device>(), NAME);

    OUTPUT_TENSOR(output, input->shape());

    AttentionMask<Device>()(
        ctx->eigen_device<Device>(), fill_value_, *attention_states_sequence_len,
        *input, output);
  }

 private:
  float fill_value_;
};

REGISTER_KERNEL_BUILDER(Name("AttentionMask")
                             .Device(DEVICE_CPU),
                        AttentionMaskOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("AttentionMask")
                             .Device(DEVICE_GPU),
                        AttentionMaskOp<GPUDevice>);
#endif  // GOOGLE_CUDA

template <typename Device>
class AttentionMaskMedianOp : public OpKernel {
 public:
  explicit AttentionMaskMedianOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_value", &fill_value_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_l", &window_l_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_r", &window_r_));
  }

  void Compute(OpKernelContext* ctx) override {
    INPUT_TENSOR(attention_states_sequence_len);
    INPUT_TENSOR(input);
    INPUT_TENSOR(prev);

    OUTPUT_TENSOR(output, input->shape());
    int64 batch_size = output->dim_size(0);

    Tensor median;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT32, TensorShape({batch_size}), &median));

    ComputeMedian<Device>()(
        ctx->eigen_device<Device>(), *prev, &median);

    AttentionMaskMedian<Device>()(
        ctx->eigen_device<Device>(), fill_value_, window_l_, window_r_,
        *attention_states_sequence_len, *input, median, output);
  }

 private:
  float fill_value_;
  int64 window_l_;
  int64 window_r_;
};

// REGISTER_KERNEL_BUILDER(Name("AttentionMaskMedian")
//                              .Device(DEVICE_CPU),
//                         AttentionMaskMedianOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("AttentionMaskMedian")
                             .Device(DEVICE_GPU),
                        AttentionMaskMedianOp<GPUDevice>);
#endif  // GOOGLE_CUDA

template <typename Device>
class AttentionMaskWindowOp : public OpKernel {
 public:
  explicit AttentionMaskWindowOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_value", &fill_value_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("s_min", &s_min_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("s_max", &s_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("v_min", &v_min_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("v_max", &v_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("index", &index_));
  }

  void Compute(OpKernelContext* ctx) override {
    INPUT_TENSOR(attention_states_sequence_len);
    INPUT_TENSOR(input);

    OUTPUT_TENSOR(output, input->shape());

    AttentionMaskWindow<Device>()(
        ctx->eigen_device<Device>(), fill_value_, s_min_, s_max_, v_min_,
        v_max_, index_, *attention_states_sequence_len, *input, output);
  }

 private:
  float fill_value_;
  int64 s_min_;
  int64 s_max_;
  float v_min_;
  float v_max_;
  int64 index_;
};

// REGISTER_KERNEL_BUILDER(Name("AttentionMaskWindow")
//                              .Device(DEVICE_CPU),
//                         AttentionMaskWindowOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("AttentionMaskWindow")
                             .Device(DEVICE_GPU),
                        AttentionMaskWindowOp<GPUDevice>);
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
