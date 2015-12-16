#define EIGEN_USE_THREADS

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/user_kernels/gru_op.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/user_kernels/gru_op.h"
#include "tensorflow/core/public/tensor.h"

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// Partial specialization MatMulFunctor<Device=CPUDevice, T>.
template <>
struct MatMulFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d, typename MatMulTypes<float>::out_type out,
      typename MatMulTypes<float>::in_type in0,
      typename MatMulTypes<float>::in_type in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair, float beta) {
    MatMul<CPUDevice>(d, out, in0, in1, dim_pair, beta);
  }
};

}  // end namespace functor

#if GOOGLE_CUDA

namespace {
template <typename T>
perftools::gputools::DeviceMemory<float> AsDeviceMemory(const T* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  perftools::gputools::DeviceMemory<float> typed(wrapped);
  return typed;
}
}  // namespace

#endif  // GOOGLE_CUDA

template <typename Device>
struct LaunchMatMul;

struct LaunchMatMulCPU {
  static void launch(
      OpKernelContext* ctx, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      float beta, Tensor* out) {
    functor::MatMulFunctor<CPUDevice>()(ctx->eigen_device<CPUDevice>(),
                                               out->matrix<float>(), a.matrix<float>(),
                                               b.matrix<float>(), dim_pair, beta);
  }
};

template <>
struct LaunchMatMul<CPUDevice> : public LaunchMatMulCPU {};

#if GOOGLE_CUDA

template <>
struct LaunchMatMul<GPUDevice> {
  static void launch(
      OpKernelContext* ctx, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      float beta, Tensor* out) {
    perftools::gputools::blas::Transpose trans[] = {
        perftools::gputools::blas::Transpose::kNoTranspose,
        perftools::gputools::blas::Transpose::kTranspose};
    const uint64 m = a.dim_size(1 - dim_pair[0].first);
    const uint64 k = a.dim_size(dim_pair[0].first);
    const uint64 n = b.dim_size(1 - dim_pair[0].second);
    bool transpose_a = dim_pair[0].first == 0;
    bool transpose_b = dim_pair[0].second == 1;
    auto blas_transpose_a = trans[transpose_a];
    auto blas_transpose_b = trans[transpose_b];

    auto* stream = ctx->op_device_context<GPUDeviceContext>()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    auto a_ptr = AsDeviceMemory(a.template flat<float>().data());
    auto b_ptr = AsDeviceMemory(b.template flat<float>().data());
    auto c_ptr = AsDeviceMemory(out->template flat<float>().data());

    // Cublas does
    // C = A x B
    // where A, B and C are assumed to be in column major.
    // We want the output to be in row-major, so we can compute
    // C' = B' x A' (' stands for transpose)
    bool blas_launch_status =
        stream->ThenBlasGemm(blas_transpose_b, blas_transpose_a, n, m, k, 1.0f,
                             b_ptr, transpose_b ? k : n, a_ptr,
                             transpose_a ? m : k, beta, &c_ptr, n)
            .ok();
    if (!blas_launch_status) {
      ctx->SetStatus(errors::Internal(
          "Blas SGEMM launch failed : a.shape=(", a.dim_size(0), ", ",
          a.dim_size(1), "), b.shape=(", b.dim_size(0), ", ", b.dim_size(1),
          "), m=", m, ", n=", n, ", k=", k));
    }
  }
};

#endif  // GOOGLE_CUDA

template <typename Device>
void MatMul(OpKernelContext* ctx, bool a_trans, const Tensor& a,
    bool b_trans, const Tensor& b, float beta, Tensor* out) {
  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
  dim_pair[0].first = a_trans ? 0 : 1;
  dim_pair[0].second = b_trans ? 1 : 0;

  LaunchMatMul<Device>::launch(ctx, a, b, dim_pair, beta, out);
}

template <typename Device>
class GruOp : public OpKernel {
 public:
  explicit GruOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sequence_len_max", &sequence_len_max_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* sequence_len = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sequence_len", &sequence_len));

    // Get the sequence length in CPU memory.
    auto sequence_len_t = sequence_len->vec<int64>();
    std::vector<int64> seq_lens_vec(sequence_len_t.size());
    ctx->eigen_device<Device>().memcpyDeviceToHost(
        seq_lens_vec.data(), sequence_len_t.data(),
        sizeof(int64) * sequence_len_t.size());

    // Maximum number of compute unrolls.
    int64 sequence_len_max = std::distance(seq_lens_vec.begin(),
        std::max_element(seq_lens_vec.begin(), seq_lens_vec.end()));
    const int64 batch_size = seq_lens_vec.size();

    // Input.
    OpInputList xs;
    OP_REQUIRES_OK(ctx, ctx->input_list("xs", &xs));

    // Weights.
    const Tensor* wxr = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wxr", &wxr));
    const Tensor* whr = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("whr", &wxr));
    const Tensor* wxz = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wxz", &wxr));
    const Tensor* whz = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("whz", &wxr));
    const Tensor* wxh = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("wxh", &wxr));
    const Tensor* whh = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("whh", &wxr));

    // Outputs.
    OpOutputList rs;
    OP_REQUIRES_OK(ctx, ctx->output_list("rs", &rs));
    OpOutputList zs;
    OP_REQUIRES_OK(ctx, ctx->output_list("zs", &zs));
    OpOutputList rhs;
    OP_REQUIRES_OK(ctx, ctx->output_list("rhs", &rhs));
    OpOutputList gs;
    OP_REQUIRES_OK(ctx, ctx->output_list("gs", &gs));
    OpOutputList hs;
    OP_REQUIRES_OK(ctx, ctx->output_list("hs", &hs));

    if (sequence_len_max >= sequence_len_max_) {
      sequence_len_max = sequence_len_max_;
    }
    for (int64 t = 0; t < sequence_len_max; ++t) {
      const Tensor* x = &xs[t];
      const Tensor* h_prev = t <= 0 ? nullptr : hs[t - 1];
      Tensor* r = nullptr;
      rs.allocate(t, TensorShape({batch_size, cell_size_}), &r);
      Tensor* z = nullptr;
      zs.allocate(t, TensorShape({batch_size, cell_size_}), &z);
      Tensor* rh = nullptr;
      rhs.allocate(t, TensorShape({batch_size, cell_size_}), &rh);
      Tensor* g = nullptr;
      gs.allocate(t, TensorShape({batch_size, cell_size_}), &g);
      Tensor* h = nullptr;
      hs.allocate(t, TensorShape({batch_size, cell_size_}), &h);

      MatMul<Device>(ctx, false, *x, false, *wxr, 0.0f, r);
      if (h_prev) MatMul<Device>(ctx, false, *h_prev, false, *whr, 1.0f, r);
      r->vec<float>().device(ctx->eigen_device<Device>()) =
          r->vec<float>().sigmoid();

      MatMul<Device>(ctx, false, *x, false, *wxz, 0.0f, r);
      if (h_prev) MatMul<Device>(ctx, false, *h_prev, false, *whz, 1.0f, r);
      z->vec<float>().device(ctx->eigen_device<Device>()) =
          z->vec<float>().sigmoid();

      if (h_prev) {
        rh->vec<float>().device(ctx->eigen_device<Device>()) =
            r->vec<float>() * h_prev->vec<float>();
      } else {
        rh->vec<float>().setZero();
      }
      MatMul<Device>(ctx, false, *x, false, *wxh, 0.0f, r);
      if (h_prev) MatMul<Device>(ctx, false, *rh, false, *whh, 1.0f, r);

      g->vec<float>().device(ctx->eigen_device<Device>()) =
          g->vec<float>().tanh();

      h->vec<float>().device(ctx->eigen_device<Device>()) =
          z->vec<float>() * h_prev->vec<float>() + (z->vec<float>().constant(1.0f) - z->vec<float>()) * g->vec<float>();
    }
  }

 protected:
  int64 cell_size_;
  int64 sequence_len_max_;
};

REGISTER_KERNEL_BUILDER(Name("AttentionContextReduce")
                             .Device(DEVICE_CPU),
                        GruOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("AttentionContextReduce")
                             .Device(DEVICE_GPU),
                        GruOp<GPUDevice>);
#endif  // GOOGLE_CUDA
}  // namespace tensorflow
