#define EIGEN_USE_THREADS

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/gru_op.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

namespace functor {

// Partial specialization GruMatMulFunctor<Device=CPUDevice, T>.
template <>
struct GruMatMulFunctor<CPUDevice> {
  void operator()(
      const CPUDevice& d, typename GruMatMulTypes<float>::out_type out,
      typename GruMatMulTypes<float>::in_type in0,
      typename GruMatMulTypes<float>::in_type in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair, float beta) {
    GruMatMul<CPUDevice>(d, out, in0, in1, dim_pair, beta);
  }
};

}  // end namespace functor

#if GOOGLE_CUDA

template <typename T>
perftools::gputools::DeviceMemory<float> AsDeviceMemory(const T* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  perftools::gputools::DeviceMemory<float> typed(wrapped);
  return typed;
}

#endif  // GOOGLE_CUDA

template <typename Device>
struct LaunchGruMatMul;

struct LaunchGruMatMulCPU {
  static void launch(
      OpKernelContext* ctx, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      float beta, Tensor* out) {
    functor::GruMatMulFunctor<CPUDevice>()(ctx->eigen_device<CPUDevice>(),
        out->matrix<float>(), a.matrix<float>(), b.matrix<float>(), dim_pair,
        beta);
  }
};

template <>
struct LaunchGruMatMul<CPUDevice> : public LaunchGruMatMulCPU {};

#if GOOGLE_CUDA

template <>
struct LaunchGruMatMul<GPUDevice> {
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
void GruMatMul(OpKernelContext* ctx, bool a_trans, const Tensor& a,
    bool b_trans, const Tensor& b, float beta, Tensor* out) {
  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
  dim_pair[0].first = a_trans ? 0 : 1;
  dim_pair[0].second = b_trans ? 1 : 0;

  LaunchGruMatMul<Device>::launch(ctx, a, b, dim_pair, beta, out);
}

template <>
struct GruDeviceSynchronize<CPUDevice> {
  void operator()(const CPUDevice& d) {}
};

template <>
struct GruDeviceSynchronize<GPUDevice> {
  void operator()(const GPUDevice& d) {
    d.synchronize();
  }
};

template <>
struct GruSetZero<CPUDevice> {
  void operator()(const CPUDevice& d, Tensor* x) {
    x->matrix<float>().device(d) = x->matrix<float>().constant(0.0f);
  }
};

template <>
struct GruSetZero<GPUDevice> {
  void operator()(const GPUDevice& d, Tensor* x) {
    GruSetZeroGPU(d, x);
  }
};

template <>
struct GruActivationSigmoid<CPUDevice> {
  void operator()(const CPUDevice& d, Tensor* x) {
    x->matrix<float>().device(d) = x->matrix<float>().sigmoid();
  }
};

template <>
struct GruActivationSigmoid<GPUDevice> {
  void operator()(const GPUDevice& d, Tensor* x) {
    GruActivationSigmoidGPU(d, x);
  }
};

template <>
struct GruActivationTanh<CPUDevice> {
  void operator()(const CPUDevice& d, Tensor* x) {
    x->matrix<float>().device(d) = x->matrix<float>().tanh();
  }
};

template <>
struct GruActivationTanh<GPUDevice> {
  void operator()(const GPUDevice& d, Tensor* x) {
    GruActivationTanhGPU(d, x);
  }
};

template <>
struct GruCWiseMult<CPUDevice> {
  void operator()(const CPUDevice& d, const Tensor& a, const Tensor& b, Tensor* c) {
    c->matrix<float>().device(d) = a.matrix<float>() * b.matrix<float>();
  }
};

template <>
struct GruCWiseMult<GPUDevice> {
  void operator()(const GPUDevice& d, const Tensor& a, const Tensor& b, Tensor* c) {
    GruCWiseMultGPU(d, a, b, c);
  }
};

template <>
struct GruH<CPUDevice> {
  void operator()(
      const CPUDevice& d, const Tensor& z, const Tensor& h_prev, const Tensor& g, Tensor* h) {
    h->matrix<float>().device(d) =
        z.matrix<float>() * h_prev.matrix<float>() +
        (z.matrix<float>().constant(1.0f) - z.matrix<float>()) * g.matrix<float>();
  }
};

template <>
struct GruH<GPUDevice> {
  void operator()(
      const GPUDevice& d, const Tensor& z, const Tensor& h_prev, const Tensor& g, Tensor* h) {
    GruHGPU(d, z, h_prev, g, h);
  }
};

// Partial specialization GruMatMulFunctor<Device=CPUDevice, T>.
template <typename Device>
class GruOp : public OpKernel {
 public:
  explicit GruOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sequence_len_max", &sequence_len_max_));
  }

  void Compute(OpKernelContext* ctx) override {
    LOG(INFO) << "OK...";
    const Tensor* sequence_len = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sequence_len", &sequence_len));

    LOG(INFO) << "OK...";
    // Get the sequence length in CPU memory.
    auto sequence_len_t = sequence_len->vec<int64>();
    std::vector<int64> seq_lens_vector(sequence_len_t.size());
    ctx->eigen_device<Device>().memcpyDeviceToHost(
        seq_lens_vector.data(), sequence_len_t.data(),
        sizeof(int64) * sequence_len_t.size());
    GruDeviceSynchronize<Device>()(ctx->eigen_device<Device>());
    LOG(INFO) << "OK...";

    // Maximum number of compute unrolls.
    int64 sequence_len_max =
        *std::max_element(seq_lens_vector.begin(), seq_lens_vector.end());
    const int64 batch_size = seq_lens_vector.size();

  #define INPUT_LIST(T)                                                        \
    OpInputList T;                                                             \
    OP_REQUIRES_OK(ctx, ctx->input_list(#T, &T));

    LOG(INFO) << "OK...";
    INPUT_LIST(xs);

  #define INPUT_TENSOR(T)                                                      \
    const Tensor* T = nullptr;                                                 \
    OP_REQUIRES_OK(ctx, ctx->input(#T, &T));

    INPUT_TENSOR(wxr);
    INPUT_TENSOR(whr);
    INPUT_TENSOR(wxz);
    INPUT_TENSOR(whz);
    INPUT_TENSOR(wxh);
    INPUT_TENSOR(whh);

    CHECK_EQ(wxr->dims(), 2);
    CHECK_EQ(whr->dims(), 2);
    CHECK_EQ(wxz->dims(), 2);
    CHECK_EQ(whz->dims(), 2);
    CHECK_EQ(wxh->dims(), 2);
    CHECK_EQ(whh->dims(), 2);

  #define OUTPUT_LIST(T)                                                       \
    OpOutputList T;                                                            \
    OP_REQUIRES_OK(ctx, ctx->output_list(#T, &T));

    OUTPUT_LIST(rs);
    OUTPUT_LIST(zs);
    OUTPUT_LIST(rhs);
    OUTPUT_LIST(gs);
    OUTPUT_LIST(hs);

    LOG(INFO) << "OK...";
    for (int64 t = 0; t < sequence_len_max_; ++t) {
  #define OUTPUT_LIST_ALLOCATE(OP_OUTPUT_LIST, NAME, SHAPE)                    \
      Tensor* NAME = nullptr;                                                  \
      OP_OUTPUT_LIST.allocate(t, SHAPE, &NAME);                                \
      GruSetZero<Device>()(ctx->eigen_device<Device>(), NAME);

      OUTPUT_LIST_ALLOCATE(rs, r, TensorShape({batch_size, cell_size_}));
      OUTPUT_LIST_ALLOCATE(zs, z, TensorShape({batch_size, cell_size_}));
      OUTPUT_LIST_ALLOCATE(rhs, rh, TensorShape({batch_size, cell_size_}));
      OUTPUT_LIST_ALLOCATE(gs, g, TensorShape({batch_size, cell_size_}));
      OUTPUT_LIST_ALLOCATE(hs, h, TensorShape({batch_size, cell_size_}));
    }

    if (sequence_len_max >= sequence_len_max_) {
      sequence_len_max = sequence_len_max_;
    }
    LOG(INFO) << sequence_len_max;
    for (int64 t = 0; t < sequence_len_max; ++t) {
      const Tensor x = xs[t];
      CHECK_EQ(x.dims(), 2);
      const Tensor* h_prev = t <= 0 ? nullptr : hs[t - 1];

      Tensor* r = rs[t];
      Tensor* z = zs[t];
      Tensor* rh = rhs[t];
      Tensor* g = gs[t];
      Tensor* h = hs[t];

      // r[t] = sigm(x[t] Wxr + h[t - 1] Whr)
      GruMatMul<Device>(ctx, false, x, false, *wxr, 0.0f, r);
      if (t >= 0) GruMatMul<Device>(ctx, false, *h_prev, false, *whr, 1.0f, r);
      GruActivationSigmoid<Device>()(ctx->eigen_device<Device>(), r);

      // z[t] = sigm(x[t] Wxz + h[t - 1] Whz)
      GruMatMul<Device>(ctx, false, x, false, *wxz, 0.0f, z);
      if (t >= 0) GruMatMul<Device>(ctx, false, *h_prev, false, *whz, 1.0f, z);
      GruActivationSigmoid<Device>()(ctx->eigen_device<Device>(), z);

      // rh[t] = r[t] .* h_prev[t]
      // g[t] = tanh(x[t] Wxh + rh[t] Whh)
      if (t >= 0) {
        GruCWiseMult<Device>()(ctx->eigen_device<Device>(), *r, *h_prev, rh);
      } else {
        rh->matrix<float>().setZero();
      }
      GruMatMul<Device>(ctx, false, x, false, *wxh, 0.0f, g);
      if (h_prev) GruMatMul<Device>(ctx, false, *rh, false, *whh, 1.0f, g);
      GruActivationTanh<Device>()(ctx->eigen_device<Device>(), g);

      // h[t] = z[t] .* h[t - 1] + (1 - z[t]) .* g[t]
      GruH<Device>()(ctx->eigen_device<Device>(), *z, *h_prev, *g, h);
    }
  }

 protected:
  int64 cell_size_;
  int64 sequence_len_max_;
};

REGISTER_KERNEL_BUILDER(Name("Gru")
                             .Device(DEVICE_CPU),
                        GruOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Gru")
                             .Device(DEVICE_GPU),
                        GruOp<GPUDevice>);
#endif  // GOOGLE_CUDA

template <typename Device>
class GruGradOp : public OpKernel {
 public:
  explicit GruGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sequence_len_max", &sequence_len_max_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* sequence_len = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sequence_len", &sequence_len));

    // Get the sequence length in CPU memory.
    auto sequence_len_t = sequence_len->vec<int64>();
    std::vector<int64> seq_lens_vector(sequence_len_t.size());
    ctx->eigen_device<Device>().memcpyDeviceToHost(
        seq_lens_vector.data(), sequence_len_t.data(),
        sizeof(int64) * sequence_len_t.size());
    GruDeviceSynchronize<Device>()(ctx->eigen_device<Device>());

    // Maximum number of compute unrolls.
    int64 sequence_len_max =
        *std::max_element(seq_lens_vector.begin(), seq_lens_vector.end());

    INPUT_LIST(xs);
    INPUT_LIST(rs);
    INPUT_LIST(zs);
    INPUT_LIST(rhs);
    INPUT_LIST(gs);
    INPUT_LIST(hs);

    INPUT_TENSOR(wxr);
    INPUT_TENSOR(whr);
    INPUT_TENSOR(wxz);
    INPUT_TENSOR(whz);
    INPUT_TENSOR(wxh);
    INPUT_TENSOR(whh);

#define MUTABLE_INPUT_LIST(T)                                                  \
    OpMutableInputList T;                                                      \
    OP_REQUIRES_OK(ctx, ctx->mutable_input_list(#T, &T));

    // Gradients.
    MUTABLE_INPUT_LIST(drs);
    MUTABLE_INPUT_LIST(dzs);
    MUTABLE_INPUT_LIST(drhs);
    MUTABLE_INPUT_LIST(dgs);
    MUTABLE_INPUT_LIST(dhs);

#define OUTPUT_TENSOR(NAME, SHAPE)                                             \
    Tensor* NAME = nullptr;                                                    \
    ctx->allocate_output(#NAME, SHAPE, &NAME);
    OUTPUT_TENSOR(dwxr, wxr->shape());
    OUTPUT_TENSOR(dwhr, whr->shape());
    OUTPUT_TENSOR(dwxz, wxz->shape());
    OUTPUT_TENSOR(dwhz, whz->shape());
    OUTPUT_TENSOR(dwxh, wxh->shape());
    OUTPUT_TENSOR(dwhh, whh->shape());

    OUTPUT_LIST(dxs);

    if (sequence_len_max >= sequence_len_max_) {
      sequence_len_max = sequence_len_max_;
    }
    for (int64 t = sequence_len_max - 1; t >= 0; --t) {
      const Tensor x = xs[t];
      const Tensor r = rs[t];
      const Tensor z = zs[t];
      const Tensor rh = rhs[t];
      const Tensor g = gs[t];
      const Tensor h = hs[t];
      const Tensor dh = dhs.at(t, false);
      Tensor dh_prev = t <= 0 ? dhs.at(0, false) : dhs.at(t - 1, false);

      const Tensor* h_prev = t <= 0 ? nullptr : &hs[t - 1];

      Tensor dr = drs.at(t, false);
      Tensor dz = dzs.at(t, false);
      Tensor drh = drhs.at(t, false);
      Tensor dg = dgs.at(t, false);

      OUTPUT_LIST_ALLOCATE(dxs, dx, x.shape());

      // h[t] = z[t] .* h[t - 1] + (1 - z[t]) .* g[t]
      // dz.matrix<float>().device(ctx->eigen_device<Device>()) =
      //     dh.matrix<float>() * h_prev->matrix<float>() -
      //     dh.matrix<float>() * g.matrix<float>();

      // dh_prev.matrix<float>().device(ctx->eigen_device<Device>()) =
      //     dh_prev.matrix<float>() + dh.matrix<float>() * z.matrix<float>();

      // dg.matrix<float>().device(ctx->eigen_device<Device>()) =
      //     dh.matrix<float>() * (z.matrix<float>().constant(1.0f) - z.matrix<float>());

      // // g[t] = tanh(x[t] Wxh + rh[t] Whh)
      // dg.matrix<float>().device(ctx->eigen_device<Device>()) = dg.matrix<float>() *
      //     (g.matrix<float>().constant(1.0f) - g.matrix<float>() * g.matrix<float>());
      // GruMatMul<Device>(ctx, false, dg, true, *wxh, 0.0f, dx);
      // GruMatMul<Device>(ctx, false, dg, true, *whh, 0.0f, &drh);

      // if (t > 0) {
      //   dr.matrix<float>().device(ctx->eigen_device<Device>()) =
      //       drh.matrix<float>() * h_prev->matrix<float>();
      //   dh_prev.matrix<float>().device(ctx->eigen_device<Device>()) =
      //       dh_prev.matrix<float>() + drh.matrix<float>() * r.matrix<float>();
      // } else {
      //   dr.matrix<float>().setZero();
      // }

      // // z[t] = sigm(x[t] Wxz + h[t - 1] Whz)
      // dz.matrix<float>().device(ctx->eigen_device<Device>()) = dz.matrix<float>() *
      //     z.matrix<float>() * (z.matrix<float>().constant(1.0f) - z.matrix<float>());
      // GruMatMul<Device>(ctx, false, dz, true, *wxz, 1.0f, dx);
      // if (t > 0) GruMatMul<Device>(ctx, false, dz, true, *whz, 1.0f, &dh_prev);

      // // r[t] = sigm(x[t] Wxr + h[t - 1] Whr)
      // if (t > 0) {
      //   dr.matrix<float>().device(ctx->eigen_device<Device>()) = dr.matrix<float>() *
      //       r.matrix<float>() * (r.matrix<float>().constant(1.0f) - r.matrix<float>());
      //   GruMatMul<Device>(ctx, false, dr, true, *wxr, 1.0f, dx);
      //   GruMatMul<Device>(ctx, false, dr, true, *whr, 1.0f, &dh_prev);
      // }
    }

    for (int64 t = 0; t < sequence_len_max; ++t) {
      const Tensor x = xs[t];
      const Tensor rh = rhs[t];
      const Tensor h = hs[t];

      const Tensor dr = drs.at(t, false);
      const Tensor dz = dzs.at(t, false);
      const Tensor dg = dgs.at(t, false);

      GruMatMul<Device>(ctx, true, x, false, dr, t > 0, dwxr);
      GruMatMul<Device>(ctx, true, h, false, dr, t > 0, dwhr);
      GruMatMul<Device>(ctx, true, x, false, dz, t > 0, dwxz);
      GruMatMul<Device>(ctx, true, h, false, dz, t > 0, dwhz);
      GruMatMul<Device>(ctx, true, x, false, dg, t > 0, dwxh);
      GruMatMul<Device>(ctx, true, rh, false, dg, t > 0, dwhh);
    }
  }

 protected:
  int64 cell_size_;
  int64 sequence_len_max_;
};

REGISTER_KERNEL_BUILDER(Name("GruGrad")
                             .Device(DEVICE_CPU),
                        GruGradOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("GruGrad")
                             .Device(DEVICE_GPU),
                        GruGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
