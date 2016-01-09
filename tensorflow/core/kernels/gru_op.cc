#define EIGEN_USE_THREADS

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/gru_op.h"

#include <random>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/guarded_philox_random.h"

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
    GruDeviceSynchronizeGPU(d);
  }
};

template <>
struct GruSetZero<CPUDevice> {
  void operator()(const CPUDevice& d, Tensor* x) {
    x->flat<float>().device(d) = x->flat<float>().constant(0.0f);
  }
};

template <>
struct GruSetZero<GPUDevice> {
  void operator()(const GPUDevice& d, Tensor* x) {
    GruSetZeroGPU(d, x);
  }
};

template <>
struct GruCopy<CPUDevice> {
  void operator()(const CPUDevice& d, const Tensor& x, Tensor* y) {
    y->matrix<float>().device(d) = x.matrix<float>();
  }
};

template <>
struct GruCopy<GPUDevice> {
  void operator()(const GPUDevice& d, const Tensor& x, Tensor* y) {
    GruCopyGPU(d, x, y);
  }
};

template <>
struct GruBias<CPUDevice> {
  void operator()(const CPUDevice& d, const Tensor& b, Tensor* y) {
    const int batch_size = y->dim_size(0);
    y->matrix<float>().device(d) =
        y->matrix<float>() + b.vec<float>().broadcast(Eigen::array<int, 2>({batch_size, 1}));
  }
};

template <>
struct GruBias<GPUDevice> {
  void operator()(const GPUDevice& d, const Tensor& b, Tensor* y) {
    GruBiasGPU(d, b, y);
  }
};

template <>
struct GruBiasGrad<CPUDevice> {
  void operator()(const CPUDevice& d, const Tensor& dx, Tensor* db) {
    db->vec<float>().device(d) =
        db->vec<float>() + dx.matrix<float>().sum(Eigen::array<int, 1>(0));
  }
};

template <>
struct GruBiasGrad<GPUDevice> {
  void operator()(const GPUDevice& d, const Tensor& dx, Tensor* db) {
    GruBiasGradGPU(d, dx, db);
  }
};

template <>
struct GruWxhrz<CPUDevice> {
  void operator()(
      const CPUDevice& d, const Tensor& wxr, const Tensor& whr, const Tensor& wxz,
      const Tensor& whz, Tensor* w) {
  const int x_size = wxr.dim_size(0);
  const int h_size = whr.dim_size(0);

  w->matrix<float>().slice(
      Eigen::array<int, 2>(0, 0),           Eigen::array<int, 2>(x_size, h_size)).device(d) =
      wxr.matrix<float>();
  w->matrix<float>().slice(
      Eigen::array<int, 2>(x_size, 0),      Eigen::array<int, 2>(h_size, h_size)).device(d) =
      whr.matrix<float>();

  w->matrix<float>().slice(
      Eigen::array<int, 2>(0, h_size),      Eigen::array<int, 2>(x_size, h_size)).device(d) =
      wxz.matrix<float>();
  w->matrix<float>().slice(
      Eigen::array<int, 2>(x_size, h_size), Eigen::array<int, 2>(h_size, h_size)).device(d) =
      whz.matrix<float>();
  }
};

template <>
struct GruWxhrz<GPUDevice> {
  void operator()(
      const GPUDevice& d, const Tensor& wxr, const Tensor& whr, const Tensor& wxz,
      const Tensor& whz, Tensor* w) {
    GruWxhrzGPU(d, wxr, whr, wxz, whz, w);
  }
};

template <>
struct GruXH<CPUDevice> {
  void operator()(
      const CPUDevice& d, const Tensor& x, const Tensor* h, Tensor* xh) {
    const int batch_size = x.dim_size(0);
    const int x_size = x.dim_size(1);

    xh->matrix<float>().slice(
        Eigen::array<int, 2>({0, 0}),      Eigen::array<int, 2>({batch_size, x_size})).device(d) =
        x.matrix<float>();
    if (h) {
      const int h_size = h->dim_size(1);
      xh->matrix<float>().slice(
          Eigen::array<int, 2>({0, x_size}), Eigen::array<int, 2>({batch_size, h_size})).device(d) =
          h->matrix<float>();
    }
  }
};

template <>
struct GruXH<GPUDevice> {
  void operator()(
      const GPUDevice& d, const Tensor& x, const Tensor* h, Tensor* xh) {
    GruXHGPU(d, x, h, xh);
  }
};

template <>
struct GruRZ<CPUDevice> {
  void operator()(
      const CPUDevice& d, const Tensor& rz, Tensor* r, Tensor* z) {
    const int batch_size = rz.dim_size(0);
    const int h_size = r->dim_size(1);

    r->matrix<float>().device(d) =
        rz.matrix<float>().slice(
            Eigen::array<int, 2>({0, 0}),      Eigen::array<int, 2>({batch_size, h_size}));
    z->matrix<float>().device(d) =
        rz.matrix<float>().slice(
            Eigen::array<int, 2>({0, h_size}), Eigen::array<int, 2>({batch_size, h_size}));
  }
};

template <>
struct GruRZ<GPUDevice> {
  void operator()(
      const GPUDevice& d, const Tensor& rz, Tensor* r, Tensor* h) {
    GruRZGPU(d, rz, r, h);
  }
};


template <>
struct GruAdd<CPUDevice> {
  void operator()(const CPUDevice& d, const Tensor& a, const Tensor& b, Tensor* c) {
    c->matrix<float>().device(d) = a.matrix<float>() + b.matrix<float>();
  }
};

template <>
struct GruAdd<GPUDevice> {
  void operator()(const GPUDevice& d, const Tensor& a, const Tensor& b, Tensor* c) {
    GruAddGPU(d, a, b, c);
  }
};

template <>
struct GruPadTime<CPUDevice> {
  void operator()(
      const CPUDevice& d, const Tensor& sequence_len, const int64 sequence_idx,
      float value, Tensor* x) {
    x->matrix<float>().device(d) = x->matrix<float>() *
        (sequence_len.vec<int64>().cast<float>().constant(value) < sequence_len.vec<int64>().cast<float>())
        .broadcast(Eigen::array<int, 2>({1, static_cast<int>(x->dim_size(1))}));
  }
};

template <>
struct GruPadTime<GPUDevice> {
  void operator()(
      const GPUDevice& d, const Tensor& sequence_len, const int64 sequence_idx,
      float value, Tensor* x) {
    GruPadTimeGPU(d, sequence_len, sequence_idx, value, x);
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
struct GruActivationSigmoidGradient<CPUDevice> {
  void operator()(const CPUDevice& d, const Tensor& x, Tensor* dx) {
    dx->matrix<float>().device(d) = dx->matrix<float>() *
        x.matrix<float>() * (x.matrix<float>().constant(1.0f) - x.matrix<float>());
  }
};

template <>
struct GruActivationSigmoidGradient<GPUDevice> {
  void operator()(const GPUDevice& d, const Tensor& x, Tensor* dx) {
    GruActivationSigmoidGradientGPU(d, x, dx);
  }
};

template <>
struct GruActivationTanhGradient<CPUDevice> {
  void operator()(const CPUDevice& d, const Tensor& x, Tensor* dx) {
    dx->matrix<float>().device(d) = dx->matrix<float>() *
        (x.matrix<float>().constant(1.0f) - x.matrix<float>() * x.matrix<float>());
  }
};

template <>
struct GruActivationTanhGradient<GPUDevice> {
  void operator()(const GPUDevice& d, const Tensor& x, Tensor* dx) {
    GruActivationTanhGradientGPU(d, x, dx);
  }
};

template <>
struct GruCWiseMult<CPUDevice> {
  void operator()(const CPUDevice& d, const Tensor& a, const Tensor& b, float beta, Tensor* c) {
    if (beta == 0.0f) {
      c->matrix<float>().device(d) = a.matrix<float>() * b.matrix<float>();
    } else if (beta == 1.0f) {
      c->matrix<float>().device(d) = c->matrix<float>() +
          a.matrix<float>() * b.matrix<float>();
    }
  }
};

template <>
struct GruCWiseMult<GPUDevice> {
  void operator()(const GPUDevice& d, const Tensor& a, const Tensor& b, float beta, Tensor* c) {
    GruCWiseMultGPU(d, a, b, beta, c);
  }
};

template <>
struct GruH<CPUDevice> {
  void operator()(
      const CPUDevice& d, const Tensor& z, const Tensor* h_prev,
      const Tensor& g, Tensor* h) {
    if (h_prev) {
      h->matrix<float>().device(d) =
          z.matrix<float>() * h_prev->matrix<float>() +
          (z.matrix<float>().constant(1.0f) - z.matrix<float>()) * g.matrix<float>();
    } else {
      h->matrix<float>().device(d) =
          (z.matrix<float>().constant(1.0f) - z.matrix<float>()) * g.matrix<float>();
    }
  }
};

template <>
struct GruH<GPUDevice> {
  void operator()(
      const GPUDevice& d, const Tensor& z, const Tensor* h_prev,
      const Tensor& g, Tensor* h) {
    GruHGPU(d, z, h_prev, g, h);
  }
};

template <>
struct GruDz<CPUDevice> {
  void operator()(
      const CPUDevice& d, const Tensor& dh, const Tensor* h_prev, const Tensor& g,
      Tensor* dz) {
    if (h_prev) {
      dz->matrix<float>().device(d) =
          dh.matrix<float>() * h_prev->matrix<float>() -
          dh.matrix<float>() * g.matrix<float>();
    } else {
      dz->matrix<float>().device(d) = dh.matrix<float>() * g.matrix<float>();
    }
  }
};

template <>
struct GruDz<GPUDevice> {
  void operator()(
      const GPUDevice& d, const Tensor& dh, const Tensor* h_prev, const Tensor& g,
      Tensor* dz) {
    GruDzGPU(d, dh, h_prev, g, dz);
  }
};

template <>
struct GruDg<CPUDevice> {
  void operator()(
      const CPUDevice& d, const Tensor& dh, const Tensor& z, Tensor* dg) {
    dg->matrix<float>().device(d) =
        dh.matrix<float>() * (z.matrix<float>().constant(1.0f) - z.matrix<float>());
  }
};

template <>
struct GruDg<GPUDevice> {
  void operator()(
      const GPUDevice& d, const Tensor& dh, const Tensor& z, Tensor* dg) {
    GruDgGPU(d, dh, z, dg);
  }
};

template <typename Device>
class TokenSampleOp : public OpKernel {
 public:
  explicit TokenSampleOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sample_prob", &sample_prob_));
  }

  void Compute(OpKernelContext* ctx) override {
    #define INPUT_TENSOR(T)                                                      \
      const Tensor* T = nullptr;                                                 \
      OP_REQUIRES_OK(ctx, ctx->input(#T, &T));

    INPUT_TENSOR(ground_truth);
    INPUT_TENSOR(token_distribution);

    Tensor* token = nullptr;
    ctx->allocate_output("token", ground_truth->shape(), &token);
    token->vec<int32>().device(ctx->eigen_device<Device>()) =
        ground_truth->vec<int32>();

    const int64 batch_size = token->dim_size(0);
    const int64 vocab_size = token_distribution->dim_size(1);

    typedef random::UniformDistribution<random::PhiloxRandom, float>
        Distribution;
    Distribution dist;

    // First determine whether we want to sample or not sample from our distribution.
    std::vector<float> sample_prab(batch_size);
    std::vector<float> sample_token_prob(batch_size);

    // Sample our random numbers.
    const int kGroupSize = Distribution::kResultElementCount;
    auto local_generator = generator_.ReserveSamples32(batch_size * 2);
    for (int64 i = 0; i < batch_size; i += kGroupSize) {
      auto samples = dist(&local_generator);
      std::copy(&samples[0], &samples[0] + kGroupSize, &sample_prab[i]);
      samples = dist(&local_generator);
      std::copy(&samples[0], &samples[0] + kGroupSize, &sample_token_prob[i]);
    }

    if (sample_prob_ > 0.0f) {
      for (int64 b = 0; b < batch_size; ++b) {
        if (sample_prab[b] < sample_prob_) {
          float prob = 0.0f;
          int64 v = 0;
          for (; v < vocab_size; ++v) {
            prob += token_distribution->matrix<float>()(b, v);
            if (prob >= sample_token_prob[b]) break;
          }

          token->vec<int32>()(b) = v;
        }
      }
    }
  }

 private:
  float sample_prob_;
  GuardedPhiloxRandom generator_;
};

template <typename Device>
class GruCellOp : public OpKernel {
 public:
  explicit GruCellOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("time_idx", &time_idx_));
  }

  void Compute(OpKernelContext* ctx) override {
    INPUT_TENSOR(sequence_len);
    const int64 batch_size = sequence_len->NumElements();

    INPUT_TENSOR(h_prev);
    INPUT_TENSOR(x);

    INPUT_TENSOR(wxr);
    INPUT_TENSOR(whr);
    INPUT_TENSOR(wxz);
    INPUT_TENSOR(whz);
    INPUT_TENSOR(wxh);
    INPUT_TENSOR(whh);

    INPUT_TENSOR(br);
    INPUT_TENSOR(bz);
    INPUT_TENSOR(bh);

    #define OUTPUT_TENSOR(NAME, SHAPE)                                         \
      Tensor* NAME = nullptr;                                                  \
      ctx->allocate_output(#NAME, SHAPE, &NAME);                               \
      GruSetZero<Device>()(ctx->eigen_device<Device>(), NAME);

    OUTPUT_TENSOR(r, TensorShape({batch_size, cell_size_}));
    OUTPUT_TENSOR(z, TensorShape({batch_size, cell_size_}));
    OUTPUT_TENSOR(rh, TensorShape({batch_size, cell_size_}));
    OUTPUT_TENSOR(g, TensorShape({batch_size, cell_size_}));
    OUTPUT_TENSOR(h, TensorShape({batch_size, cell_size_}));

    // Dynamic computation of this cell based on the sequence_len (if porvided the time_idx_).
    if (time_idx_ >= 0) {
      INPUT_TENSOR(sequence_len);

      auto sequence_len_t = sequence_len->vec<int64>();
      std::vector<int64> seq_lens_vector(sequence_len_t.size());
      ctx->eigen_device<Device>().memcpyDeviceToHost(
          seq_lens_vector.data(), sequence_len_t.data(),
          sizeof(int64) * sequence_len_t.size());
      GruDeviceSynchronize<Device>()(ctx->eigen_device<Device>());

      int64 sequence_len_max =
          *std::max_element(seq_lens_vector.begin(), seq_lens_vector.end());

      if (time_idx_ >= sequence_len_max) return;
    }

    // r[t] = sigm(x[t] Wxr + h[t - 1] Whr)
    GruMatMul<Device>(ctx, false, *x, false, *wxr, 0.0f, r);
    GruMatMul<Device>(ctx, false, *h_prev, false, *whr, 1.0f, r);
    GruBias<Device>()(ctx->eigen_device<Device>(), *br, r);
    GruActivationSigmoid<Device>()(ctx->eigen_device<Device>(), r);

    // z[t] = sigm(x[t] Wxz + h[t - 1] Whz)
    GruMatMul<Device>(ctx, false, *x, false, *wxz, 0.0f, z);
    GruMatMul<Device>(ctx, false, *h_prev, false, *whz, 1.0f, z);
    GruBias<Device>()(ctx->eigen_device<Device>(), *bz, z);
    GruActivationSigmoid<Device>()(ctx->eigen_device<Device>(), z);

    // rh[t] = r[t] .* h_prev[t]
    // g[t] = tanh(x[t] Wxh + rh[t] Whh)
    GruCWiseMult<Device>()(ctx->eigen_device<Device>(), *r, *h_prev, 0.0f, rh);
    GruMatMul<Device>(ctx, false, *x, false, *wxh, 0.0f, g);
    GruMatMul<Device>(ctx, false, *rh, false, *whh, 1.0f, g);
    GruBias<Device>()(ctx->eigen_device<Device>(), *bh, g);
    GruActivationTanh<Device>()(ctx->eigen_device<Device>(), g);

    // h[t] = z[t] .* h[t - 1] + (1 - z[t]) .* g[t]
    GruH<Device>()(ctx->eigen_device<Device>(), *z, h_prev, *g, h);

    //GruPadTime<Device>()(ctx->eigen_device<Device>(), *sequence_len, t, 0.0f, h);
  }

 private:
  int64 cell_size_;
  int64 time_idx_;
};

template <typename Device>
class GruCellGradOp : public OpKernel {
 public:
  explicit GruCellGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("time_idx", &time_idx_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* sequence_len = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sequence_len", &sequence_len));

    const int64 batch_size = sequence_len->NumElements();

    INPUT_TENSOR(wxr);
    INPUT_TENSOR(whr);
    INPUT_TENSOR(wxz);
    INPUT_TENSOR(whz);
    INPUT_TENSOR(wxh);
    INPUT_TENSOR(whh);

    INPUT_TENSOR(br);
    INPUT_TENSOR(bz);
    INPUT_TENSOR(bh);

    INPUT_TENSOR(h_prev);
    INPUT_TENSOR(x);
    INPUT_TENSOR(r);
    INPUT_TENSOR(z);
    INPUT_TENSOR(rh);
    INPUT_TENSOR(hh);
    INPUT_TENSOR(h);

#define MUTABLE_INPUT(NAME)                                                    \
    Tensor NAME;                                                               \
    OP_REQUIRES_OK(ctx, ctx->mutable_input(#NAME, &NAME, true));

#define ALLOCATE_TEMP(NAME, SHAPE)                                             \
    Tensor NAME;                                                               \
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, SHAPE, &NAME));           \
    GruSetZero<Device>()(ctx->eigen_device<Device>(), &NAME);

    ALLOCATE_TEMP(dr, h->shape());
    ALLOCATE_TEMP(dz, h->shape());
    ALLOCATE_TEMP(drh, h->shape());
    ALLOCATE_TEMP(dg, h->shape());

    INPUT_TENSOR(dh);

    OUTPUT_TENSOR(dwxr, wxr->shape());
    OUTPUT_TENSOR(dwhr, whr->shape());
    OUTPUT_TENSOR(dwxz, wxz->shape());
    OUTPUT_TENSOR(dwhz, whz->shape());
    OUTPUT_TENSOR(dwxh, wxh->shape());
    OUTPUT_TENSOR(dwhh, whh->shape());

    OUTPUT_TENSOR(dbr, br->shape());
    OUTPUT_TENSOR(dbz, bz->shape());
    OUTPUT_TENSOR(dbh, bh->shape());

    const int64 input_size = wxr->dim_size(0);
    OUTPUT_TENSOR(dx, TensorShape({batch_size, input_size}));
    OUTPUT_TENSOR(dh_prev, TensorShape({batch_size, cell_size_}));

    // Dynamic computation of this cell based on the sequence_len (if porvided the time_idx_).
    if (time_idx_ >= 0) {
      INPUT_TENSOR(sequence_len);

      auto sequence_len_t = sequence_len->vec<int64>();
      std::vector<int64> seq_lens_vector(sequence_len_t.size());
      ctx->eigen_device<Device>().memcpyDeviceToHost(
          seq_lens_vector.data(), sequence_len_t.data(),
          sizeof(int64) * sequence_len_t.size());
      GruDeviceSynchronize<Device>()(ctx->eigen_device<Device>());

      int64 sequence_len_max =
          *std::max_element(seq_lens_vector.begin(), seq_lens_vector.end());

      if (time_idx_ >= sequence_len_max) return;
    }

    // h[t] = z[t] .* h[t - 1] + (1 - z[t]) .* g[t]
    GruDz<Device>()(ctx->eigen_device<Device>(), *dh, h_prev, *hh, &dz);
    GruCWiseMult<Device>()(ctx->eigen_device<Device>(), *dh, *z, 0.0f, dh_prev);
    GruDg<Device>()(ctx->eigen_device<Device>(), *dh, *z, &dg);

    // g[t] = tanh(x[t] Wxh + rh[t] Whh)
    GruActivationTanhGradient<Device>()(ctx->eigen_device<Device>(), *hh, &dg);
    GruMatMul<Device>(ctx, false, dg, true, *wxh, 0.0f, dx);
    GruMatMul<Device>(ctx, false, dg, true, *whh, 0.0f, &drh);

    GruCWiseMult<Device>()(ctx->eigen_device<Device>(), drh, *h_prev, 0.0f, &dr);
    GruCWiseMult<Device>()(ctx->eigen_device<Device>(), drh, *r, 1.0f, dh_prev);

    // z[t] = sigm(x[t] Wxz + h[t - 1] Whz)
    GruActivationSigmoidGradient<Device>()(ctx->eigen_device<Device>(), *z, &dz);
    GruMatMul<Device>(ctx, false, dz, true, *wxz, 1.0f, dx);
    GruMatMul<Device>(ctx, false, dz, true, *whz, 1.0f, dh_prev);

    // r[t] = sigm(x[t] Wxr + h[t - 1] Whr)
    GruActivationSigmoidGradient<Device>()(ctx->eigen_device<Device>(), *r, &dr);
    GruMatMul<Device>(ctx, false, dr, true, *wxr, 1.0f, dx);
    GruMatMul<Device>(ctx, false, dr, true, *whr, 1.0f, dh_prev);

    // Gradients wrt to weights.
    GruMatMul<Device>(ctx, true, *x, false, dr, 0.0f, dwxr);
    GruMatMul<Device>(ctx, true, *x, false, dz, 0.0f, dwxz);
    GruMatMul<Device>(ctx, true, *x, false, dg, 0.0f, dwxh);

    GruMatMul<Device>(ctx, true, *h_prev, false, dr, 0.0f, dwhr);
    GruMatMul<Device>(ctx, true, *h_prev, false, dz, 0.0f, dwhz);
    GruMatMul<Device>(ctx, true, *rh, false, dg, 0.0f, dwhh);

    GruBiasGrad<Device>()(ctx->eigen_device<Device>(), dr, dbr);
    GruBiasGrad<Device>()(ctx->eigen_device<Device>(), dz, dbz);
    GruBiasGrad<Device>()(ctx->eigen_device<Device>(), dg, dbh);
  }

 private:
  int64 cell_size_;
  int64 time_idx_;
};

template <typename Device>
class GruOp : public OpKernel {
 public:
  explicit GruOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sequence_len_max", &sequence_len_max_));
  }

  void Compute(OpKernelContext* ctx) override {
    INPUT_TENSOR(sequence_len);

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
    const int64 batch_size = seq_lens_vector.size();

  #define INPUT_LIST(T)                                                        \
    OpInputList T;                                                             \
    OP_REQUIRES_OK(ctx, ctx->input_list(#T, &T));

    INPUT_LIST(xs);

    INPUT_TENSOR(wxr);
    INPUT_TENSOR(whr);
    INPUT_TENSOR(wxz);
    INPUT_TENSOR(whz);
    INPUT_TENSOR(wxh);
    INPUT_TENSOR(whh);

    INPUT_TENSOR(br);
    INPUT_TENSOR(bz);
    INPUT_TENSOR(bh);

    // Fused matrix mult is faster.
    const int64 input_size = wxr->dim_size(0);
    ALLOCATE_TEMP(
        wxhrz, TensorShape({cell_size_ + input_size, cell_size_ * 2}));
    ALLOCATE_TEMP(
        xh, TensorShape({batch_size, input_size + cell_size_}));
    ALLOCATE_TEMP(
        rz, TensorShape({batch_size, cell_size_ * 2}));
    GruWxhrz<Device>()(
        ctx->eigen_device<Device>(), *wxr, *whr, *wxz, *whz, &wxhrz);

  #define OUTPUT_LIST(T)                                                       \
    OpOutputList T;                                                            \
    OP_REQUIRES_OK(ctx, ctx->output_list(#T, &T));

    OUTPUT_LIST(rs);
    OUTPUT_LIST(zs);
    OUTPUT_LIST(rhs);
    OUTPUT_LIST(gs);
    OUTPUT_LIST(hs);

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
    for (int64 t = 0; t < sequence_len_max; ++t) {
      const Tensor x = xs[t];
      const Tensor* h_prev = t <= 0 ? nullptr : hs[t - 1];

      Tensor* r = rs[t];
      Tensor* z = zs[t];
      Tensor* rh = rhs[t];
      Tensor* g = gs[t];
      Tensor* h = hs[t];

      GruXH<Device>()(ctx->eigen_device<Device>(), x, h_prev, &xh);
      GruMatMul<Device>(ctx, false, xh, false, wxhrz, 0.0f, &rz);
      GruRZ<Device>()(ctx->eigen_device<Device>(), rz, r, z);

      // r[t] = sigm(x[t] Wxr + h[t - 1] Whr)
      // GruMatMul<Device>(ctx, false, x, false, *wxr, 0.0f, r);
      // if (t > 0) GruMatMul<Device>(ctx, false, *h_prev, false, *whr, 1.0f, r);
      GruBias<Device>()(ctx->eigen_device<Device>(), *br, r);
      GruActivationSigmoid<Device>()(ctx->eigen_device<Device>(), r);

      // z[t] = sigm(x[t] Wxz + h[t - 1] Whz)
      // GruMatMul<Device>(ctx, false, x, false, *wxz, 0.0f, z);
      // if (t > 0) GruMatMul<Device>(ctx, false, *h_prev, false, *whz, 1.0f, z);
      GruBias<Device>()(ctx->eigen_device<Device>(), *bz, z);
      GruActivationSigmoid<Device>()(ctx->eigen_device<Device>(), z);

      // rh[t] = r[t] .* h_prev[t]
      // g[t] = tanh(x[t] Wxh + rh[t] Whh)
      if (t > 0) {
        GruCWiseMult<Device>()(ctx->eigen_device<Device>(), *r, *h_prev, 0.0f, rh);
      } else {
        GruSetZero<Device>()(ctx->eigen_device<Device>(), rh);
      }
      GruMatMul<Device>(ctx, false, x, false, *wxh, 0.0f, g);
      if (t > 0) GruMatMul<Device>(ctx, false, *rh, false, *whh, 1.0f, g);
      GruBias<Device>()(ctx->eigen_device<Device>(), *bh, g);
      GruActivationTanh<Device>()(ctx->eigen_device<Device>(), g);

      // h[t] = z[t] .* h[t - 1] + (1 - z[t]) .* g[t]
      GruH<Device>()(ctx->eigen_device<Device>(), *z, h_prev, *g, h);

      GruPadTime<Device>()(ctx->eigen_device<Device>(), *sequence_len, t, 0.0f, h);
    }
  }

 protected:
  int64 cell_size_;
  int64 sequence_len_max_;
};

template <typename Device>
class GruFusedOp : public OpKernel {
 public:
  explicit GruFusedOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("cell_size", &cell_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sequence_len_max", &sequence_len_max_));
  }

  void Compute(OpKernelContext* ctx) override {
    INPUT_TENSOR(sequence_len);

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
    const int64 batch_size = seq_lens_vector.size();

  #define INPUT_LIST(T)                                                        \
    OpInputList T;                                                             \
    OP_REQUIRES_OK(ctx, ctx->input_list(#T, &T));

    INPUT_LIST(xs);

    INPUT_TENSOR(wxr);
    INPUT_TENSOR(whr);
    INPUT_TENSOR(wxz);
    INPUT_TENSOR(whz);
    INPUT_TENSOR(wxh);
    INPUT_TENSOR(whh);

    // Fused matrix mult is faster.
    const int64 input_size = wxr->dim_size(0);
    ALLOCATE_TEMP(
        wxhrz, TensorShape({cell_size_ + input_size, cell_size_ * 2}));
    ALLOCATE_TEMP(
        xh, TensorShape({batch_size, input_size + cell_size_}));
    ALLOCATE_TEMP(
        r, TensorShape({batch_size, cell_size_}));
    ALLOCATE_TEMP(
        z, TensorShape({batch_size, cell_size_}));
    GruWxhrz<Device>()(
        ctx->eigen_device<Device>(), *wxr, *whr, *wxz, *whz, &wxhrz);

  #define OUTPUT_LIST(T)                                                       \
    OpOutputList T;                                                            \
    OP_REQUIRES_OK(ctx, ctx->output_list(#T, &T));

    OUTPUT_LIST(rzs);
    OUTPUT_LIST(rhs);
    OUTPUT_LIST(gs);
    OUTPUT_LIST(hs);

    for (int64 t = 0; t < sequence_len_max_; ++t) {
  #define OUTPUT_LIST_ALLOCATE(OP_OUTPUT_LIST, NAME, SHAPE)                    \
      Tensor* NAME = nullptr;                                                  \
      OP_OUTPUT_LIST.allocate(t, SHAPE, &NAME);                                \
      GruSetZero<Device>()(ctx->eigen_device<Device>(), NAME);

      OUTPUT_LIST_ALLOCATE(rzs, r, TensorShape({batch_size, cell_size_ * 2}));
      OUTPUT_LIST_ALLOCATE(rhs, rh, TensorShape({batch_size, cell_size_}));
      OUTPUT_LIST_ALLOCATE(gs, g, TensorShape({batch_size, cell_size_}));
      OUTPUT_LIST_ALLOCATE(hs, h, TensorShape({batch_size, cell_size_}));
    }

    if (sequence_len_max >= sequence_len_max_) {
      sequence_len_max = sequence_len_max_;
    }
    for (int64 t = 0; t < sequence_len_max; ++t) {
      const Tensor x = xs[t];
      const Tensor* h_prev = t <= 0 ? nullptr : hs[t - 1];

      Tensor* rz = rzs[t];
      Tensor* rh = rhs[t];
      Tensor* g = gs[t];
      Tensor* h = hs[t];

      // r[t] = sigm(x[t] Wxr + h[t - 1] Whr)
      // z[t] = sigm(x[t] Wxz + h[t - 1] Whz)
      GruXH<Device>()(ctx->eigen_device<Device>(), x, h_prev, &xh);
      GruMatMul<Device>(ctx, false, xh, false, wxhrz, 0.0f, rz);
      GruActivationSigmoid<Device>()(ctx->eigen_device<Device>(), rz);
      GruRZ<Device>()(ctx->eigen_device<Device>(), *rz, &r, &z);

      // rh[t] = r[t] .* h_prev[t]
      // g[t] = tanh(x[t] Wxh + rh[t] Whh)
      if (t > 0) {
        GruCWiseMult<Device>()(ctx->eigen_device<Device>(), r, *h_prev, 0.0f, rh);
      } else {
        GruSetZero<Device>()(ctx->eigen_device<Device>(), rh);
      }
      GruMatMul<Device>(ctx, false, x, false, *wxh, 0.0f, g);
      if (t > 0) GruMatMul<Device>(ctx, false, *rh, false, *whh, 1.0f, g);
      GruActivationTanh<Device>()(ctx->eigen_device<Device>(), g);

      // h[t] = z[t] .* h[t - 1] + (1 - z[t]) .* g[t]
      GruH<Device>()(ctx->eigen_device<Device>(), z, h_prev, *g, h);

      GruPadTime<Device>()(ctx->eigen_device<Device>(), *sequence_len, t, 0.0f, h);
    }
  }

 protected:
  int64 cell_size_;
  int64 sequence_len_max_;
};

REGISTER_KERNEL_BUILDER(Name("GruCell")
                             .Device(DEVICE_CPU),
                        GruCellOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("GruCell")
                             .Device(DEVICE_GPU),
                        GruCellOp<GPUDevice>);
#endif  // GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("Gru")
                             .Device(DEVICE_CPU),
                        GruOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Gru")
                             .Device(DEVICE_GPU),
                        GruOp<GPUDevice>);
#endif  // GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("GruFused")
                             .Device(DEVICE_CPU),
                        GruFusedOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("GruFused")
                             .Device(DEVICE_GPU),
                        GruFusedOp<GPUDevice>);
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

    // Get the batch size.
    const int64 batch_size = sequence_len->NumElements();

    // Maximum number of compute unrolls.
    int64 sequence_len_max =
        *std::max_element(seq_lens_vector.begin(), seq_lens_vector.end());

    INPUT_TENSOR(wxr);
    INPUT_TENSOR(whr);
    INPUT_TENSOR(wxz);
    INPUT_TENSOR(whz);
    INPUT_TENSOR(wxh);
    INPUT_TENSOR(whh);

    INPUT_TENSOR(br);
    INPUT_TENSOR(bz);
    INPUT_TENSOR(bh);

    INPUT_LIST(xs);
    INPUT_LIST(rs);
    INPUT_LIST(zs);
    INPUT_LIST(rhs);
    INPUT_LIST(gs);
    INPUT_LIST(hs);

    // Gradients.
    INPUT_LIST(dhs);

    ALLOCATE_TEMP(dh, TensorShape({batch_size, cell_size_}));
    ALLOCATE_TEMP(dh_prev, dh.shape());
    ALLOCATE_TEMP(dr, dh.shape());
    ALLOCATE_TEMP(dz, dh.shape());
    ALLOCATE_TEMP(drh, dh.shape());
    ALLOCATE_TEMP(dg, dh.shape());

    OUTPUT_TENSOR(dwxr, wxr->shape());
    OUTPUT_TENSOR(dwhr, whr->shape());
    OUTPUT_TENSOR(dwxz, wxz->shape());
    OUTPUT_TENSOR(dwhz, whz->shape());
    OUTPUT_TENSOR(dwxh, wxh->shape());
    OUTPUT_TENSOR(dwhh, whh->shape());

    OUTPUT_TENSOR(dbr, br->shape());
    OUTPUT_TENSOR(dbz, bz->shape());
    OUTPUT_TENSOR(dbh, bh->shape());

    OUTPUT_LIST(dxs);

    CHECK_EQ(xs.size(), sequence_len_max_);
    for (int64 t = 0; t < sequence_len_max_; ++t) {
      const Tensor x = xs[t];
      OUTPUT_LIST_ALLOCATE(dxs, dx, x.shape());
    }

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
      // const Tensor dhtt = dhs.at(t, true);
      // const Tensor dh = dhs.at(t, true);
      GruAdd<Device>()(ctx->eigen_device<Device>(), dhs[t], dh_prev, &dh);
      // Tensor dh_prev = t <= 0 ? dhs.at(0, true) : dhs.at(t - 1, true);

      const Tensor* h_prev = t <= 0 ? nullptr : &hs[t - 1];

      GruSetZero<Device>()(ctx->eigen_device<Device>(), &dh_prev);
      GruSetZero<Device>()(ctx->eigen_device<Device>(), &dr);
      GruSetZero<Device>()(ctx->eigen_device<Device>(), &dz);
      GruSetZero<Device>()(ctx->eigen_device<Device>(), &drh);
      GruSetZero<Device>()(ctx->eigen_device<Device>(), &dg);

      Tensor* dx = dxs[t];

      // h[t] = z[t] .* h[t - 1] + (1 - z[t]) .* g[t]
      GruDz<Device>()(ctx->eigen_device<Device>(), dh, h_prev, g, &dz);
      if (t > 0) {
        GruCWiseMult<Device>()(ctx->eigen_device<Device>(), dh, z, 1.0f, &dh_prev);
      }
      GruDg<Device>()(ctx->eigen_device<Device>(), dh, z, &dg);

      // g[t] = tanh(x[t] Wxh + rh[t] Whh)
      GruActivationTanhGradient<Device>()(ctx->eigen_device<Device>(), g, &dg);
      GruMatMul<Device>(ctx, false, dg, true, *wxh, 0.0f, dx);
      GruMatMul<Device>(ctx, false, dg, true, *whh, 0.0f, &drh);

      if (t > 0) {
        GruCWiseMult<Device>()(ctx->eigen_device<Device>(), drh, *h_prev, 0.0f, &dr);
        GruCWiseMult<Device>()(ctx->eigen_device<Device>(), drh, r, 1.0f, &dh_prev);
      } else {
        GruSetZero<Device>()(ctx->eigen_device<Device>(), &dr);
      }

      // z[t] = sigm(x[t] Wxz + h[t - 1] Whz)
      GruActivationSigmoidGradient<Device>()(ctx->eigen_device<Device>(), z, &dz);
      GruMatMul<Device>(ctx, false, dz, true, *wxz, 1.0f, dx);
      if (t > 0) GruMatMul<Device>(ctx, false, dz, true, *whz, 1.0f, &dh_prev);

      // r[t] = sigm(x[t] Wxr + h[t - 1] Whr)
      if (t > 0) {
        GruActivationSigmoidGradient<Device>()(ctx->eigen_device<Device>(), r, &dr);
        GruMatMul<Device>(ctx, false, dr, true, *wxr, 1.0f, dx);
        GruMatMul<Device>(ctx, false, dr, true, *whr, 1.0f, &dh_prev);
      }

      // Gradient wrt to the weights.
      GruMatMul<Device>(ctx, true, x, false, dr, 1.0f, dwxr);
      GruMatMul<Device>(ctx, true, x, false, dz, 1.0f, dwxz);
      GruMatMul<Device>(ctx, true, x, false, dg, 1.0f, dwxh);

      if (t > 0) {
        GruMatMul<Device>(ctx, true, *h_prev, false, dr, 1.0f, dwhr);
        GruMatMul<Device>(ctx, true, *h_prev, false, dz, 1.0f, dwhz);
        GruMatMul<Device>(ctx, true, rh, false, dg, 1.0f, dwhh);
      }

      GruBiasGrad<Device>()(ctx->eigen_device<Device>(), dr, dbr);
      GruBiasGrad<Device>()(ctx->eigen_device<Device>(), dz, dbz);
      GruBiasGrad<Device>()(ctx->eigen_device<Device>(), dg, dbh);
    }
  }

 protected:
  int64 cell_size_;
  int64 sequence_len_max_;
};

template <typename Device>
class GruFusedGradOp : public OpKernel {
 public:
  explicit GruFusedGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
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

    // Get the batch size.
    const int64 batch_size = sequence_len->NumElements();

    // Maximum number of compute unrolls.
    int64 sequence_len_max =
        *std::max_element(seq_lens_vector.begin(), seq_lens_vector.end());

    INPUT_TENSOR(wxr);
    INPUT_TENSOR(whr);
    INPUT_TENSOR(wxz);
    INPUT_TENSOR(whz);
    INPUT_TENSOR(wxh);
    INPUT_TENSOR(whh);

    INPUT_LIST(xs);
    INPUT_LIST(rzs);
    INPUT_LIST(rhs);
    INPUT_LIST(gs);
    INPUT_LIST(hs);

    // Gradients.
    INPUT_LIST(dhs);

    ALLOCATE_TEMP(dh, TensorShape({batch_size, cell_size_}));
    ALLOCATE_TEMP(r, dh.shape());
    ALLOCATE_TEMP(z, dh.shape());
    ALLOCATE_TEMP(dh_prev, dh.shape());
    ALLOCATE_TEMP(dr, dh.shape());
    ALLOCATE_TEMP(dz, dh.shape());
    ALLOCATE_TEMP(drh, dh.shape());
    ALLOCATE_TEMP(dg, dh.shape());

    OUTPUT_TENSOR(dwxr, wxr->shape());
    OUTPUT_TENSOR(dwhr, whr->shape());
    OUTPUT_TENSOR(dwxz, wxz->shape());
    OUTPUT_TENSOR(dwhz, whz->shape());
    OUTPUT_TENSOR(dwxh, wxh->shape());
    OUTPUT_TENSOR(dwhh, whh->shape());

    OUTPUT_LIST(dxs);

    CHECK_EQ(xs.size(), sequence_len_max_);
    for (int64 t = 0; t < sequence_len_max_; ++t) {
      const Tensor x = xs[t];
      OUTPUT_LIST_ALLOCATE(dxs, dx, x.shape());
    }

    if (sequence_len_max >= sequence_len_max_) {
      sequence_len_max = sequence_len_max_;
    }
    for (int64 t = sequence_len_max - 1; t >= 0; --t) {
      const Tensor x = xs[t];
      const Tensor rz = rzs[t];
      const Tensor rh = rhs[t];
      const Tensor g = gs[t];
      const Tensor h = hs[t];
      // const Tensor dhtt = dhs.at(t, true);
      // const Tensor dh = dhs.at(t, true);
      GruAdd<Device>()(ctx->eigen_device<Device>(), dhs[t], dh_prev, &dh);
      // Tensor dh_prev = t <= 0 ? dhs.at(0, true) : dhs.at(t - 1, true);

      const Tensor* h_prev = t <= 0 ? nullptr : &hs[t - 1];

      GruSetZero<Device>()(ctx->eigen_device<Device>(), &dh_prev);
      GruSetZero<Device>()(ctx->eigen_device<Device>(), &dr);
      GruSetZero<Device>()(ctx->eigen_device<Device>(), &dz);
      GruSetZero<Device>()(ctx->eigen_device<Device>(), &drh);
      GruSetZero<Device>()(ctx->eigen_device<Device>(), &dg);

      Tensor* dx = dxs[t];

      GruRZ<Device>()(ctx->eigen_device<Device>(), rz, &r, &z);

      // h[t] = z[t] .* h[t - 1] + (1 - z[t]) .* g[t]
      GruDz<Device>()(ctx->eigen_device<Device>(), dh, h_prev, g, &dz);
      if (t > 0) {
        GruCWiseMult<Device>()(ctx->eigen_device<Device>(), dh, z, 1.0f, &dh_prev);
      }
      GruDg<Device>()(ctx->eigen_device<Device>(), dh, z, &dg);

      // g[t] = tanh(x[t] Wxh + rh[t] Whh)
      GruActivationTanhGradient<Device>()(ctx->eigen_device<Device>(), g, &dg);
      GruMatMul<Device>(ctx, false, dg, true, *wxh, 0.0f, dx);
      GruMatMul<Device>(ctx, false, dg, true, *whh, 0.0f, &drh);

      if (t > 0) {
        GruCWiseMult<Device>()(ctx->eigen_device<Device>(), drh, *h_prev, 0.0f, &dr);
        GruCWiseMult<Device>()(ctx->eigen_device<Device>(), drh, r, 1.0f, &dh_prev);
      } else {
        GruSetZero<Device>()(ctx->eigen_device<Device>(), &dr);
      }

      // z[t] = sigm(x[t] Wxz + h[t - 1] Whz)
      GruActivationSigmoidGradient<Device>()(ctx->eigen_device<Device>(), z, &dz);
      GruMatMul<Device>(ctx, false, dz, true, *wxz, 1.0f, dx);
      if (t > 0) GruMatMul<Device>(ctx, false, dz, true, *whz, 1.0f, &dh_prev);

      // r[t] = sigm(x[t] Wxr + h[t - 1] Whr)
      if (t > 0) {
        GruActivationSigmoidGradient<Device>()(ctx->eigen_device<Device>(), r, &dr);
        GruMatMul<Device>(ctx, false, dr, true, *wxr, 1.0f, dx);
        GruMatMul<Device>(ctx, false, dr, true, *whr, 1.0f, &dh_prev);
      }

      // Gradient wrt to the weights.
      GruMatMul<Device>(ctx, true, x, false, dr, 1.0f, dwxr);
      GruMatMul<Device>(ctx, true, x, false, dz, 1.0f, dwxz);
      GruMatMul<Device>(ctx, true, x, false, dg, 1.0f, dwxh);

      if (t > 0) {
        GruMatMul<Device>(ctx, true, *h_prev, false, dr, 1.0f, dwhr);
        GruMatMul<Device>(ctx, true, *h_prev, false, dz, 1.0f, dwhz);
        GruMatMul<Device>(ctx, true, rh, false, dg, 1.0f, dwhh);
      }
    }
  }

 protected:
  int64 cell_size_;
  int64 sequence_len_max_;
};

REGISTER_KERNEL_BUILDER(Name("GruCellGrad")
                             .Device(DEVICE_CPU),
                        GruCellGradOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("GruCellGrad")
                             .Device(DEVICE_GPU),
                        GruCellGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("GruGrad")
                             .Device(DEVICE_CPU),
                        GruGradOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("GruGrad")
                             .Device(DEVICE_GPU),
                        GruGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("GruFusedGrad")
                             .Device(DEVICE_CPU),
                        GruFusedGradOp<CPUDevice>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("GruFusedGrad")
                             .Device(DEVICE_GPU),
                        GruFusedGradOp<GPUDevice>);
#endif  // GOOGLE_CUDA

REGISTER_KERNEL_BUILDER(Name("TokenSample")
                             .Device(DEVICE_CPU),
                        TokenSampleOp<CPUDevice>);

}  // namespace tensorflow
