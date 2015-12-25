#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/gru_op.h"

namespace tensorflow {
// typedef Eigen::GpuDevice GPUDevice;

void GruSetZeroGPU(const GPUDevice& d, Tensor* x) {
  x->matrix<float>().device(d) = x->matrix<float>().constant(0.0f);
}

void GruAddGPU(const GPUDevice& d, const Tensor& a, const Tensor& b, Tensor* c) {
  c->matrix<float>().device(d) = a.matrix<float>() + b.matrix<float>();
}

void AttentionMaskGPU(
    const GPUDevice& d, float fill_value, const Tensor& sequence_len,
    const Tensor& input, Tensor* output) {
  generator::AttentionMaskGenerator generator(
      fill_value, sequence_len.vec<int64>(), input.matrix<float>());
  output->matrix<float>().device(d) = input.matrix<float>().generate(generator);
}

void GruPadTimeGPU(
    const GPUDevice& d, const Tensor& sequence_len, const int64 sequence_idx,
    float value, Tensor* x) {
  x->matrix<float>().device(d) = x->matrix<float>() *
      (sequence_len.vec<int64>().cast<float>().constant(value) < sequence_len.vec<int64>().cast<float>())
      .broadcast(Eigen::array<int, 2>({1, static_cast<int>(x->dim_size(1))}));
};

void GruActivationSigmoidGPU(const GPUDevice& d, Tensor* x) {
  x->matrix<float>().device(d) = x->matrix<float>().sigmoid();
}

void GruActivationTanhGPU(const GPUDevice& d, Tensor* x) {
  x->matrix<float>().device(d) = x->matrix<float>().tanh();
}

void GruActivationSigmoidGradientGPU(const GPUDevice& d, const Tensor& x, Tensor* dx) {
  dx->matrix<float>().device(d) = dx->matrix<float>() *
      x.matrix<float>() * (x.matrix<float>().constant(1.0f) - x.matrix<float>());
}

void GruActivationTanhGradientGPU(const GPUDevice& d, const Tensor& x, Tensor* dx) {
  dx->matrix<float>().device(d) = dx->matrix<float>() *
      (x.matrix<float>().constant(1.0f) - x.matrix<float>() * x.matrix<float>());
}

void GruCWiseMultGPU(const GPUDevice& d, const Tensor& a, const Tensor& b,
    float beta, Tensor* c) {
  if (beta == 0.0f) {
    c->matrix<float>().device(d) = a.matrix<float>() * b.matrix<float>();
  } else if (beta == 1.0f) {
    c->matrix<float>().device(d) = c->matrix<float>() +
        a.matrix<float>() * b.matrix<float>();
  }
}

void GruHGPU(
    const GPUDevice& d, const Tensor& z, const Tensor* h_prev,
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

void GruDzGPU(
    const GPUDevice& d, const Tensor& dh, const Tensor* h_prev, const Tensor& g,
    Tensor* dz) {
  if (h_prev) {
    dz->matrix<float>().device(d) =
        dh.matrix<float>() * h_prev->matrix<float>() -
        dh.matrix<float>() * g.matrix<float>();
  } else {
    dz->matrix<float>().device(d) = dh.matrix<float>() * g.matrix<float>();
  }
}

void GruDgGPU(
    const GPUDevice& d, const Tensor& dh, const Tensor& z, Tensor* dg) {
  dg->matrix<float>().device(d) =
      dh.matrix<float>() * (z.matrix<float>().constant(1.0f) - z.matrix<float>());
}

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
