#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/gru_op.h"

namespace tensorflow {
// typedef Eigen::GpuDevice GPUDevice;

void GruSetZeroGPU(const GPUDevice& d, Tensor* x) {
  x->matrix<float>().device(d) = x->matrix<float>().constant(0.0f);
}

void GruActivationSigmoidGPU(const GPUDevice& d, Tensor* x) {
  x->matrix<float>().device(d) = x->matrix<float>().sigmoid();
}

void GruActivationTanhGPU(const GPUDevice& d, Tensor* x) {
  x->matrix<float>().device(d) = x->matrix<float>().tanh();
}

void GruCWiseMultGPU(const GPUDevice& d, const Tensor& a, const Tensor& b,
    Tensor* c) {
  c->matrix<float>().device(d) = a.matrix<float>() * b.matrix<float>();
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

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
