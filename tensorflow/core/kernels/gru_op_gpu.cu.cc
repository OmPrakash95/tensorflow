#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/gru_op.h"

namespace tensorflow {
// typedef Eigen::GpuDevice GPUDevice;

void GruDeviceSynchronizeGPU(const GPUDevice& d) {
  d.synchronize();
}

void GruSetZeroGPU(const GPUDevice& d, Tensor* x) {
  x->flat<float>().device(d) = x->flat<float>().constant(0.0f);
}

void GruCopyGPU(const GPUDevice& d, const Tensor& x, Tensor* y) {
  y->matrix<float>().device(d) = x.matrix<float>();
}

void GruBiasGPU(const GPUDevice& d, const Tensor& b, Tensor* y) {
  const int batch_size = y->dim_size(0);
  y->matrix<float>().device(d) =
      y->matrix<float>() + b.vec<float>().broadcast(Eigen::array<int, 2>({batch_size, 1}));
}

void GruBiasGradGPU(const GPUDevice& d, const Tensor& dx, Tensor* db) {
  db->vec<float>().device(d) =
      db->vec<float>() + dx.matrix<float>().sum(Eigen::array<int, 1>(0));
}

void GruWxhrzGPU(
    const GPUDevice& d, const Tensor& wxr, const Tensor& whr, const Tensor& wxz,
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

void GruXHGPU(
    const GPUDevice& d, const Tensor& x, const Tensor* h, Tensor* xh) {
  const int batch_size = x.dim_size(0);
  const int x_size = x.dim_size(1);

  CHECK_EQ(xh->dim_size(0), batch_size);
  xh->matrix<float>().slice(
      Eigen::array<int, 2>({0, 0}),      Eigen::array<int, 2>({batch_size, x_size})).device(d) =
      x.matrix<float>();
  if (h) {
    const int h_size = h->dim_size(1);
    CHECK_EQ(xh->dim_size(1), x_size + h_size);
    xh->matrix<float>().slice(
        Eigen::array<int, 2>({0, x_size}), Eigen::array<int, 2>({batch_size, h_size})).device(d) =
        h->matrix<float>();
  }
}

void GruRZGPU(
    const GPUDevice& d, const Tensor& rz, Tensor* r, Tensor* z) {
  const int batch_size = rz.dim_size(0);
  const int h_size = r->dim_size(1);

  CHECK_EQ(rz.dim_size(1), h_size * 2);
  CHECK_EQ(r->dim_size(0), batch_size);
  CHECK_EQ(z->dim_size(0), batch_size);

  r->matrix<float>().device(d) =
      rz.matrix<float>().slice(
          Eigen::array<int, 2>({0, 0}),      Eigen::array<int, 2>({batch_size, h_size}));
  z->matrix<float>().device(d) =
      rz.matrix<float>().slice(
          Eigen::array<int, 2>({0, h_size}), Eigen::array<int, 2>({batch_size, h_size}));
}

void GruAddGPU(const GPUDevice& d, const Tensor& a, const Tensor& b, Tensor* c) {
  c->matrix<float>().device(d) = a.matrix<float>() + b.matrix<float>();
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
