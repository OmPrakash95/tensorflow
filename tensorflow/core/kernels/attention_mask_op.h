// Copyright 2015 William Chan <williamchan@cmu.edu>.

#ifndef TENSORFLOW_KERNELS_ATTENTION_MASK_OP_H_
#define TENSORFLOW_KERNELS_ATTENTION_MASK_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace generator {
class AttentionMaskGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  AttentionMaskGenerator(
      float fill_value, TTypes<int64>::ConstVec sequence_len,
      TTypes<float, 2>::ConstTensor input)
    : fill_value_(fill_value), sequence_len_(sequence_len), input_(input) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  float operator()(const Eigen::array<Eigen::DenseIndex, 2>& coords) const {
    if (coords[1] < sequence_len_(coords[0])) {
      return input_(coords);
    } else {
      return fill_value_;
    }
  }

 private:
  float fill_value_;
  TTypes<int64>::ConstVec sequence_len_;
  TTypes<float, 2>::ConstTensor input_;
};

class AttentionMaskMedianGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  AttentionMaskMedianGenerator(
      float fill_value, int window_l, int window_r,
      TTypes<int64>::ConstVec sequence_len, TTypes<int>::ConstVec median,
      TTypes<float, 2>::ConstTensor input)
    : fill_value_(fill_value), window_l_(window_l), window_r_(window_r),
      sequence_len_(sequence_len), median_(median), input_(input) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  float operator()(const Eigen::array<Eigen::DenseIndex, 2>& coords) const {
    const int median = median_(coords[0]);
    const int idx_min = median - window_l_;
    const int idx_max = median + window_r_;
    const int idx = coords[1];
    if (idx >= idx_min && idx <= idx_max && idx < sequence_len_(coords[0])) {
      return input_(coords);
    } else {
      return fill_value_;
    }
  }

 private:
  float fill_value_;
  int window_l_;
  int window_r_;
  TTypes<int64>::ConstVec sequence_len_;
  TTypes<int>::ConstVec median_;
  TTypes<float, 2>::ConstTensor input_;
};
}  // end namespace generator

template <typename Device>
struct AttentionMask {
  void operator()(
      const Device& d, float fill_value, const Tensor& sequence_len,
      const Tensor& input, Tensor* output);
};

template <typename Device>
struct AttentionMaskMedian {
  void operator()(
      const Device& d, float fill_value, int64 window_l, int64 window_r,
      const Tensor& sequence_len, const Tensor& input, const Tensor& median,
      Tensor* output);
};

template <typename Device>
struct ComputeMedian {
  void operator()(
      const GPUDevice& d, const Tensor& input, Tensor* median);
};

template <typename Device>
struct SetZero {
  void operator()(const Device& d, Tensor* x);
};

void AttentionMaskGPU(
    const GPUDevice& d, float fill_value, const Tensor& sequence_len,
    const Tensor& input, Tensor* output);

void AttentionMaskMedianGPU(
    const GPUDevice& d, float fill_value, int64 window_l, int64 window_r,
    const Tensor& sequence_len, const Tensor& input, const Tensor& median,
    Tensor* output);

void ComputeMedianGPU(
    const GPUDevice& d, const Tensor& input, Tensor* median);

void SetZeroGPU(const GPUDevice& d, Tensor* x);
}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_ATTENTION_MASK_OP_H_
