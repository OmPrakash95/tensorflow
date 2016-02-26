#ifndef TENSORFLOW_KERNELS_TENSOR_ARRAY_OPS_H_
#define TENSORFLOW_KERNELS_TENSOR_ARRAY_OPS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

namespace generator {
template <typename T>
class TensorArraySubsampleGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  TensorArraySubsampleGenerator(
      int stride, typename TTypes<T, 3>::ConstTensor input)
    : stride_(stride), input_(input) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  float operator()(const Eigen::array<Eigen::DenseIndex, 3>& coords) const {
    Eigen::array<Eigen::DenseIndex, 3> input_coords = coords;
    input_coords[0] = input_coords[0] / stride_;

    return input_(input_coords);
  }

 private:
  int stride_;
  typename TTypes<T, 3>::ConstTensor input_;
};

template <typename T>
class TensorArraySubsampleGradGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  TensorArraySubsampleGradGenerator(
      int stride, typename TTypes<T, 3>::ConstTensor input)
    : stride_(stride), input_(input) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  float operator()(const Eigen::array<Eigen::DenseIndex, 3>& coords) const {
    if (coords[0] % stride_ == 0) {
      Eigen::array<Eigen::DenseIndex, 3> input_coords = coords;
      input_coords[0] = input_coords[0] * stride_;
      return input_(input_coords);
    } else {
      return 0.0f;
    }
  }

 private:
  int stride_;
  typename TTypes<T, 3>::ConstTensor input_;
};
}  // namespace generator

namespace functor {
template <typename Device, typename T>
struct TensorArraySubsample {
  void operator()(const Device& d, int stride,
                  typename TTypes<T, 3>::ConstTensor input,
                  typename TTypes<T, 3>::Tensor output) {
    generator::TensorArraySubsampleGenerator<T> generator(stride, input);
    output.device(d) = input.generate(generator);
  }
};

template <typename Device, typename T>
struct TensorArraySubsampleGrad {
  void operator()(const Device& d, int stride,
                  typename TTypes<T, 3>::ConstTensor input,
                  typename TTypes<T, 3>::Tensor output) {
    generator::TensorArraySubsampleGradGenerator<T> generator(stride, input);
    output.device(d) = input.generate(generator);
  }
};
}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_TENSOR_ARRAY_OPS_H_
