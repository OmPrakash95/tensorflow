// Copyright 2015 William Chan <williamchan@cmu.edu>.

#ifndef TENSORFLOW_USER_KERNELS_S4_ATTENTION_MASK_ENERGIES_OP_H_
#define TENSORFLOW_USER_KERNELS_S4_ATTENTION_MASK_ENERGIES_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

namespace generator {

class S4MaskAttentionEnergiesDefaultGenerator {
 public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  S4MaskAttentionEnergiesDefaultGenerator(
      typename TTypes<float, 2>::ConstTensor energies,
      TTypes<int64>::ConstVec energies_len)
    : energies_(energies), energies_len_(energies_len) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE
  float operator()(const Eigen::array<Eigen::DenseIndex, 2>& coords) const {
    if (energies_len_(coords[0]) >= coords[1]) {
      return -FLT_MAX;
    } else {
      return energies_(coords);
    }
  }

 private:
  typename TTypes<float, 2>::ConstTensor energies_;
  TTypes<int64>::ConstVec energies_len_;
};

}  // namespace generator

namespace functor {

template <typename Device>
struct S4MaskAttentionEnergies {
  EIGEN_ALWAYS_INLINE static void ComputeDefault(
      const Device& d, typename TTypes<float, 2>::ConstTensor energies,
      TTypes<int64>::ConstVec energies_len,
      TTypes<float, 2>::Tensor masked_energies) {
    generator::S4MaskAttentionEnergiesDefaultGenerator generator(
        energies, energies_len);

    masked_energies.device(d) = energies.generate(generator);
  }

  EIGEN_ALWAYS_INLINE static void ComputeMedian(
      const Device& d, typename TTypes<float, 2>::ConstTensor energies,
      typename TTypes<float, 2>::ConstTensor energies_prev,
      int64 median_window_left, int64 median_window_right,
      TTypes<int64>::ConstVec energies_len,
      TTypes<float, 2>::Tensor masked_energies) {
  }
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_USER_KERNELS_S4_ATTENTION_MASK_ENERGIES_OP_H_
