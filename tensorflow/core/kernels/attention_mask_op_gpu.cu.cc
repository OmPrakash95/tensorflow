// Copyright 2015 William Chan <williamchan@cmu.edu>.

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/attention_mask_op.h"

namespace tensorflow {
void SetZeroGPU(const GPUDevice& d, Tensor* x) {
  x->matrix<float>().device(d) = x->matrix<float>().constant(0.0f);
}

void AttentionMaskGPU(
    const GPUDevice& d, float fill_value, const Tensor& sequence_len,
    const Tensor& input, Tensor* output) {
  generator::AttentionMaskGenerator generator(
      fill_value, sequence_len.vec<int64>(), input.matrix<float>());
  output->matrix<float>().device(d) = input.matrix<float>().generate(generator);
}

void AttentionMaskMedianGPU(
    const GPUDevice& d, float fill_value, int64 window_l, int64 window_r,
    const Tensor& sequence_len, const Tensor& input, const Tensor& median,
    Tensor* output) {
  generator::AttentionMaskMedianGenerator generator(
      fill_value, window_l, window_r, sequence_len.vec<int64>(),
      median.vec<int>(), input.matrix<float>());
  output->matrix<float>().device(d) = input.matrix<float>().generate(generator);
}

void AttentionMaskWindowGPU(
    const GPUDevice& d, float fill_value, int64 s_min, int64 s_max,
    float v_min, float v_max, int64 index, const Tensor& sequence_len,
    const Tensor& input, Tensor* output) {
  generator::AttentionMaskWindowGenerator generator(
      fill_value, s_min, s_max, v_min, v_max, index, sequence_len.vec<int64>(),
      input.matrix<float>());
  output->matrix<float>().device(d) = input.matrix<float>().generate(generator);
}

__global__
void ComputeMedianGPU_kernel(
    const int batch_size,
    const int dist_size,
    const float* input,
    int* median) {
  const int b = blockIdx.x;

  if (b < batch_size) {
    input += b * dist_size;

    int median_idx = 0;
    float sum = 0.0f;
    for (; median_idx < dist_size; ++median_idx) {
      sum += input[median_idx];
      if (sum > 0.5f) break;
    }

    median[b] = median_idx;
  }
}

void ComputeMedianGPU(
    const GPUDevice& d, const Tensor& input, Tensor* median) {
  const int64 batch_size = input.dim_size(0);
  const int64 dist_size = input.dim_size(1);

  // Our batch sizes are usually 16 or smaller, so we give each their own block.
  ComputeMedianGPU_kernel<<<batch_size, 1, 0, d.stream()>>>(
      batch_size, dist_size, input.flat<float>().data(),
      median->flat<int>().data());
}
}  // namespace tensorflow
#endif  // GOOGLE_CUDA
