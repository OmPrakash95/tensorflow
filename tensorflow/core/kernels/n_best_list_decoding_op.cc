#define EIGEN_USE_THREADS

#include <algorithm>
#include <utility>
#include <vector>


#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/random/random_distributions.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

class NBestListDecodingOp : public OpKernel {
 public:
  explicit NBestListDecodingOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    #define INPUT_TENSOR(T)                                                    \
      const Tensor* T = nullptr;                                               \
      OP_REQUIRES_OK(ctx, ctx->input(#T, &T));

    #define OUTPUT_TENSOR(NAME, SHAPE)                                         \
      Tensor* NAME = nullptr;                                                  \
      ctx->allocate_output(#NAME, SHAPE, &NAME);

    OpInputList state;
    OP_REQUIRES_OK(ctx, ctx->input_list("state", &state));

    INPUT_TENSOR(prefix_logprob);
    INPUT_TENSOR(token_logprob);

    const int64 batch_size = token_logprob->dim_size(0);
    const int64 vocab_size = token_logprob->dim_size(1);

    OpOutputList path_state;
    OP_REQUIRES_OK(ctx, ctx->output_list("path_state", &path_state));
    for (int64 s = 0; s < state.size(); ++s) {
      Tensor* t = nullptr;
      OP_REQUIRES_OK(ctx, path_state.allocate(s, state[s].shape(), &t));
    }

    OUTPUT_TENSOR(path, TensorShape({batch_size}));
    OUTPUT_TENSOR(token, TensorShape({batch_size}));
    OUTPUT_TENSOR(logprob, TensorShape({batch_size}));

    std::vector<std::pair<float, std::pair<int64, int64>>> n_best_list;
    for (int64 p = 0; p < batch_size; ++p) {
      const float prefix = prefix_logprob->vec<float>()(p);
      for (int64 v = 0; v < vocab_size; ++v) {
        n_best_list.emplace_back(std::make_pair(
              prefix + token_logprob->matrix<float>()(p, v),
              std::make_pair(p, v)));
      }
    }

    // Sort and prune to n-best list (n == batch_size).
    std::sort(n_best_list.begin(), n_best_list.end(), [](
          const std::pair<float, std::pair<int64, int64>>& a,
          const std::pair<float, std::pair<int64, int64>>& b) {
      return a.first > b.first;
    });
    n_best_list.resize(batch_size);

    for (int64 p = 0; p < batch_size; ++p) {
      path->vec<int32>()(p) = n_best_list[p].second.first;
      token->vec<int32>()(p) = n_best_list[p].second.second;
      logprob->vec<float>()(p) = n_best_list[p].first;

      // Update state.
      for (int64 s = 0; s < state.size(); ++s) {
        path_state[s]->matrix<float>().chip(p, 0).device(
            ctx->eigen_device<CPUDevice>()) =
                state[s].matrix<float>().chip(n_best_list[p].second.first, 0);
      }
    }
  }
};

}  // namespace tensorflow
