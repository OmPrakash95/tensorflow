#define EIGEN_USE_THREADS

#include <limits>

#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/edit_distance.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

template <typename T>
class EditDistanceListOp : public OpKernel {
 public:
  explicit EditDistanceListOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("eos_token", &eos_token_));
  }

  void ExtractSequence(
      const OpInputList& list, std::vector<std::vector<T>>* sequence) {
    const int64 batch_size = list[0].dim_size(0);
    sequence->resize(batch_size);

    std::vector<bool> terminated(batch_size, false);
    for (int64 t = 0; t < list.size(); ++t) {
      for (int64 b = 0; b < batch_size; ++b) {
        if (!terminated[b]) {
          const T token = list[t].vec<T>()(b);
          if (token == eos_token_) {
            terminated[b] = true;
          } else {
            (*sequence)[b].emplace_back(token);
          }
        }
      }
    }
  }

  void Compute(OpKernelContext* ctx) override {
    OpInputList ref_list;
    OpInputList hyp_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("ref", &ref_list));
    OP_REQUIRES_OK(ctx, ctx->input_list("hyp", &hyp_list));

    std::vector<std::vector<T>> ref;
    std::vector<std::vector<T>> hyp;
    ExtractSequence(ref_list, &ref);
    ExtractSequence(hyp_list, &hyp);

    const int64 batch_size = ref_list[0].dim_size(0);
    const TensorShape output_shape({batch_size});

    Tensor* tensor_edit_distance = nullptr;
    ctx->allocate_output(
        "edit_distance", output_shape, &tensor_edit_distance);

    Tensor* tensor_ref_length = nullptr;
    ctx->allocate_output(
        "ref_length", output_shape, &tensor_ref_length);

    auto cmp = std::equal_to<T>();
    for (int64 b = 0; b < batch_size; ++b) {
      const int64 edit_distance =
          gtl::LevenshteinDistance<T>(ref[b], hyp[b], cmp);
      tensor_edit_distance->vec<int64>()(b) = edit_distance;

      const int64 ref_length = ref[b].size();
      tensor_ref_length->vec<int64>()(b) = ref_length;
    }
  }

 private:
  T eos_token_;
};

#define REGISTER_CPU_KERNEL(T)      \
  REGISTER_KERNEL_BUILDER(          \
      Name("EditDistanceList")      \
          .Device(DEVICE_CPU)       \
          .TypeConstraint<T>("T"),  \
      EditDistanceListOp<T>);

REGISTER_CPU_KERNEL(int32);
REGISTER_CPU_KERNEL(int64);

#undef REGISTER_CPU_KERNEL

}  // end namespace tensor
