#define EIGEN_USE_THREADS

#include <algorithm>
#include <iterator>
#include <limits>
#include <sstream>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
// #include "tensorflow/core/lib/gtl/edit_distance.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

namespace {
void ExtractSequence(
    const OpInputList& list, const Tensor& len,
    std::vector<std::vector<int32>>* sequence) {
  const int64 batch_size = list[0].dim_size(0);
  sequence->resize(batch_size);

  for (int64 t = 0; t < list.size(); ++t) {
    for (int64 b = 0; b < batch_size; ++b) {
      if (t < len.flat<int64>()(b)) {
        const int32 token = list[t].vec<int32>()(b);
        (*sequence)[b].emplace_back(token);
      }
    }
  }
}

void ExtractSequence(
    const OpInputList& list, std::vector<std::vector<int32>>* sequence) {
  const int64 batch_size = list[0].dim_size(0);
  sequence->resize(batch_size);

  for (int64 t = 0; t < list.size(); ++t) {
    for (int64 b = 0; b < batch_size; ++b) {
      const int32 token = list[t].vec<int32>()(b);
      (*sequence)[b].emplace_back(token);
    }
  }
}

void CollapseSequence(
    int32 blank_token, std::vector<std::vector<int32>>* sequences) {
  for (int64 b = 0; b < static_cast<int64>(sequences->size()); ++b) {
    std::vector<int32> filtered;
    std::copy_if(
        (*sequences)[b].begin(), (*sequences)[b].end(),
        std::back_inserter(filtered), [blank_token](const int32 x)->bool {
            return x != blank_token;
        });
    (*sequences)[b].swap(filtered);
  }
}

template <typename T>
std::string VectorToString(const std::vector<T>& vec) {
  std::stringstream ss;
  for (int64 i = 0; i < vec.size(); ++i) {
    if (i > 0) ss << ", ";
    ss << vec[i];
  }
  return ss.str();
}

class EditDistance {
 public:
  enum EditType {
    NONE = 0,
    BLANK,
    INSERTION,
    DELETION,
    SUBSTITUTION
  };

  explicit EditDistance() {}

  const std::vector<EditType> edits() const {
    return edits_;
  }

  void append_edit(EditType type) {
    edits_.emplace_back(type);
    compute_edits();
  }

  void compute_edits() {
    int64 edits = 0;
    for (const EditType& edit : edits_) {
      edits += (edit == INSERTION) || (edit == DELETION) || (edit == SUBSTITUTION);
    }
    edit_distance_ = edits;
  }

  int64 edit_distance() const {
    return edit_distance_;
  }

  void clear() {
    edits_.clear();
    edit_distance_ = 0;
  }

  static std::string TypeToString(const EditType& type) {
    if (type == NONE) {
      return " ";
    } else if (type == BLANK) {
      return "_";
    } else if (type == INSERTION) {
      return "I";
    } else if (type == DELETION) {
      return "D";
    } else if (type == SUBSTITUTION) {
      return "S";
    }
  }

 private:
  std::vector<EditType> edits_;
  int64 edit_distance_ = 0;
};

// Computes the EditDistance and returns an alignment of the errors.
void ComputeEditDistance(
    const std::vector<int32>& ref, const std::vector<int32>& hyp_original,
    const int32 blank_token, EditDistance* err) {
  std::vector<int32> hyp;
  for (int32 token : hyp_original) {
    if (token != blank_token) {
      hyp.emplace_back(token);
    }
  }

  const int64 ref_size = ref.size();
  const int64 hyp_size = hyp.size();

  std::vector<EditDistance> v0(hyp_size + 1);
  std::vector<EditDistance> v1(hyp_size + 1);

  for (int64 i = 0; i < hyp_size + 1; ++i) {
    for (int64 j = 0; j < i; ++j) {
      v1[i].append_edit(EditDistance::DELETION);
    }
  }

  for (int64 i = 0; i < ref_size; ++i) {
    std::swap(v0, v1);

    v1[0].clear();
    for (int64 k = 0; k < i + 1; ++k) {
      v1[0].append_edit(EditDistance::INSERTION);
    }

    for (int64 j = 0; j < hyp_size; ++j) {
      if (ref[i] == hyp[j]) {
        v1[j + 1] = v0[j];
        v1[j + 1].append_edit(EditDistance::NONE);
      } else {
        int64 deletion_cost = v1[j].edit_distance() + 1;
        int64 insertion_cost = v0[j + 1].edit_distance() + 1;
        int64 substitution_cost = v0[j].edit_distance() + 1;

        if (deletion_cost < insertion_cost &&
            deletion_cost < substitution_cost) {
          v1[j + 1] = v1[j];
          v1[j + 1].append_edit(EditDistance::DELETION);
        } else if (insertion_cost < substitution_cost) {
          v1[j + 1] = v0[j + 1];
          v1[j + 1].append_edit(EditDistance::INSERTION);
        } else {
          v1[j + 1] = v0[j];
          v1[j + 1].append_edit(EditDistance::SUBSTITUTION);
        }
      }
    }
  }
  
  *err = v1[hyp.size()];
}

class CCTCWeaklySupervisedAlignmentLabelOp : public OpKernel {
 public:
  explicit CCTCWeaklySupervisedAlignmentLabelOp(OpKernelConstruction* ctx)
    : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, generator_.Init(ctx));

    OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_token", &blank_token_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lpad", &lpad_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rpad", &rpad_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("flen_max", &flen_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tlen", &tlen_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("hlen_max", &hlen_max_));
  }

  void Compute(OpKernelContext* ctx) override {
    OpInputList ref_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("ref", &ref_list));

    OpInputList hyp_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("hyp", &hyp_list));

    const Tensor* ref_len = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("ref_len", &ref_len));

    const Tensor* hyp_len = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("hyp_len", &hyp_len));

    std::vector<std::vector<int32>> refs;
    ExtractSequence(ref_list, *ref_len, &refs);

    std::vector<std::vector<int32>> hyps;
    ExtractSequence(hyp_list, &hyps);
    CollapseSequence(blank_token_, &hyps);

    const int64 batch_size = ref_len->dim_size(0);

    Tensor* label = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("label", TensorShape({batch_size}), &label));

   Tensor* label_weight = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output("label_weight", TensorShape({batch_size}), &label_weight));

    // Random number generator setup.
    typedef random::UniformDistribution<random::PhiloxRandom, float>
        Distribution;
    Distribution dist;

    // Sample our random numbers.
    const int kGroupSize = Distribution::kResultElementCount;
    auto local_generator = generator_.ReserveSamples32(batch_size);
    std::vector<float> sample_prob(batch_size);
    for (int64 i = 0; i < batch_size; i += kGroupSize) {
      auto samples = dist(&local_generator);
      std::copy(&samples[0], &samples[0] + kGroupSize, &sample_prob[i]);
    }

    for (int64 b = 0; b < batch_size; ++b) {
      float w_t = 1.0f;
      int32 l_t = blank_token_;

      const std::vector<int32>& ref = refs[b];
      const std::vector<int32>& hyp = hyps[b];
      CHECK(std::equal(hyp.begin(), hyp.end(), ref.begin()));

      // ali_len: the length of the total alignment including blanks.
      const int32 ali_len = hyp_len->flat<int64>()(b);

      // Compute the minimum lpad.
      const int32 ref_size = ref.size();
      const int32 lpad = std::min(lpad_, ali_len - ref_size - 1);
      const int32 rpad = std::min(rpad_, ali_len - ref_size - lpad - 1);

      if (tlen_ > lpad) {
        if (hyp.size() >= ref.size()) {
          CHECK_EQ(hyp.size(), ref.size());
          w_t = 0.0f;
        } else {
          // Number of tokens (including blank) we still have to emit.
          const int32 c_t = ali_len - tlen_ - rpad;
          CHECK_GE(c_t, 0);

          // Number of tokens (excluding blank) we still have to emit.
          const int32 d_t = ref.size() - hyp.size();
          CHECK_GE(d_t, 0);

          // Number of blanks we still have to emit.
          const int32 b_t = c_t - d_t;
          CHECK_GE(b_t, 0);

          const float blank_prob =
              static_cast<float>(b_t) / static_cast<float>(c_t);
          if (sample_prob[b] > blank_prob) {
            // Get the next correct token in the sequence.
            CHECK_GT(ref.size(), hyp.size());
            l_t = ref[hyp.size()];
          }
        }
      }

      label->flat<int32>()(b) = l_t;
      label_weight->flat<float>()(b) = w_t;
    }
  }

 private:
  int32 blank_token_;
  int32 lpad_;
  int32 rpad_;
  int32 flen_max_;
  int32 tlen_;
  int32 hlen_max_;

  GuardedPhiloxRandom generator_;
};
REGISTER_KERNEL_BUILDER(Name("CCTCWeaklySupervisedAlignmentLabel")
                            .Device(DEVICE_CPU),
                        CCTCWeaklySupervisedAlignmentLabelOp);


class CCTCBootstrapAlignmentOp : public OpKernel {
 public:
  explicit CCTCBootstrapAlignmentOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_token", &blank_token_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("lpad", &lpad_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rpad", &rpad_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("features_len_max", &features_len_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tokens_len_max", &tokens_len_max_));
  }

  void Compute(OpKernelContext* ctx) override {
    CPUDevice device = ctx->eigen_device<CPUDevice>();

    OpInputList tokens_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("tokens", &tokens_list));

    const Tensor* tokens_len = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("tokens_len", &tokens_len));

    const Tensor* features_len = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("features_len", &features_len));

    std::vector<std::vector<int32>> tokens;
    ExtractSequence(tokens_list, *tokens_len, &tokens);

    const int64 batch_size = tokens_len->dim_size(0);

    OpOutputList tokens_aligned_list;
    OpOutputList tokens_aligned_weight_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("tokens_aligned", &tokens_aligned_list));
    OP_REQUIRES_OK(ctx, ctx->output_list("tokens_aligned_weight", &tokens_aligned_weight_list));
    for (int64 t = 0; t < features_len_max_; ++t) {
      Tensor* tokens_aligned_tensor = nullptr;
      tokens_aligned_list.allocate(t, TensorShape({batch_size}), &tokens_aligned_tensor);
      tokens_aligned_tensor->flat<int32>().device(device) =
          tokens_aligned_tensor->flat<int32>().constant(blank_token_);

      Tensor* tokens_aligned_weight_tensor = nullptr;
      tokens_aligned_weight_list.allocate(t, TensorShape({batch_size}), &tokens_aligned_weight_tensor);
      tokens_aligned_weight_tensor->flat<float>().device(device) =
          tokens_aligned_weight_tensor->flat<float>().constant(0.0f);
    }

    for (int64 b = 0; b < batch_size; ++b) {
      const std::vector<int32>& tokens_b = tokens[b];

      const int64 flen = features_len->flat<int64>()(b);
      const int64 tlen = tokens_b.size();

      const int32 lpad = std::min(lpad_, static_cast<int32>(flen - tlen));
      const int32 rpad = std::min(rpad_, static_cast<int32>(flen - tlen) - lpad);

      const float f_per_t = static_cast<float>(flen - lpad - rpad) / static_cast<float>(tlen);
      CHECK_GE(f_per_t, 1);

      for (int t = 0; t < tlen; ++t) {
        int32 token = tokens_b[t];
        tokens_aligned_list[lpad + t * f_per_t]->flat<int32>()(b) = token;
      }
      for (int t = 0; t < flen; ++t) {
        tokens_aligned_weight_list[t]->flat<float>()(b) = 1.0f;
      }
    }
  }

 private:
  int32 blank_token_;
  int32 lpad_;
  int32 rpad_;
  int32 features_len_max_;
  int32 tokens_len_max_;
};

REGISTER_KERNEL_BUILDER(Name("CCTCBootstrapAlignment")
                            .Device(DEVICE_CPU),
                        CCTCBootstrapAlignmentOp);

class CCTCEditDistanceOp : public OpKernel {
 public:
  explicit CCTCEditDistanceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_token", &blank_token_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("features_len_max", &features_len_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tokens_len_max", &tokens_len_max_));
  }

  void Compute(OpKernelContext* ctx) override {
    // Get the ref and hyp.
    OpInputList ref_list;
    OpInputList hyp_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("ref", &ref_list));
    OP_REQUIRES_OK(ctx, ctx->input_list("hyp", &hyp_list));

    const Tensor* ref_len = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("ref_len", &ref_len));

    std::vector<std::vector<int32>> ref;
    ExtractSequence(ref_list, *ref_len, &ref);
    std::vector<std::vector<int32>> hyp;
    ExtractSequence(hyp_list, &hyp);

    const int64 batch_size = ref_list[0].dim_size(0);
    Tensor* tensor_edit_distance = nullptr;
    ctx->allocate_output(
        "edit_distance", TensorShape({batch_size}), &tensor_edit_distance);

    for (int64 b = 0; b < batch_size; ++b) {
      EditDistance err;
      ComputeEditDistance(ref[b], hyp[b], blank_token_, &err);

      const int64 edit_distance = err.edit_distance();
      tensor_edit_distance->vec<int64>()(b) = edit_distance;
    }
  }

 private:
  int32 blank_token_;
  int32 features_len_max_;
  int32 tokens_len_max_;
};
REGISTER_KERNEL_BUILDER(Name("CCTCEditDistance")
                            .Device(DEVICE_CPU),
                        CCTCEditDistanceOp);

class CCTCEditDistanceReinforceGradOp : public OpKernel {
 public:
  explicit CCTCEditDistanceReinforceGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blank_token", &blank_token_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("features_len_max", &features_len_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tokens_len_max", &tokens_len_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("discount_factor", &discount_factor_));
  }

  // This function computes the aligned errors for the edit distance.
  void ComputeRewards(
      OpKernelContext* ctx, const std::vector<int32>& ref,
      const std::vector<int32>& hyp, const EditDistance& err, int32 blank_token,
      float discount_factor, std::vector<float>* rewards) {
    int64 ref_idx = 0;
    int64 hyp_idx = 0;
    int64 err_idx = 0;

    int64 insertions = 0;

    // This is the error aligned to the hyp.
    std::vector<int32> err_aligned;
    while (hyp_idx < static_cast<int64>(hyp.size())) {
      if (hyp[hyp_idx] == blank_token) {
        // Blank token, we advance the hyp by one and there is no error here?
        ++hyp_idx;
        err_aligned.emplace_back(0);
      } else {
        CHECK_LT(err_idx, err.edits().size());
        EditDistance::EditType edit = err.edits()[err_idx];

        if (edit == EditDistance::NONE) {
          ++ref_idx;
          ++hyp_idx;
          ++err_idx;
          err_aligned.emplace_back(insertions);
          insertions = 0;
        } else if (edit == EditDistance::DELETION) {
          ++hyp_idx;
          ++err_idx;
          err_aligned.emplace_back(insertions + 1);
          insertions = 0;
        } else if (edit == EditDistance::INSERTION) {
          ++ref_idx;
          ++err_idx;
          // err_aligned.emplace_back(1);
          ++insertions;
        } else if (edit == EditDistance::SUBSTITUTION) {
          ++ref_idx;
          ++hyp_idx;
          ++err_idx;
          err_aligned.emplace_back(insertions + 1);
          insertions = 0;
        } else {
          ctx->SetStatus(errors::InvalidArgument("Unknown edit type."));
        }
      }
    }
    CHECK_EQ(err_aligned.size(), hyp.size());

    rewards->clear();
    for (int64 t = 0; t < static_cast<int64>(hyp.size()); ++t) {
      float r = 0.0f;
      for (int64 u = err_aligned.size() - 1; u >= t; --u) {
        r += err_aligned[u] + discount_factor * r;
      }
      rewards->emplace_back(r);
    }
    CHECK_EQ(rewards->size(), hyp.size());
  }

  void Compute(OpKernelContext* ctx) override {
    CPUDevice device = ctx->eigen_device<CPUDevice>();

    // Get the ref and hyp sequences.
    OpInputList ref_list;
    OpInputList hyp_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("ref", &ref_list));
    OP_REQUIRES_OK(ctx, ctx->input_list("hyp", &hyp_list));

    const Tensor* ref_len = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("ref_len", &ref_len));

    std::vector<std::vector<int32>> ref;
    ExtractSequence(ref_list, *ref_len, &ref);
    std::vector<std::vector<int32>> hyp;
    ExtractSequence(hyp_list, &hyp);

    // Get our predicted probs.
    OpInputList hyp_probs_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("hyp_probs", &hyp_probs_list));

    // Get the predicted baseline.
    OpInputList hyp_baseline_list;
    OP_REQUIRES_OK(ctx, ctx->input_list("hyp_baseline", &hyp_baseline_list));

    const int64 batch_size = ref_list[0].dim_size(0);
    const int64 vocab_size = hyp_probs_list[0].dim_size(1);

    OpOutputList hyp_logits_backprop_list;
    OpOutputList hyp_baseline_backprop_list;
    OP_REQUIRES_OK(ctx, ctx->output_list("hyp_logits_backprop", &hyp_logits_backprop_list));
    OP_REQUIRES_OK(ctx, ctx->output_list("hyp_baseline_backprop", &hyp_baseline_backprop_list));
    for (int64 t = 0; t < features_len_max_; ++t) {
      Tensor* hyp_logits_backprop_tensor = nullptr;
      hyp_logits_backprop_list.allocate(
          t, TensorShape({batch_size, vocab_size}), &hyp_logits_backprop_tensor);
      hyp_logits_backprop_tensor->flat<float>().device(device) =
          hyp_logits_backprop_tensor->flat<float>().constant(0.0f);

      Tensor* hyp_baseline_backprop_tensor = nullptr;
      hyp_baseline_backprop_list.allocate(
          t, TensorShape({batch_size, 1}), &hyp_baseline_backprop_tensor);
      hyp_baseline_backprop_tensor->flat<float>().device(device) =
          hyp_baseline_backprop_tensor->flat<float>().constant(0.0f);
    }

    for (int64 b = 0; b < batch_size; ++b) {
      EditDistance err;
      ComputeEditDistance(ref[b], hyp[b], blank_token_, &err);

      std::vector<float> rewards;
      ComputeRewards(
          ctx, ref[b], hyp[b], err, blank_token_, discount_factor_, &rewards);

      for (int64 t = 0; t < static_cast<int64>(rewards.size()); ++t) {
        // de/dx for the baseline prediction is simply (hyp_baseline - rewards).
        const float delta_baseline = rewards[t] - hyp_baseline_list[t].flat<float>()(b);
        hyp_baseline_backprop_list[t]->flat<float>()(b) = -delta_baseline;

        const int32 label = ref[b][t];
        for (int64 l = 0; l < vocab_size; ++l) {
          // TODO(williamchan): I think I need to flip the sign of the gradient... because we want to minimize the reward rather than maximize (since our reward is EditDistance).
          const float prob = hyp_probs_list[t].matrix<float>()(b, l);
          hyp_logits_backprop_list[t]->matrix<float>()(b, l) =
              (prob - (l == label)) * delta_baseline;
        }
      }
    }
  }

 private:
  int32 blank_token_;
  int32 features_len_max_;
  int32 tokens_len_max_;
  float discount_factor_;
};

REGISTER_KERNEL_BUILDER(Name("CCTCEditDistanceReinforceGrad")
                            .Device(DEVICE_CPU),
                        CCTCEditDistanceReinforceGradOp);
}  // namespace
}  // namespace tensor
