// Copyright 2015 William Chan <williamchan@cmu.edu>.

#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/token_model.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

using namespace tensorflow;

namespace {
REGISTER_OP("S4ParseUtterance")
    .Attr("features_fbank_dim: int = 40")
    .Attr("features_len_max: int")
    .Attr("alignment_len_max: int")
    .Attr("tokens_len_max: int")
    .Attr("eow_weight: float = 1.0")
    .Attr("token_model: string = 'speech4/conf/token_model_character_simple.pbtxt'")
    .Attr("frame_stack: int = 1")
    .Attr("frame_skip: int = 1")
    .Input("serialized: string")
    .Output("features: features_len_max * float")
    .Output("alignment: alignment_len_max * int32")
    .Output("alignment_weight: alignment_len_max * float")
    .Output("features_fbank: features_len_max * float")
    .Output("features_len: int64")
    .Output("features_width: int64")
    .Output("features_weight: features_len_max* float")
    .Output("s_min: int32")
    .Output("s_max: int32")
    .Output("text: string")
    .Output("tokens: tokens_len_max * int32")
    .Output("tokens_pinyin: tokens_len_max * int32")
    .Output("tokens_len: int64")
    .Output("tokens_weights: tokens_len_max * float")
    .Output("tokens_pinyin_weights: tokens_len_max * float")
    .Output("uttid: string")
    .Doc(R"doc(
SPEECH4, parse an utterance!
)doc");

class S4ParseUtterance : public OpKernel {
 public:
  explicit S4ParseUtterance(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("features_fbank_dim", &features_fbank_dim_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("features_len_max", &features_len_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("alignment_len_max", &alignment_len_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tokens_len_max", &tokens_len_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("eow_weight", &eow_weight_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("frame_stack", &frame_stack_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("frame_skip", &frame_skip_));
    if (frame_stack_ == 0) frame_stack_ = 1;
    if (frame_skip_ == 0) frame_skip_ = 1;

    // Read the TokenModelProto.
    string token_model_pbtxt_path;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("token_model", &token_model_pbtxt_path));
    string token_model_pbtxt;
    TF_CHECK_OK(ReadFileToString(ctx->env(), token_model_pbtxt_path, &token_model_pbtxt));
    CHECK(protobuf::TextFormat::ParseFromString(token_model_pbtxt, &token_model_proto_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* serialized = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized));
    auto serialized_t = serialized->vec<string>();
    const int64 batch_size = serialized_t.size();

    OpOutputList output_list_features;
    OpOutputList output_list_alignment;
    OpOutputList output_list_alignment_weight;
    OpOutputList output_list_features_fbank;
    Tensor* output_tensor_features_len = nullptr;
    Tensor* output_tensor_features_width = nullptr;
    OpOutputList output_list_features_weight;
    Tensor* output_tensor_s_min = nullptr;
    Tensor* output_tensor_s_max = nullptr;
    Tensor* output_tensor_text = nullptr;
    OpOutputList output_list_tokens;
    Tensor* output_tensor_tokens_len = nullptr;
    OpOutputList output_list_tokens_pinyin;
    Tensor* output_tensor_uttid = nullptr;
    OpOutputList output_list_tokens_weights;
    OpOutputList output_list_tokens_pinyin_weights;

    for (int64 b = 0; b < batch_size; ++b) {
      // Parse our serialized string into our Example proto.
      Example ex;
      OP_REQUIRES(
          ctx, ParseProtoUnlimited(&ex, serialized_t(b)),
          errors::InvalidArgument("Could not parse example input, value: '",
                                  serialized_t(b), "'"));
      const auto& feature_dict = ex.features().feature();

      // Extract the features_len.
      const auto& features_len_iter = feature_dict.find("features_len");
      CHECK(features_len_iter != feature_dict.end());
      int64 features_len = features_len_iter->second.int64_list().value(0);

      // Extract the features.
      const auto& features_iter = feature_dict.find("features");
      CHECK(features_iter != feature_dict.end());
      const auto& features = features_iter->second.float_list();

      // The features_width is a function of len(features) and features_len.
      if (features.value().size()) CHECK_EQ(features.value().size() % features_len, 0);
      const int64 frame_width =
          features_len ? features.value().size() / features_len : 0;

      // Adjust for frame_stack and frame_skip.
      const int64 features_width = frame_width * frame_stack_;
      const int64 frame_total = features_len;
      features_len = features_len / frame_skip_;

      if (b == 0) {
        // Allocate the memory.
        OP_REQUIRES_OK(ctx, ctx->output_list("features", &output_list_features));
        OP_REQUIRES_OK(ctx, ctx->output_list("alignment", &output_list_alignment));
        OP_REQUIRES_OK(ctx, ctx->output_list("alignment_weight", &output_list_alignment_weight));
        OP_REQUIRES_OK(ctx, ctx->output_list("features_fbank", &output_list_features_fbank));
        OP_REQUIRES_OK(ctx, ctx->output_list("features_weight", &output_list_features_weight));
        for (int64 t = 0; t < features_len_max_; ++t) {
          TensorShape feature_shape({batch_size, features_width});
          Tensor* feature_slice = nullptr;
          output_list_features.allocate(t, feature_shape, &feature_slice);

          TensorShape feature_fbank_shape({batch_size, features_fbank_dim_});
          Tensor* feature_fbank_slice = nullptr;
          output_list_features_fbank.allocate(t, feature_fbank_shape, &feature_fbank_slice);

          Tensor* feature_weight = nullptr;
          output_list_features_weight.allocate(t, TensorShape({batch_size}), &feature_weight);

          std::fill_n(feature_slice->flat<float>().data(), feature_shape.num_elements(), 0.0f);
          std::fill_n(feature_fbank_slice->flat<float>().data(), feature_fbank_shape.num_elements(), 0.0f);
          std::fill_n(feature_weight->flat<float>().data(), batch_size, 0.0f);
        }

        for (int64 t = 0; t < alignment_len_max_; ++t) {
          Tensor* alignment_slice = nullptr;
          output_list_alignment.allocate(t, TensorShape({batch_size}), &alignment_slice);
          std::fill_n(alignment_slice->flat<int32>().data(), batch_size, 0);

          Tensor* alignment_weight_slice = nullptr;
          output_list_alignment_weight.allocate(t, TensorShape({batch_size}), &alignment_weight_slice);
          std::fill_n(alignment_weight_slice->flat<float>().data(), batch_size, 0.0f);
        }

        TensorShape x({batch_size});
        OP_REQUIRES_OK(
            ctx, ctx->allocate_output("features_len", TensorShape({batch_size}), &output_tensor_features_len));

        OP_REQUIRES_OK(
            ctx, ctx->allocate_output("features_width", TensorShape(), &output_tensor_features_width));
        output_tensor_features_width->flat<int64>().data()[0] = features_width;

        OP_REQUIRES_OK(
            ctx, ctx->allocate_output("s_min", TensorShape({batch_size}), &output_tensor_s_min));
        std::fill_n(output_tensor_s_min->flat<int32>().data(), batch_size, 0);
        OP_REQUIRES_OK(
            ctx, ctx->allocate_output("s_max", TensorShape({batch_size}), &output_tensor_s_max));
        std::fill_n(output_tensor_s_max->flat<int32>().data(), batch_size, 0);
        OP_REQUIRES_OK(
            ctx, ctx->allocate_output("text", TensorShape({batch_size}), &output_tensor_text));

        OP_REQUIRES_OK(ctx, ctx->output_list("tokens", &output_list_tokens));
        OP_REQUIRES_OK(ctx, ctx->output_list("tokens_pinyin", &output_list_tokens_pinyin));
        for (int64 s = 0; s < tokens_len_max_; ++s) {
          TensorShape token_shape({batch_size});
          Tensor* token_slice = nullptr;
          output_list_tokens.allocate(s, token_shape, &token_slice);
          std::fill_n(token_slice->flat<int32>().data(), token_shape.num_elements(), 0);

          TensorShape token_pinyin_shape({batch_size * 7});
          Tensor* token_pinyin_slice = nullptr;
          output_list_tokens_pinyin.allocate(s, token_pinyin_shape, &token_pinyin_slice);
          std::fill_n(token_pinyin_slice->flat<int32>().data(), token_pinyin_shape.num_elements(), 0);
        }

        OP_REQUIRES_OK(
            ctx, ctx->allocate_output("tokens_len", TensorShape({batch_size}), &output_tensor_tokens_len));

        OP_REQUIRES_OK(ctx, ctx->output_list("tokens_weights", &output_list_tokens_weights));
        OP_REQUIRES_OK(ctx, ctx->output_list("tokens_pinyin_weights", &output_list_tokens_pinyin_weights));
        for (int64 s = 0; s < tokens_len_max_; ++s) {
          {
            TensorShape weight_shape({batch_size});
            Tensor* weight_slice = nullptr;
            output_list_tokens_weights.allocate(s, weight_shape, &weight_slice);
            std::fill_n(weight_slice->flat<float>().data(), weight_shape.num_elements(), 0.0f);
          }

          {
            TensorShape pinyin_weight_shape({batch_size * 7});
            Tensor* pinyin_weight_slice = nullptr;
            output_list_tokens_pinyin_weights.allocate(s, pinyin_weight_shape, &pinyin_weight_slice);
            std::fill_n(pinyin_weight_slice->flat<float>().data(), pinyin_weight_shape.num_elements(), 0.0f);
          }
        }

        OP_REQUIRES_OK(
            ctx, ctx->allocate_output("uttid", TensorShape({batch_size}), &output_tensor_uttid));
      }

      // Copy the features across.
      output_tensor_features_len->flat<int64>().data()[b] = features_len;
      if (features_len > features_len_max_) {
        // LOG(WARNING) << "Utterance has feature_len: " << features_len << " but graph maximum is: " << features_len_max_;
        features_len = features_len_max_;
      }
      for (int64 t = 0; t < features_len; ++t) {
        // Copy the features from the proto to our Tensor.
        const int64 frame_remaining = frame_total - t * frame_skip_;
        CHECK_GE(frame_remaining, 0);
        // const int64 copy_width = std::min(features_width, frame_remaining);
        const int64 copy_width = features_width;
        CHECK_EQ(frame_skip_, 1);
        CHECK_EQ(frame_stack_, 1);

        Tensor* feature_slice = output_list_features[t];
        Tensor* feature_fbank_slice = output_list_features_fbank[t];

        std::copy_n(features.value().data() + t * frame_skip_ * frame_width,
                    copy_width,
                    feature_slice->flat<float>().data() + b * features_width);
        if (frame_stack_ == 1 && frame_skip_ == 1) {
          std::copy_n(features.value().data() + t * frame_skip_ * frame_width,
                      features_fbank_dim_,
                      feature_fbank_slice->flat<float>().data() + b * features_fbank_dim_);
        }

        Tensor* feature_weight = output_list_features_weight[t];
        feature_weight->flat<float>().data()[b] = 1.0f;
      }

      // See if we have an alignment.
      if (feature_dict.find("alignment") != feature_dict.end()) {
        const auto& alignment_iter = feature_dict.find("alignment");
        const auto& alignment = alignment_iter->second.int64_list();
        for (int64 t = 0; t < alignment.value().size(); ++t) {
          Tensor* alignment_slice = output_list_alignment[t];
          int32 phone = alignment.value(t);
          alignment_slice->flat<int32>().data()[b] = phone;

          Tensor* alignment_weight_slice = output_list_alignment_weight[t];
          if (phone == 4) {
            alignment_weight_slice->flat<float>().data()[b] = 0.1f;
          } else {
            alignment_weight_slice->flat<float>().data()[b] = 1.0f;
          }
        }
      }

      // Copy the text across.
      const auto& text_iter = feature_dict.find("text");
      CHECK(text_iter != feature_dict.end());
      const auto& text = text_iter->second.bytes_list().value(0);

      output_tensor_text->flat<string>().data()[b] = text;
      
      // Copy the tokens.
      const auto& tokens_iter = feature_dict.find("tokens");
      CHECK(tokens_iter != feature_dict.end());
      const auto& tokens = tokens_iter->second.int64_list();
      int64 tokens_len = tokens.value().size();

      output_tensor_tokens_len->flat<int64>().data()[b] = tokens_len;
      if (tokens_len > tokens_len_max_) {
        // LOG(WARNING) << "Utterance has tokens_len: " << tokens_len << " but graph maximum is: " << tokens_len_max_;
        tokens_len = tokens_len_max_;
      }
      for (int64 s = 0; s < tokens_len; ++s) {
        Tensor* token_slice = output_list_tokens[s];
        int32 token = tokens.value(s);
        token_slice->flat<int32>().data()[b] = token;

        Tensor* weight_slice = output_list_tokens_weights[s];
        Tensor* pinyin_weight_slice = output_list_tokens_pinyin_weights[s];
        if (token == token_model_proto_.token_eow()) {
          weight_slice->flat<float>().data()[b] = eow_weight_;
          for (int64 u = 0; u < 7; ++u) {
            pinyin_weight_slice->flat<float>().data()[b * 7 + u] = eow_weight_ / 7.0f;
          }
        } else {
          weight_slice->flat<float>().data()[b] = 1.0f;
          for (int64 u = 0; u < 7; ++u) {
            pinyin_weight_slice->flat<float>().data()[b * 7 + u] = 1.0f / 7.0f;
          }
        }
      }

      if (feature_dict.find("tokens_pinyin") != feature_dict.end()) {
        const auto& tokens_pinyin_iter = feature_dict.find("tokens_pinyin");
        CHECK(tokens_pinyin_iter != feature_dict.end());
        const auto& tokens_pinyin = tokens_pinyin_iter->second.int64_list();
        for (int64 s = 0; s < tokens_len; ++s) {
          Tensor* token_slice = output_list_tokens_pinyin[s];
          for (int64 u = 0; u < 7; ++u) {
            int32 token = tokens_pinyin.value(s * 7 + u);
            token_slice->flat<int32>().data()[b * 7 + u] = token;
          }
        }
      }  // else it is pre-zeroed.

      // s_min / s_max
      if (feature_dict.find("s_min") != feature_dict.end()) {
        const auto& s_min_iter = feature_dict.find("s_min");
        const auto& s_max_iter = feature_dict.find("s_max");

        output_tensor_s_min->flat<int>().data()[b] = s_min_iter->second.int64_list().value(0);
        output_tensor_s_max->flat<int>().data()[b] = s_max_iter->second.int64_list().value(0);
      }

      // Copy the uttid across.
      const auto& uttid_iter = feature_dict.find("uttid");
      CHECK(uttid_iter != feature_dict.end());
      const auto& uttid = uttid_iter->second.bytes_list().value(0);

      output_tensor_uttid->flat<string>().data()[b] = uttid;
    }
  }

 protected:
  int64 features_fbank_dim_;
  int64 features_len_max_;
  int64 alignment_len_max_;
  int64 tokens_len_max_;
  int64 tokens_pinyin_len_max_;
  float eow_weight_;
  int64 frame_stack_;
  int64 frame_skip_;

  speech4::TokenModelProto token_model_proto_;
};

REGISTER_KERNEL_BUILDER(Name("S4ParseUtterance").Device(DEVICE_CPU), S4ParseUtterance);
}
