#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/contrib/speech/data/token_model.pb.h"


namespace tensorflow {

class ParseUtteranceOp : public OpKernel {
 public:
  explicit ParseUtteranceOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feats_dim", &feats_dim_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feats_len_max", &feats_len_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tokens_len_max", &tokens_len_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("token_model", &token_model_path_));

    // Load the token model.
    string token_model_pbtxt;
    TF_CHECK_OK(ReadFileToString(ctx->env(), token_model_path_,
                                 &token_model_pbtxt));
    CHECK(protobuf::TextFormat::ParseFromString(token_model_pbtxt,
                                                &token_model_proto_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* serialized_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized_tensor));

    const int64 batch_size = serialized_tensor->dim_size(0);

    OpOutputList feats_tensor_list;
    OpOutputList tokens_tensor_list;
    OpOutputList tokens_weight_tensor_list;
    Tensor* feats_len_tensor = nullptr;
    Tensor* tokens_len_tensor = nullptr;
    Tensor* text_tensor = nullptr;
    Tensor* uttid_tensor = nullptr;

    for (int64 t = 0; t < feats_len_max_; ++t) {
      Tensor* feats_tensor = nullptr;
      feats_tensor_list.allocate(t, {batch_size, feats_dim_}, &feats_tensor);
      std::fill_n(feats_tensor->flat<float>().data(),
                  feats_tensor->NumElements(), 0.0f);
    }

    for (int64 t = 0; t < tokens_len_max_; ++t) {
      Tensor* tokens_tensor = nullptr;
      tokens_tensor_list.allocate(t, {batch_size}, &tokens_tensor);
      std::fill_n(tokens_tensor->flat<int64>().data(),
                  tokens_tensor->NumElements(), 0);

      Tensor* tokens_weight_tensor = nullptr;
      tokens_weight_tensor_list.allocate(t, {batch_size}, &tokens_weight_tensor);
      std::fill_n(tokens_weight_tensor->flat<float>().data(),
                  tokens_weight_tensor->NumElements(), 0.0f);
    }

    OP_REQUIRES_OK(ctx,
        ctx->allocate_output("feats_len", {batch_size}, &feats_len_tensor));
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output("tokens_len", {batch_size}, &tokens_len_tensor));
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output("text", {batch_size}, &text_tensor));
    OP_REQUIRES_OK(ctx,
        ctx->allocate_output("uttid", {batch_size}, &uttid_tensor));

    for (int64 b = 0; b < batch_size; ++b) {
      // Parse our serialized string into our Example proto.
      Example ex;
      OP_REQUIRES(
          ctx, ParseProtoUnlimited(&ex, serialized_tensor->vec<string>()(b)),
          errors::InvalidArgument("Could not parse example input, value: '",
                                  serialized_tensor->vec<string>()(b), "'"));
      const auto& feature_dict = ex.features().feature();

      // Extract the feats_len.
      const auto& feats_len_iter = feature_dict.find("feats_len");
      CHECK(feats_len_iter != feature_dict.end());
      int64 feats_len = feats_len_iter->second.int64_list().value(0);
      feats_len_tensor->vec<int64>()(b) = feats_len;

      // Extract the features.
      const auto& feats_iter = feature_dict.find("feats");
      CHECK(feats_iter != feature_dict.end());
      const auto& feats = feats_iter->second.float_list();
      CHECK_EQ(feats.value().size() / feats_len, feats_dim_);
      if (feats_len > feats_len_max_) {
        feats_len = feats_len_max_;
      }
      for (int64 t = 0; t < feats_len; ++t) {
        Tensor* feats_tensor = feats_tensor_list[t];
        std::copy_n(feats.value().data() + t * feats_dim_, feats_dim_,
                    feats_tensor->flat<float>().data() + b * feats_dim_);
      }

      // Extract the tokens_len.
      const auto& tokens_iter = feature_dict.find("tokens");
      CHECK(tokens_iter != feature_dict.end());
      const auto& tokens = tokens_iter->second.int64_list();
      int64 tokens_len = tokens.value().size();
      tokens_len_tensor->vec<int64>()(b) = tokens_len;
      if (tokens_len > tokens_len_max_) {
        tokens_len = tokens_len_max_;
      }
      for (int64 s = 0; s < tokens_len; ++s) {
        Tensor* tokens_tensor = tokens_tensor_list[s];
        int64 token = tokens.value(s);
        tokens_tensor->vec<int64>().data()[b] = token;

        Tensor* tokens_weight_tensor = tokens_weight_tensor_list[s];
        tokens_weight_tensor->vec<float>().data()[b] = 1.0f;
      }

      // Extract the text.
      const auto& text_iter = feature_dict.find("text");
      CHECK(text_iter != feature_dict.end());
      const auto& text = text_iter->second.bytes_list().value(0);
      text_tensor->vec<string>().data()[b] = text;

      // Extract the uttid.
      const auto& uttid_iter = feature_dict.find("uttid");
      CHECK(uttid_iter != feature_dict.end());
      const auto& uttid = uttid_iter->second.bytes_list().value(0);
      uttid_tensor->vec<string>().data()[b] = uttid;
    }
  }

 private:
  int64 feats_dim_;
  int64 feats_len_max_;
  int64 tokens_len_max_;
  string token_model_path_;
  TokenModelProto token_model_proto_;
};

REGISTER_KERNEL_BUILDER(
    Name("ParseUtterance")
        .Device(DEVICE_CPU),
    ParseUtteranceOp);

}  // end namespace tensorflow
