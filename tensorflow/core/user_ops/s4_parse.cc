#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/protobuf.h"

#include "tensorflow/core/platform/logging.h"

using namespace tensorflow;

namespace {
REGISTER_OP("S4ParseUtterance")
    .Attr("features_len_max: int")
    .Attr("tokens_len_max: int")
    .Input("serialized: string")
    .Output("features: features_len_max * float")
    .Output("features_len: int64")
    .Output("text: string")
    .Output("tokens: tokens_len_max * int64")
    .Output("tokens_len: int64")
    .Output("uttid: string")
    .Doc(R"doc(
SPEECH4, parse an utterance!
)doc");

class S4ParseUtterance : public OpKernel {
 public:
  explicit S4ParseUtterance(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("features_len_max", &features_len_max_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("tokens_len_max", &tokens_len_max_));
  }

  void Compute(OpKernelContext* ctx) override {
    // Parse our serialized string into our Example proto.
    const Tensor* serialized;
    OP_REQUIRES_OK(ctx, ctx->input("serialized", &serialized));
    auto serialized_t = serialized->scalar<string>();

    Example ex;
    OP_REQUIRES(
        ctx, ParseProtoUnlimited(&ex, serialized_t()) & false,
        errors::InvalidArgument("Could not parse example input, value: '",
                                serialized_t(), "'"));
    const auto& feature_dict = ex.features().feature();

    // Extract the features_len.
    const auto& features_len_iter = feature_dict.find("features_len");
    CHECK(features_len_iter != feature_dict.end());
    const int64 features_len = features_len_iter->second.int64_list().value(0);

    Tensor* output_tensor_features_len = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("features_len", TensorShape(), &output_tensor_features_len));
    *output_tensor_features_len->flat<int64>().data() = features_len;

    // Extract the features.
    const auto& features_iter = feature_dict.find("features");
    CHECK(features_iter != feature_dict.end());
    const auto& features = features_iter->second.float_list();

    // The features_width is a function of len(features) and features_len.
    CHECK_EQ(features.value().size() % features_len, 0);
    const int64 features_width = features.value().size() / features_len;

    // Copy the features across.
    OpOutputList output_list_features;
    OP_REQUIRES_OK(ctx, ctx->output_list("features", &output_list_features));

    for (int64 t = 0; t < features_len; ++t) {
      TensorShape feature_shape;
      feature_shape.AddDim(1);
      feature_shape.AddDim(features_width);

      Tensor* feature_slice = nullptr;
      output_list_features.allocate(t, feature_shape, &feature_slice);

      const int64 offset = t * feature_shape.num_elements();

      std::copy_n(features.value().data() + offset, feature_shape.num_elements(), feature_slice->flat<float>().data());
    }

    // Padding.
    for (int64 t = features_len; t < features_len_max_; ++t) {
      TensorShape feature_shape;
      feature_shape.AddDim(1);
      feature_shape.AddDim(features_width);

      Tensor* feature_slice = nullptr;
      output_list_features.allocate(t, feature_shape, &feature_slice);

      std::fill_n(feature_slice->flat<float>().data(), feature_shape.num_elements(), 0.0f);
    }

    // Copy the text across.
    const auto& text_iter = feature_dict.find("text");
    CHECK(text_iter != feature_dict.end());
    const auto& text = features_iter->second.bytes_list().value(0);

    Tensor* output_tensor_text = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("text", TensorShape(), &output_tensor_text));
    output_tensor_text->flat<string>()(0) = text;
  }

 protected:
  int64 features_len_max_;
  int64 tokens_len_max_;
};

REGISTER_KERNEL_BUILDER(Name("S4ParseUtterance").Device(DEVICE_CPU), S4ParseUtterance);
}
