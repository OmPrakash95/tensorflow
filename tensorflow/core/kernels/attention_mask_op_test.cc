#include <functional>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {
namespace {

class AttentionMaskOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    RequireDefaultOps();
    ASSERT_OK(NodeDefBuilder("myop", "AttentionMask")
                  .Input(FakeInput())
                  .Input(FakeInput())
                  .Attr("fill_value", 0.0f)
                  .Finalize(node_def()));
    ASSERT_OK(InitOp());
  }
};

TEST_F(AttentionMaskOpTest, AttentionMask_02) {
  MakeOp();

  AddInputFromArray<int64>(TensorShape({2}), {0, 2});
  AddInputFromArray<float>(TensorShape({2, 2}),
                           {0.0f, 1.0f, 2.0f, 3.0f});

  ASSERT_OK(RunOpKernel());

  // Check the new state of the input
  Tensor* params_tensor = GetOutput(0);
  Tensor expected(allocator(), DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(
      &expected, {0.0f, 0.0f, 2.0f, 3.0f});
  test::ExpectTensorEqual<float>(expected, *params_tensor);
}

}  // end namespace
}  // end namespace tensorflow
