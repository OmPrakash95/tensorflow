licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

tf_custom_op_library(
    name = "python/ops/_edit_distance_ops.so",
    srcs = [
        "ops/edit_distance_ops.cc",
        "kernels/edit_distance_ops.cc",
    ],
)

tf_gen_op_libs(
    op_lib_names = ["edit_distance_ops"],
)

tf_gen_op_wrapper_py(
    name = "edit_distance_ops",
    deps = [":edit_distance_ops_op_lib"],
)

py_library(
    name = "edit_distance_py",
    srcs = ["__init__.py"] + glob(["python/ops/*.py"]),
    srcs_version = "PY2AND3",
    data = [":python/ops/_edit_distance_ops.so"],
    deps = [":edit_distance_ops",],
)

cuda_py_tests(
    name = "edit_distance_tests",
    size = "small",
    srcs = glob(["python/kernel_tests/*.py"]),
    additional_deps = [
        ":edit_distance_py",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:platform_test",
    ],
)

