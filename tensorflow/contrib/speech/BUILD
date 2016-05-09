package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "cuda_py_tests")

load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library")
load("//tensorflow/core:platform/default/build_config.bzl", "tf_proto_library_cc")

tf_proto_library(
    name = "speech_protos",
    srcs = glob(
        ["**/*.proto"],
    ),
    cc_api_version = 2,
    py_api_version = 0,
    visibility = ["//visibility:public"],
)

tf_gen_op_libs(
    op_lib_names = ["speech_ops"],
)

tf_gen_op_wrapper_py(
    name = "speech_ops",
    deps = [":speech_ops_op_lib"],
    require_shape_functions = True,
)

tf_custom_op_library(
    name = "python/ops/_speech_ops.so",
    srcs = [
        "ops/speech_ops.cc",
        "kernels/speech_ops.cc",
    ],
    deps = [
        ":speech_protos_cc",
    ],
)

py_library(
    name = "token_model",
    srcs = ["data/token_model.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":speech_protos_py",
        "//google/protobuf:protobuf_python",
    ],
)

py_binary(
    name = "create_token_model",
    srcs = ["data/create_token_model.py"],
    main = "data/create_token_model.py",
    srcs_version = "PY2AND3",
    deps = [
        ":token_model",
        "//tensorflow:tensorflow_py",
        "//tensorflow/python:platform",
    ]
)
