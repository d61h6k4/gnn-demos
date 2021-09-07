load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")
load("@gnn_demos_deps//:requirements.bzl", "requirement")

py_library(
    name = "batching_utils",
    srcs = ["batching_utils.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("jax"),
        requirement("jraph"),
    ],
)

py_library(
    name = "input_pipeline",
    srcs = ["input_pipeline.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":batching_utils",
        requirement("jax"),
        requirement("tensorflow"),
        requirement("graph_nets"),
    ],
)

py_library(
    name = "train",
    srcs = ["train.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":input_pipeline",
        "//datasets/graphs:shortest_path",
        requirement("tensorflow"),
        requirement("jax"),
        requirement("flax"),
        requirement("jraph"),
        requirement("ml_collections"),
        requirement("tensorflow_datasets"),
    ],
)

py_binary(
    name = "main",
    srcs = ["main.py"],
    deps = [
        ":train",
        requirement("absl_py"),
        requirement("tensorflow"),
        requirement("clu"),
        requirement("jax"),
        requirement("ml_collections"),
    ],
)

py_binary(
    name = "eval",
    srcs = ["eval.py"],
    deps = [
        ":input_pipeline",
        ":train",
        "//visualization/graphs:shortest_path",
        requirement("absl_py"),
        requirement("tensorflow"),
        requirement("clu"),
        requirement("jax"),
        requirement("ml_collections"),
    ],
)

py_test(
    name = "input_pipeline_test",
    srcs = ["input_pipeline_test.py"],
    deps = [
        ":input_pipeline",
        "//datasets/graphs:shortest_path",
        requirement("absl_py"),
        requirement("tensorflow_datasets"),
    ],
)

py_test(
    name = "train_test",
    srcs = ["train_test.py"],
    deps = [
        ":input_pipeline",
        ":train",
        "//datasets/graphs:shortest_path",
        requirement("tensorflow_datasets"),
    ],
)
