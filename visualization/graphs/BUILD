load("@rules_python//python:defs.bzl", "py_library")
load("@gnn_demos_deps//:requirements.bzl", "requirement")

py_library(
    name = "shortest_path",
    srcs = ["shortest_path.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//datasets/graphs:shortest_path",
        requirement("numpy"),
        requirement("matplotlib"),
        requirement("networkx"),
    ],
)
