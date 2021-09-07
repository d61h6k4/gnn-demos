load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@gnn_demos_deps//:requirements.bzl", "requirement")

py_library(
    name = "shortest_path",
    srcs = ["shortest_path.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("numpy"),
        requirement("networkx"),
        requirement("scipy"),
        requirement("graph_nets"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "shortest_path_test",
    srcs = ["shortest_path_test.py"],
    deps = [":shortest_path"],
)
