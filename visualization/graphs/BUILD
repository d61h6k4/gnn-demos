load("@rules_python//python:defs.bzl", "py_binary")
load("@gnn_demos_deps//:requirements.bzl", "requirement")

py_binary(
    name = "shortest_path",
    srcs = ["shortest_path.py"],
    deps = [
        "//datasets/graphs:shortest_path",
        requirement("numpy"),
        requirement("matplotlib"),
        requirement("networkx"),
    ],
)
