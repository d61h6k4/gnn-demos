workspace(name = "gnn_demos")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_python",
    sha256 = "934c9ceb552e84577b0faf1e5a2f0450314985b4d8712b2b70717dc679fdc01b",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.3.0/rules_python-0.3.0.tar.gz",
)

load("@rules_python//python:pip.bzl", "pip_parse")

# Create a central repo that knows about the dependencies needed from
# requirements_lock.txt.
pip_parse(
    name = "gnn_demos_deps",
    requirements_lock = "//:requirements_lock.txt",
)

# Load the starlark macro which will define your dependencies.
load("@gnn_demos_deps//:requirements.bzl", "install_deps")

# Call it to define repos for your requirements.
install_deps()
