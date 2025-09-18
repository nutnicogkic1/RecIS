import os
import re

import torch


def get_package_version():
    pwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(pwd, "recis", "__init__.py")) as f:
        groups = re.findall(r"__version__.*([0-9]+)\.([0-9]+)\.([0-9]+)", f.read())
        main_version, minor_version, patch_version = groups[0]
        print(f"RecIS version {main_version}.{minor_version}.{patch_version}")
        return main_version, minor_version, patch_version


def get_cuda_version():
    return torch.version.cuda


def get_version():
    version = get_package_version()
    torch_version = "torch" + torch.__version__.replace(".", "").replace("+", "")
    cuda_version = "cuda" + torch.version.cuda.replace(".", "")
    version = ".".join(version) + "+" + cuda_version + torch_version
    return version


if __name__ == "__main__":
    print(get_version())
