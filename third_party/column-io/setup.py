from setuptools import setup, find_packages
import subprocess
import os,time,json

COLUMN_IO_VERSION = ["v0.2.9"]

def get_compile_cmd():  # type: () -> str
    try:
        cmd = subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "cmd="], stderr=subprocess.STDOUT,
        ).strip().decode("utf-8")
        return cmd
    except subprocess.CalledProcessError as e:
        print("Error:", e.output.decode("utf-8"))
        return ""
def get_compile_env():  # type: () -> dict[str,str]
    envset = set({"PATH", "LD", "FLAG", "NEED", "TF", "TORCH"}) 
    env_dict = {}
    for k, v in os.environ.items():
        # if  k not like %envset[i]%  continue
        if not any(x in k for x in envset):
            continue
        env_dict[k] = v
    return env_dict

def is_internal_enabled():
    return int(os.environ.get("INTERNAL_VERSION", 0))


def cmake():
    is_internal = "ON" if is_internal_enabled() else "OFF"
    build_dir = "build"
    os.makedirs(build_dir, exist_ok=True)
    if subprocess.check_call(["cmake", "..", "-DINTERNAL_VERSION={}".format(is_internal)], cwd=build_dir) != 0:
        raise RuntimeError("run cmake failed")
    if subprocess.check_call(["make", "-j"], cwd=build_dir) != 0:
        raise RuntimeError("run make failed")

def get_version():
    # Doc: PEP 440 Wheel Version Identification and Dependency Specification https://peps.python.org/pep-0440/
    version = COLUMN_IO_VERSION
    if str(os.environ.get("NEED_ODPS_COLUMN", "0")) == "0" :
        version.append("abi0")
    else:
        version.append("abi1")
    version = version[0] + "+" + ".".join(version[1:])
    return version

version = get_version()
print(f"version {version}")
cmake()
setup(
    name="column_io",
    version=version,
    packages=find_packages(),
    include_package_data=True,
)
