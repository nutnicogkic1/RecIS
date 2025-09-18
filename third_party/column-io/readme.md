# ColumnIo - A python lib for building pipeline to read columnar data.

## 构建镜像
```
docker build --network=host --no-cache --cpuset-cpus=0-95 -t ${image_name}
```

## 下载编译依赖
```
# 推荐在容器内安装依赖并编译columnIO
yum install -y t_search_kmonitor_client alog-devel autil-devel -b current
pip install cmake
```


## 编译及安装

`bash -x ./build_and_install.sh`

编译选项:
- `export NEED_ODPS_COLUMN=1` default:0  使用ODPS storage接口(同时开启cxx11_abi1并禁用abi0模块(如直读))

## 测试
```
bash -x ./test.sh
```
