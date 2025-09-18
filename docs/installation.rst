安装指南
========

系统要求
--------

**Python 版本**
- Python 3.10+

**依赖要求**
- PyTorch 2.4+
- CUDA 12.4 

安装方式
--------

Docker 安装(推荐) 
~~~~~~~~~~~~~~~~~~

我们提供了 pytorch240/251/260 三个版本的镜像构建脚本，您可以通过命令构建基础镜像，例如

克隆代码仓库::

    git clone https://github.com/alibaba/RecIS.git
    cd recis
    git submodule update --init --recursive

构建pytorch240版本的 Docker 镜像::

    docker build --network=host -f docker/Dockerfile.torch240 -t recis:torch240 .

启动torch240版本镜像::

    docker run --runtime=nvidia --net=host -it --cpuset-cpus="0-63" -m 300G recis:torch240 /bin/bash

源码安装
~~~~~~~~

1. 克隆代码仓库::

    git clone https://github.com/alibaba/RecIS.git
    cd recis
    git submodule update --init --recursive

2. 构建安装RecIS::

    bash build.sh 0
    pip install `find ./dist -name "recis*.whl" -maxdepth 1`

3. 构建安装column-io::

    cd third_party/column-io/
    bash tools/build_and_install.sh 0

4. 验证安装::

    python -c "import recis; print('RecIS installed successfully!')"

验证安装
--------

运行以下代码验证安装是否成功::
    
    import torch
    import recis
    
    # 检查版本
    print(f"PyTorch version: {torch.__version__}")
    
    # 检查 GPU 支持
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    
    # 简单功能测试
    from recis.nn.modules.embedding import DynamicEmbedding, EmbeddingOption
    
    emb_opt = EmbeddingOption(embedding_dim=16)
    emb = DynamicEmbedding(emb_opt)
    print("RecIS 核心模块加载成功!")

如果以上代码运行无误，说明 RecIS 已成功安装。
