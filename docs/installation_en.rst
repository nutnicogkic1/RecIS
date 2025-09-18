Installation Guide
=================

System Requirements
-------------------

**Python Version**
- Python 3.10+

**Dependencies**
- PyTorch 2.4+
- CUDA 12.4 

Installation Methods
--------------------

Docker Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide build scripts for three versions of images: pytorch240/251/260. You can build the base image using commands, for example:

Clone the repository::

    git clone https://github.com/alibaba/RecIS.git
    cd recis
    git submodule update --init --recursive

Build Docker image for pytorch240 version::

    docker build --network=host -f docker/Dockerfile.torch240 -t recis:torch240 .

Start the torch240 version image::

    docker run --runtime=nvidia --net=host -it --cpuset-cpus="0-63" -m 300G recis:torch240 /bin/bash

Source Installation
~~~~~~~~~~~~~~~~~~~

1. Clone the repository::

    git clone https://github.com/alibaba/RecIS.git
    cd recis
    git submodule update --init --recursive

2. Build and install RecIS::

    bash build.sh 0
    pip install `find ./dist -name "recis*.whl" -maxdepth 1`

3. Build and install column-io::

    cd third_party/column-io/
    bash tools/build_and_install.sh 0

4. Verify installation::

    python -c "import recis; print('RecIS installed successfully!')"

Installation Verification
--------------------------

Run the following code to verify successful installation::
    
    import torch
    import recis
    
    # Check versions
    print(f"PyTorch version: {torch.__version__}")
    
    # Check GPU support
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    
    # Simple functionality test
    from recis.nn.modules.embedding import DynamicEmbedding, EmbeddingOption
    
    emb_opt = EmbeddingOption(embedding_dim=16)
    emb = DynamicEmbedding(emb_opt)
    print("RecIS core modules loaded successfully!")

If the above code runs without errors, RecIS has been successfully installed.
