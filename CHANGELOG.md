# [1.0.0] - 2025-09-11

## 🎉 Initial Release

RecIS (Recommendation Intelligence System) v1.0.0 is now officially released! This is a unified architecture deep learning framework designed specifically for ultra-large-scale sparse models, built on the PyTorch open-source ecosystem. It has been widely used in Alibaba advertising, recommendation, searching and other scenarios.

## ✨ New Features

## Core Architecture

- **ColumnIO**: Data Reading
  - Supports distributed sharded data reading
  - Completes simple feature pre-computation during the reading phase
  - Assembles samples into Torch Tensors and provides data prefetching functionality
  
- **Feature Engine**: Feature Processing
  - Provides feature engineering and feature transformation processing capabilities, including Hash / Mod / Bucketize, etc.
  - Supports automatic operator fusion optimization strategies
  
- **Embedding Engine**: Embedding Management and Computing
  - Provides conflict-free, scalable KV storage embedding tables
  - Provides multi-table fusion optimization capabilities for better memory access performance
  - Supports feature elimination and admission strategies
  
- **Saver**: Parameter Saving and Loading
  - Provides sparse parameter storage and delivery capabilities in SafeTensors standard format

- **Pipelines**: Training Process Orchestration
  - Connects the above components and encapsulates training processes
  - Supports complex training workflows such as multi-stage (training/testing interleaved) and multi-objective computation

## 🛠️ Installation & Compatibility

## System Requirements
- **Python**: 3.10+
- **PyTorch**: 2.4+
- **CUDA**: 12.4

## Installation Methods
- **Docker Installation**: Pre-built Docker images for PyTorch 2.4.0/2.5.1/2.6.0
- **Source Installation**: Complete build system with CMake and setuptools

## Dependencies
- `torch>=2.4`
- `accelerate==0.29.2`
- `simple-parsing`
- `pyarrow` (for ORC support)

## 📚 Documentation

- Complete English and Chinese documentation
- Quick start tutorials with CTR model examples
- Comprehensive API reference
- Installation guides for different environments
- FAQ and troubleshooting guides

## 📦 Package Structure

- **Core Library**: `recis/` - Main framework code
- **C++ Extensions**: `csrc/` - High-performance C++ implementations
- **Documentation**: `docs/` - Comprehensive documentation in RST format
- **Examples**: `examples/` - Practical usage examples
- **Tools**: `tools/` - Data conversion and utility tools
- **Tests**: `tests/` - Comprehensive test suite

## 🚀 Key Optimizations

## Efficient Dynamic Embedding

The RecIS framework implements efficient dynamic embedding (HashTable) through a two-level storage architecture:

- **IDMap**: Serves as first-level storage, using feature ID as key and Offset as value
- **EmbeddingBlocks**: 
  - Serves as second-level storage, continuous sharded memory blocks for storing embedding parameters and optimizer states
  - Supports dynamic sharding with flexible expansion capabilities
- **Flexible Hardware Adaptation Strategy**: Supports both GPU and CPU placement for IDMap and EmbeddingBlocks

## Distributed Optimization

- **Parameter Aggregation and Sharding**: 
  - During model creation phase, merges parameter tables with identical properties (dimensions, initializers, etc.) into one logical table
  - Parameters are evenly distributed across compute nodes
- **Request Merging and Splitting**: 
  - During forward computation, merges requests for parameter tables with identical properties and computes sharding information with deduplication
  - Obtains embedding vectors from various compute nodes through All-to-All collective communication

## Efficient Hardware Resource Utilization

- **GPU Concurrency Optimization**: 
  - Supports feature processing operator fusion optimization, significantly reducing operator count and launch overhead
  
- **Parameter Table Fusion Optimization**: 
  - Supports merging parameter tables with identical properties, reducing feature lookup frequency, significantly decreasing operator count, and improving memory space utilization efficiency

- **Operator Implementation Optimization**: 
  - Operator implementations use vectorized memory access to improve memory utilization
  - Optimizes reduction operators through warp-level merging, reducing atomic operations and improving memory access utilization

## 🤝 Community & Support

- Open source under Apache 2.0 license
- Issue tracking and community support
- Active development by XDL Team

---

For detailed usage instructions, please refer to our [documentation](https://alibaba.github.io/RecIS/) and [quick start guide](https://alibaba.github.io/RecIS/quickstart.html).