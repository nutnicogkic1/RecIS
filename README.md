<h1 align="center"> RecIS (Recommendation Intelligence System) </h1>

RecIS：一个专为超大规模稀疏+稠密计算设计的统一架构深度学习框架。基于 PyTorch 开源生态构建，为推荐模型训练，或推荐结合多模态/大模型训练提供了完整的解决方案。由阿里控股集团爱橙科技智能引擎事业部和淘天广告技术、淘天算法技术团队联合推出。目前已在阿里巴巴广告、推荐、搜索等场景广泛应用。

<p align="center">
    <img alt="Static Badge" src="https://img.shields.io/badge/made_by-XDL_Team-blue">
    <img alt="Static Badge" src="https://img.shields.io/badge/version-v1.0.0-green">
    <img alt="Static Badge" src="https://img.shields.io/badge/license-Apache--2.0-blue">
</p>

## 🎯 设计目标

**统一框架**
- 基于 PyTorch 开源生态，统一稀疏-稠密框架需求
- 满足工业级推荐模型结合多模态、大模型场景的训练需求

**性能优化**
- 优化稀疏相关算子访存性能
- 提供稀疏算子合并优化能力，充分发挥GPU算力
- 达到甚至超过基于 Tensorflow 性能

**易用性**
- 灵活的特征、Embedding配置组织方式
- 自动化特征处理及优化流程
- 简单的稀疏模型定义方式

## 🏗️ 核心架构

RecIS 采用模块化设计，主要包含以下核心组件：

<div align="center">

<img src="docs/_static/sys-recis.png" width="100%" alt="System Architecture">

</div>

- **ColumnIO**: 数据读取
  - 支持分布式分片读取数据
  - 在读取阶段完成简单的特征预计算
  - 样本组装为Torch Tensor，并提供数据预取功能
  
- **Feature Engine**: 特征处理
  - 提供特征工程和特征转换处理能力，包括 Hash / Mod / Bucketize 等
  - 支持自动的算子融合优化策略
  
- **Embedding Engine**: Embedding 管理与计算
  - 提供无冲突、可拓展的 KV 存储 Embedding 表
  - 提供多张表的合并优化能力，以获取更好的访存性能
  - 支持特征淘汰和准入策略
  
- **Saver**: 参数保存与加载
  - 提供稀疏参数以 SafeTensors 标准格式存储交付能力

- **Pipelines**: 训练流程编排
  - 将上面几个组件串联，封装训练流程
  - 支持多阶段（训练/测试交错）、多目标计算等复杂训练流程

## 🚀 关键优化

### 高效动态 Embedding

RecIS 框架通过一种两级存储架构实现了高效的动态嵌入（HashTable）：

- **IDMap**: 作为一级存储，以特征 ID 作为键，以 Offset 作为 Value
- **EmbeddingBlocks**: 
  - 作为二级存储，连续分片内存快，用于存储嵌入参数以及优化器状态。
  - 支持动态分片，可灵活拓展
- **灵活硬件适配策略**: 同时支持 GPU 和 CPU 存放 IDMap 和 EmbeddingBlocks

### 分布式优化

- **参数聚合与分片**: 
  - 在模型创建阶段，将相同属性（维度、初始化器等）的参数表合并成一个逻辑表
  - 参数均匀切分到各个计算节点
- **请求合并与切分**: 
  - 前向计算时，合并属性相同参数表请求，并对其去重计算分片信息
  - 通过集合通信 All-to-All 获取各个计算节点上的 Embedding 向量

### 高效利用硬件计算资源

- **GPU 并发优化**: 
  - 支持特征处理算子融合优化，大幅减少算子数量，减小 Launch 开销
  
- **参数表合并优化**: 
  - 支持属性相同参数表合并，减少特征查找次数，大幅减少算子数量，提升内存空间利用效率

- **算子实现优化**: 
  - 算子实现向量化访存，提高访存利用率
  - 优化 Reduction 算子，通过 Warp 级别合并，减少原子操作，提升访存利用率


## 📚 文档

- [安装指南](https://alibaba.github.io/RecIS/installation.html)
- [快速开始](https://alibaba.github.io/RecIS/quickstart.html)
- [项目介绍](https://alibaba.github.io/RecIS/introduction.html)
- [常见问题](https://alibaba.github.io/RecIS/faq.html)

## 🤝 支持与反馈

如果遇到问题，可以：

- 查看项目 [Issues](https://github.com/alibaba/RecIS/issues)

## 📄 许可证

本项目基于 [Apache 2.0](LICENSE) 许可证开源。
