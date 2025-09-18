示例教程
========

本章节提供了 RecIS 的详细使用示例，涵盖从基础到高级的各种应用场景。

.. toctree::
   :maxdepth: 2

   basic_usage
   deepfm_example
   seq2seq_example
   ctr_example

示例概览
--------

.. list-table:: 示例列表
   :header-rows: 1
   :widths: 30 50

   * - 示例名称
     - 描述
   * - :doc:`basic_usage`
     - RecIS 基础使用方法
   * - :doc:`deepfm_example`
     - 完整的 DeepFM 模型实现
   * - :doc:`seq2seq_example`
     - 完整的 Seq2Seq 模型实现
   * - :doc:`ctr_example`
     - 示例CTR模型，了解更多特征处理模式

学习路径建议
------------

**初学者路径**

1. 阅读 :doc:`../introduction` 了解 RecIS 基本概念
2. 跟随 :doc:`../quickstart` 完成第一个模型
3. 学习 :doc:`basic_usage` 掌握基础用法
4. 实践 :doc:`deepfm_example` 理解完整流程

**进阶用户路径**

1. 实践 :doc:`seq2seq_example` 理解其他训练范式
2. 实践 :doc:`ctr_example` 理解更复杂数据处理流程
3. 参考 API 文档进行自定义开发

**高级用户路径**

1. 研究源码实现原理
2. 贡献代码和功能
3. 优化性能和扩展功能
4. 分享最佳实践

常见应用场景
------------

**推荐系统**

- 用户-商品推荐
- 内容推荐
- 协同过滤
- 深度学习推荐模型

**广告系统**

- CTR 预估
- CVR 预估
- 出价优化
- 受众定向

**搜索排序**

- 搜索结果排序
- 查询理解
- 相关性计算
- 个性化搜索

**风控系统**

- 欺诈检测
- 风险评估
- 异常检测
- 信用评分

获取帮助
--------

如果在使用示例过程中遇到问题：

1. 查看 :doc:`../faq` 常见问题解答
2. 参考详细的 :doc:`../api/index` API 文档
3. 在 `GitHub Issues <https://github.com/alibaba/RecIS/issues>`_ 提问
4. 加入技术交流群获取支持

贡献示例
--------

欢迎贡献新的示例和教程：

1. Fork 项目仓库
2. 创建新的示例文件
3. 添加详细的文档说明
4. 提交 Pull Request

**示例贡献指南**

- 代码风格遵循项目规范
- 提供完整的运行说明
- 包含必要的注释和文档
- 验证代码的正确性和可运行性
