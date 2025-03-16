# SM-DirichletFusion

基于Dirichlet扩散模型和Sharma-Mittal散度的多视图不确定性建模与融合框架。

## 项目概述

SM-DirichletFusion是一个端到端的不完全多视图数据处理框架，结合了Dirichlet扩散模型、Sharma-Mittal散度正则化、UPCE多标签损失以及DST+Kalman动态融合机制，用于解决多视图数据中的不确定性建模与融合问题。

### 核心特性

- **Dirichlet扩散模型**：在单纯形空间上构建前向加噪与反向去噪过程，增强生成能力
- **Sharma-Mittal散度正则化**：替代传统KL散度，提升不确定性控制能力
- **UPCE多标签损失**：直接优化Dirichlet分布的期望负对数似然，适配多标签场景
- **DST+Kalman动态融合**：基于Dempster-Shafer理论合并多视图证据，结合Kalman滤波实现时序平滑

## 安装

```bash
git clone https://github.com/username/SM-DirichletFusion.git
cd SM-DirichletFusion
pip install -e .
```

## 支持的数据集

SM-DirichletFusion支持多种数据集类型：

### 图像数据集

- **MNIST**：手写数字识别数据集，通过不同变换创建多视图
- **CIFAR-10**：彩色图像分类数据集，通过不同变换创建多视图

### 表格数据集 (UCI)

- **Wine**：葡萄酒化学成分数据集，特征分为3个视图
- **Iris**：鸢尾花数据集，特征分为2个视图（萼片和花瓣特征）
- **Breast Cancer**：乳腺癌诊断数据集，特征分为3个视图

### 医学影像数据集

- **多模态医学影像**：支持CT、MRI、X光等多种模态的医学影像数据
- **虚拟医学数据集生成**：提供创建虚拟医学数据集的功能，用于测试和开发

## 使用示例

### MNIST示例

```bash
python examples/mnist_example.py --num-views 2 --epochs 10 --use-sm --use-kalman
```

### CIFAR-10示例

```bash
python examples/cifar10_example.py --num-views 3 --epochs 20 --use-sm --use-kalman
```

### UCI数据集示例

```bash
python examples/uci_example.py --dataset wine --num-timesteps 50 --use-sm --use-kalman
```

### 医学数据集示例

```bash
# 创建虚拟医学数据集并训练
python examples/medical_example.py --create-dummy --modalities ct mri xray --epochs 30 --use-sm --use-kalman
```

## 代码示例

```python
from sm_dirichlet_fusion.models import DirichletNet
from sm_dirichlet_fusion.diffusion import DirichletDiffusion
from sm_dirichlet_fusion.losses import UPCELoss, SMRegularizer
from sm_dirichlet_fusion.fusion import DSTFusion
from sm_dirichlet_fusion.datasets import MultiViewMNIST
from sm_dirichlet_fusion.utils import Logger

# 创建数据集
dataset = MultiViewMNIST(
    root='./data',
    train=True,
    num_views=2,
    view_missing_prob=0.2,
    download=True
)

# 创建模型
model = DirichletNet(
    input_dim=784,
    num_classes=10,
    hidden_dims=[256, 512, 256]
)

# 创建扩散过程
diffusion = DirichletDiffusion(
    model=model,
    num_classes=10,
    num_timesteps=100
)

# 设置损失函数
loss_fn = UPCELoss()
regularizer = SMRegularizer(r=0.8, t=1.2)

# 创建融合模块
fusion = DSTFusion(num_classes=10, use_kalman=True)

# 创建日志记录器
logger = Logger(log_dir='./logs', experiment_name='my_experiment')
```

## 评估指标

SM-DirichletFusion提供了丰富的评估指标，用于全面评估模型性能：

- **分类指标**：准确率、精确率、召回率、F1分数、AUC
- **不确定性指标**：预测熵、Dirichlet熵、互信息、不确定性AUROC
- **校准指标**：期望校准误差(ECE)、最大校准误差(MCE)、Brier分数

## 可视化功能

框架提供了多种可视化功能，帮助理解模型行为：

- **不确定性分布**：可视化预测的不确定性分布
- **校准曲线**：评估模型校准性能
- **混淆矩阵**：分析分类错误模式
- **TensorBoard集成**：实时监控训练过程

## 项目结构

```
SM-DirichletFusion/
├── examples/           # 使用示例
│   ├── mnist_example.py
│   ├── cifar10_example.py
│   ├── uci_example.py
│   └── medical_example.py
├── src/                # 源代码
│   ├── datasets/       # 数据集加载
│   ├── diffusion/      # 扩散模型实现
│   ├── fusion/         # 多视图融合方法
│   ├── losses/         # 损失函数实现
│   ├── models/         # 网络模型定义
│   └── utils/          # 工具函数和评估指标
└── tests/              # 单元测试
```

## 引用

如果您使用了本项目，请引用：

```
@article{sm-dirichlet-fusion,
  title={SM-DirichletFusion: Uncertainty Modeling and Fusion for Incomplete Multi-view Data},
  author={Author},
  journal={arXiv preprint},
  year={2023}
}
```

## 许可证

MIT # SM-DirichletFusion
