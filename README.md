# Post-Training（大语言模型后训练框架）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/lcdmiku/Post-Training/pulls)

一个基于MiniMind围绕大语言模型（LLM）后训练（Post-Training）的个人学习与实践项目，涵盖监督微调（SFT）与基于 PPO 的强化学习（RLHF）流程。项目采用模块化设计，将数据、模型、训练器分离，便于理解后训练各阶段的实现细节与数据流。

## 功能特性

- **监督微调（SFT）**：在指令数据集上对基座模型进行指令微调。
- **PPO 强化学习**：基于奖励模型，使用 PPO 算法进行偏好对齐。
- **模块化架构**：`dataset`、`model`、`trainer` 三层分离，职责清晰。
- **模型推理接口**：提供 `eval_llm.py`，用于加载训练后的模型进行生成与评估。
- **训练日志**：记录损失、奖励、KL 散度等关键指标，便于监控训练状态。

## 项目结构

```
Post-Training/
├── dataset/                    # 数据加载与预处理
│   ├── __init__.py
│   ├── sft_dataset.py          # SFT 数据集类
│   └── ppo_dataset.py          # PPO 数据集类
├── model/                      # 模型定义与加载
│   ├── __init__.py
│   ├── actor.py                # Actor 策略网络
│   ├── critic.py               # Critic 价值网络
│   └── reward_model.py         # 奖励模型
├── trainer/                    # 训练流程
│   ├── __init__.py
│   ├── sft_trainer.py          # SFT 训练器
│   └── ppo_trainer.py          # PPO 训练器
├── eval_llm.py                 # 模型推理与评估入口
├── requirements.txt            # Python 依赖
├── .gitignore
└── README.md
```

> 以上结构基于仓库当前可见的顶层文件与目录整理，具体文件以实际代码为准。

## 环境要求

- **Python** 3.9 及以上
- **PyTorch** 2.0+
- **Transformers** 4.30+
- **TRL** 或 **OpenRLHF**（用于 PPO 训练）
- 支持 CUDA 的 GPU（建议显存 ≥ 24GB，用于 7B 级模型训练）

安装依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/lcdmiku/Post-Training.git
cd Post-Training
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备数据

将 SFT 数据（指令-回答对）和偏好数据（chosen/rejected 对）放入 `dataset/` 目录下，或修改数据加载路径指向你的数据集。

### 4. 监督微调（SFT）

```bash
python -m trainer.sft_trainer \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --dataset_path dataset/sft_data.jsonl \
    --output_dir outputs/sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 2e-5
```

### 5. 奖励模型训练（可选）

若使用 RLHF，需要先训练奖励模型，或直接使用已有的奖励模型。

### 6. PPO 强化学习

```bash
python -m trainer.ppo_trainer \
    --sft_model_path outputs/sft \
    --reward_model_path outputs/reward_model \
    --output_dir outputs/ppo \
    --ppo_epochs 4 \
    --kl_coef 0.05 \
    --learning_rate 1e-6
```

### 7. 模型推理

```bash
python eval_llm.py --model_path outputs/ppo --prompt "请介绍一下你自己"
```

## 核心模块说明

### `dataset/` — 数据层

负责加载和预处理训练数据：

- **SFT 数据集**：将指令-回答对转换为模型输入格式。
- **PPO 数据集**：加载 prompts 和偏好数据，供 PPO 训练采样使用。

### `model/` — 模型层

定义后训练所需的各个模型组件：

- **Actor**：策略网络，即被训练的语言模型。
- **Critic**：价值网络，估计状态价值，用于计算优势函数。
- **Reward Model**：奖励模型，对生成结果打分。

### `trainer/` — 训练层

实现训练循环与优化逻辑：

- **SFT Trainer**：标准的监督微调训练流程。
- **PPO Trainer**：包含 rollout 采样、优势估计（GAE）、策略更新等完整 PPO 流程。

## 训练配置参考

### SFT 典型超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 学习率 | 1e-5 ~ 2e-5 | 全量微调时使用较小学习率 |
| 有效 batch size | 16 ~ 64 | 根据显存调整 |
| 训练轮数 | 2 ~ 3 | 通常 1-2 轮即可 |
| 最大序列长度 | 1024 ~ 2048 | 根据任务复杂度调整 |
| 学习率调度 | cosine | 带 3% warmup |

### PPO 典型超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 学习率 | 1e-6 ~ 5e-7 | PPO 阶段学习率通常比 SFT 更小 |
| KL 惩罚系数 β | 0.01 ~ 0.1 | 控制偏离参考模型的程度，常用 0.05 |
| clip 范围 ϵ | 0.1 ~ 0.2 | PPO 裁剪参数 |
| GAE 系数 λ | 0.95 | 广义优势估计的偏差-方差权衡 |
| 折扣因子 γ | 0.99 | 未来奖励的折扣 |
| PPO epochs | 4 ~ 5 | 每批数据复用轮数 |
| rollout 长度 | 1024 ~ 4096 | 每次采样的 token 数 |

## 常见问题

**Q：训练显存不足怎么办？**

- 减小 `per_device_train_batch_size`，配合 `gradient_accumulation_steps` 保持有效 batch size。
- 使用 LoRA 或 QLoRA 进行参数高效微调。
- 启用 gradient checkpointing。

**Q：PPO 训练不稳定，奖励先升后降？**

- 增大 KL 惩罚系数 β，约束策略不要偏离参考模型太远。
- 减小学习率。
- 检查奖励模型的输出尺度是否合理。

**Q：如何评估后训练效果？**

- 使用 `eval_llm.py` 加载模型进行生成，人工检查回答质量。
- 在标准 benchmark（如 MMLU、GSM8K）上评估。
- 对比 SFT 前后、PPO 前后的输出差异。

## 后续计划

- [ ] 支持 DPO（Direct Preference Optimization）
- [ ] 支持 GRPO（Group Relative Policy Optimization）
- [ ] 增加 LoRA / QLoRA 参数高效微调
- [ ] 增加分布式训练支持
- [ ] 完善评估脚本与 benchmark 集成

## 许可证

本项目采用 [MIT License](LICENSE) 许可证，可自由使用、修改和分发，只需保留原始版权声明。

Copyright (c) 2026 lcdmiku

## 作者

- **lcdmiku** — [GitHub 主页](https://github.com/lcdmiku)
