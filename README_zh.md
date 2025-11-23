# AgentGym-RL：通过多轮强化学习训练用于长时序决策的LLM智能体
<p align="center">
  📃 <a href="https://arxiv.org/abs/2509.08755" target="_blank">论文</a > • 🌐 <a href="https://agentgym-rl.github.io/" target="_blank">项目主页</a > • 🤗 <a href="https://huggingface.co/datasets/AgentGym/AgentGym-RL-Data-ID" target="_blank">AgentGym-RL-Data-ID</a >
</p >

AgentGym-RL 是一个用于通过强化学习（RL）训练具备**多轮**交互式决策能力的 LLM 智能体的新框架。它涵盖了种类广泛的**真实世界场景**，并支持主流 RL 算法。大量实验表明，我们的框架与方法显著提升了开源 7B 规模模型的能力，在跨越多样环境的**27 个任务**上，实现了**匹敌甚至超越商业模型**的表现。

![](./assets/main_performance.jpg)

## 🔔 最新动态

- **🎉[2025-09-10]** 你可以为 AgentGym 开发自定义环境并进行 RL 训练！教程见[这里](https://github.com/WooooDyy/AgentGym/blob/main/docs/tutorials/en/05-2nd-Development.md)。
- **🥳[2025-09-10]** 我们的论文已在 arXiv 发布：[AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2509.08755)
- **🍺[2025-09-10]** 我们的 RL 数据集与基准已在 Hugging Face 开放：[AgentGym-RL-Data-ID](https://huggingface.co/datasets/AgentGym/AgentGym-RL-Data-ID)

## 🌟 概述

面向复杂真实世界任务，培养能够做出一系列智能决策的自主 LLM 智能体，是快速发展的前沿方向。仅依赖人类示范进行行为克隆可以让智能体“会做事”，但很难带来真正的突破。正如 Richard Sutton 强调的那样，真正推动智能体进步的是其通过探索并与环境交互而获得的知识、技能与经验。因此，一个极具前景的路径是使用强化学习来训练这类智能体。

多数现有研究仍局限于数学、代码等单轮任务。近期将 RL 扩展到训练具备多轮能力的 LLM 智能体的尝试，面临显著挑战：

- **任务复杂度与环境多样性受限。** 在强化学习时代，环境的重要性愈发凸显。只在玩具环境表现良好的智能体难以迁移到真实场景；而环境的多样性是泛化能力的前提。
- **稳定高效优化困难。** 多轮交互显著扩展了搜索空间并增加训练信号的方差，使得探索与利用之间的平衡更具挑战。

为应对这些挑战，我们提出了 **AgentGym-RL**——一个用于通过 RL 训练具备**多轮**交互式决策能力的 LLM 智能体的新框架。它涵盖多种**真实世界场景**并支持主流 RL 算法，为**经验时代**的研究与实践奠定基础。

![](./assets/AgentGym-RL-main.png)

此外，为缓解探索-利用权衡并提升智能体 RL 训练中的优化稳定性，我们提出 **ScalingInter-RL** 方法：在训练过程中逐步扩展智能体与环境的交互时域。跨环境实验显示，结合 AgentGym-RL 框架与 ScalingInter-RL 算法，可以获得稳定、持续且幅度可观的行为提升。

同时，为便于深入探查数据与模型行为，我们提供了一个**可视化的交互式用户界面**，支持对完整交互轨迹的回放与分析，加速迭代式的经验研究。

![](./assets/env.jpg)

## 📖 目录

- [AgentGym-RL：通过多轮强化学习训练用于长时序决策的LLM智能体](#agentgym-rl通过多轮强化学习训练用于长时序决策的llm智能体)
  * [🔔 最新动态](#-最新动态)
  * [🌟 概述](#-概述)
  * [功能](#功能)
    + [AgentGym-RL 的模块化系统设计](#agentgym-rl-的模块化系统设计)
    + [环境](#环境)
    + [后训练策略](#后训练策略)
    + [ScalingInter-RL：面向智能体RL的渐进式交互扩展](#scalinginter-rl面向智能体rl的渐进式交互扩展)
    + [Verl 的扩展](#verl-的扩展)
  * [性能](#性能)
  * [运行教程](#运行教程)
    + [环境准备](#环境准备)
    + [训练](#训练)
    + [评测](#评测)
    + [可视化用户界面](#可视化用户界面)
  * [致谢](#致谢)
  * [联系](#联系)
  * [引用](#引用)

## 功能

### AgentGym-RL 的模块化系统设计

我们采用模块化、解耦的设计来实现 AgentGym-RL，将其组织为三个核心组件：

- **环境模块**：通过标准化的服务端-客户端架构、统一的 HTTP 协议与并行请求，提供多样化场景。
- **智能体模块**：封装智能体在多轮交互中的推理与决策过程，支持长时序规划、自我反思等高级机制。
- **训练模块**：实现强化学习流水线与其他训练方法，用于优化智能体策略。

![](./assets/pseudo.jpg)

### 环境

* **网页导航（Web Navigation）**：包含 **WebArena**，一个真实且可复现的网页环境，涵盖互联网上常见的 4 大域：在线购物、讨论论坛、协作开发与业务内容管理。
* **深度搜索（Deep Search）**：基于 **Search-R1** 扩展，提供 RAG 型环境，使 LLM 能与搜索引擎交互，完成多轮检索与推理任务。
* **数字游戏（Digital Games）**：包含 **TextCraft**，一款文本式合成游戏，智能体通过自然语言交互与任务规划完成目标。
* **具身任务（Embodied Tasks）**：包含 **BabyAI**，在可控的栅格世界中通过文本指令进行具身推理的仿真环境。
* **科学任务（Scientific Tasks）**：包含 **SciWorld**，一个科学探索模拟器，智能体通过文本驱动的推理循环来进行科学实验。

### 后训练策略

AgentGym-RL 支持一系列主流在线 RL 算法：**PPO、GRPO、RLOO、REINFORCE++**。

除在线 RL 外，AgentGym-RL 也支持多种互补训练范式：**SFT、DPO、AgentEvol**。

### ScalingInter-RL：面向智能体RL的渐进式交互扩展

ScalingInter-RL 是一种在确保优化稳定性的同时平衡探索与利用的训练方法。其核心是**渐进式时域扩展策略**，在 RL 过程中自适应地调整交互轮次。

![](./assets/ScalingInter-RL-Method.png)

我们以较短的交互时域起步，使智能体能够高效利用已有策略，在简单任务上快速获得早期能力，为更深层的长时序推理奠定基础。随着训练推进，我们逐步延长时域，促使智能体探索更长的决策路径，推动更高阶认知行为的涌现。

### Verl 的扩展

为构建 AgentGym-RL，我们对 Verl 进行了如下改造：

1. **使用 vLLM 引擎的 Rollout**：为支持多轮 rollout 与高效环境交互，我们引入：
   * RolloutHandler 以管理轨迹。`RolloutHandler` 能为每轮中的环境观测与助手动作，正确计算 attention mask、loss mask、position id、sequence id，并处理历史消息、状态与奖励。
   * EnvClient 以管理交互。`EnvClient` 提供多种方法便于在 rollout 期间与环境交互，例如 `observarion()` 获取当前观测、`available_actions()` 获取当前可用动作、`step()` 执行动作、`reset()` 重置环境。为提升效率，框架会并行初始化环境并收集轨迹。
2. **优势函数计算**：我们修订了 Verl 在 REINFORCE++ 与 GAE 上的优势函数实现，以确保在单轮与多轮场景下的正确性。
3. **训练中的交互时域调度**：为实现 ScalingInter-RL，我们引入 `RoundScheduler` 在训练期间扩展交互时域。`FixedRoundsScheduler` 固定最大交互轮数；`StepRoundsScheduler` 以步进方式逐渐增加时域，实现训练中的渐进式扩展。

## 性能

我们以 Qwen2.5-3B 与 Qwen2.5-7B 作为主要骨干模型。在**五类场景**上评估 AgentGym-RL 与 ScalingInter-RL，并与多种闭源与开源模型进行对比。以下展示了在 WebArena 基准上的结果，其他基准的结果请见我们的[论文](https://arxiv.org/abs/2509.08755)。

![](./assets/webarena_performance.png)

- **ScalingInter-7B** 模型显著**超越顶级商用模型**（如 GPT-4o），并可与更大规模模型（如 DeepSeek-R1-0528、Gemini-2.5-Pro）**相媲美**。此外，在购物与 CMS 两个子任务上，取得的分数与所有模型中的最佳成绩相当。
- **AgentGym-RL-7B** 的总体得分**达到 GPT-4o 的水平**。

此外，如下图所示，ScalingInter-RL 在 RL 优化过程中展现出更加**稳定且高效**的训练动态。

![](./assets/searchqa_performance.jpg)

* 更长轮次的设置在初期通过更丰富的探索获得更高回报，但容易快速崩溃；更短轮次的设置更稳定但探索不足，导致性能上限受限。
* 我们的 ScalingInter-RL 方法会**逐步增加交互时域**，最终实现**更高且更高效**的长期表现。

## 运行教程

### 环境准备

我们推荐使用 CUDA 12.4、PyTorch 2.4 与 Python 3.10。首先通过以下命令安装依赖：
```sh
echo "Preparing environment for agentgym-rl..."
conda create -n agentgym-rl python==3.10 -y
conda activate agentgym-rl
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
# install flash-atten
FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
wget -q $FLASH_ATTENTION_URL -O $FLASH_ATTENTION_NAME
pip3 install $FLASH_ATTENTION_NAME
rm -f $FLASH_ATTENTION_NAME
# for RL
cd AgentGym-RL
pip3 install -e .
# for agentgym
echo "Preparing environment for agentenv..."
cd AgentGym/agentenv
pip3 install -e .
pip3 install transformers==4.51.3
```

### 训练

关于 SFT、DPO 与 AgentEvol，请参考 [AgentGym](https://github.com/WooooDyy/AgentGym/tree/640f8bca6901a6a6d540ff61522b813988da47c4/) 的 `README.md`。

RL 训练：

**1. 环境准备**

确保所需环境已搭建完毕（见上文[环境准备](#环境准备)）。

**2. 数据准备**

从 [Huggingface](https://huggingface.co/datasets/AgentGym/AgentGym-RL-Data-ID) 下载 AgentGym-RL-Data-ID 数据集。

**3. 启动环境服务端**

请参考 [AgentGym](https://github.com/WooooDyy/AgentGym/tree/640f8bca6901a6a6d540ff61522b813988da47c4) 的 `README.md` 启动环境服务端。

**4. 开始训练**

各任务的训练示例脚本位于 [examples/train](./examples/train)，涵盖 AgentGym-RL 与 ScalingInter-RL。你也可以参考脚本中配置的训练参数。

```sh
bash webarena_train.sh
```

更多参数说明可参见 [verl 文档](https://verl.readthedocs.io/en/latest/examples/config.html)。其他关键参数：
* `data.max_prompt_length`：首轮任务描述提示词的最大长度。
* `data.max_response_length`：交互轨迹的最大总 token 数（不含任务提示词）。
* `actor_rollout_ref.agentgym.task_name`：AgentGym 的训练任务名。
* `actor_rollout_ref.agentgym.env_addr`：AgentGym 环境服务端的 URL。
* `actor_rollout_ref.rollout.max_tokens`：单轮回复的最大 token 数。
* `actor_rollout_ref.rollout.rollout_log_dir`：存放 rollout 轨迹的目录。
* `algorithm.rounds_ctrl.type`：最大交互轮数的控制策略，选项：
  - `fixed`：固定轮数。
  - `scaling_inter_stepwise`：按固定步频增加轮数。
* `algorithm.rounds_ctrl.rounds`：允许的最大交互轮数。
* `algorithm.rounds_ctrl.steps_scaling_inter`：当使用 `scaling_inter_stepwise` 时，每多少训练步增加一次最大轮数。

更多细节见 [AgentGym-RL/verl/agent_trainer/config/ppo_trainer.yaml](./AgentGym-RL/verl/agent_trainer/config/ppo_trainer.yaml)。

启动 AgentGym-RL 训练，设置：

```sh
algorithm.rounds_ctrl.type=fixed \
algorithm.rounds_ctrl.rounds=15 \
```

可参考示例 [examples/train/AgentGym-RL/webarena_train.sh](./examples/train/AgentGym-RL/webarena_train.sh)。

启动 ScalingInter-RL 训练，设置：

```sh
algorithm.rounds_ctrl.type=scaling_inter_stepwise\
algorithm.rounds_ctrl.steps_scaling_inter=100 \
algorithm.rounds_ctrl.rounds=[10,20,30] \
```

可参考示例 [examples/train/ScalingInter-RL/webarena_train.sh](./examples/train/ScalingInter-RL/webarena_train.sh)。

### 评测

**1. 环境准备**

确保所需环境已搭建完毕（见上文[环境准备](#环境准备)）。

**2. 数据准备**

从 [Huggingface](https://huggingface.co/datasets/AgentGym/AgentGym-RL-Data-ID) 下载 AgentGym-RL-Data-ID 数据集。

**3. 启动环境服务端**

请参考 [AgentGym](https://github.com/WooooDyy/AgentGym/tree/640f8bca6901a6a6d540ff61522b813988da47c4) 的 `README.md` 启动环境服务端。

**4. 开始评测**

各任务的评测示例脚本位于 `examples/eval`，你也可以参考这些脚本中的评测参数配置。

要运行评测，可参考 `examples/eval/webarena_eval.sh`：

```sh
bash webarena_eval.sh
```

更多参数说明可参见 [verl 文档](https://verl.readthedocs.io/en/latest/examples/config.html)。详见 `AgentGym-RL/verl/agent_trainer/config/generation.yaml`。

### 可视化用户界面

搭建说明见[此处](https://github.com/WooooDyy/AgentGym/tree/640f8bca6901a6a6d540ff61522b813988da47c4/env-visualization)。

## 致谢

AgentGym-RL 的训练模块基于 [Verl](https://github.com/volcengine/verl) 构建，环境模块基于 [AgentGym](https://github.com/WooooDyy/AgentGym) 构建。感谢上述项目提供的基础设施支持。同时感谢 [TextCraft](https://github.com/archiki/ADaPT)、[BabyAI](https://github.com/mila-iqia/babyai)、[SciWorld](https://github.com/allenai/ScienceWorld)、[WebArena](https://github.com/web-arena-x/webarena)、[Search-R1](https://github.com/nyu-dl/dl4ir-searchQA) 的开源贡献。

## 联系

- zhxi22@m.fudan.edu.cn

## 引用

如果你觉得 AgentGym-RL 对你的工作有帮助，请引用以下论文！

```
@misc{xi2025agentgymrltrainingllmagents,
      title={AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning}, 
      author={Zhiheng Xi and Jixuan Huang and Chenyang Liao and Baodai Huang and Honglin Guo and Jiaqi Liu and Rui Zheng and Junjie Ye and Jiazheng Zhang and Wenxiang Chen and Wei He and Yiwen Ding and Guanyu Li and Zehui Chen and Zhengyin Du and Xuesong Yao and Yufei Xu and Jiecao Chen and Tao Gui and Zuxuan Wu and Qi Zhang and Xuanjing Huang and Yu-Gang Jiang},
      year={2025},
      eprint={2509.08755},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.08755}, 
}
```

```
@misc{xi2024agentgymevolvinglargelanguage,
      title={AgentGym: Evolving Large Language Model-based Agents across Diverse Environments}, 
      author={Zhiheng Xi and Yiwen Ding and Wenxiang Chen and Boyang Hong and Honglin Guo and Junzhe Wang and Dingwen Yang and Chenyang Liao and Xin Guo and Wei He and Songyang Gao and Lu Chen and Rui Zheng and Yicheng Zou and Tao Gui and Qi Zhang and Xipeng Qiu and Xuanjing Huang and Zuxuan Wu and Yu-Gang Jiang},
      year={2024},
      eprint={2406.04151},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2406.04151}, 
}
```

<div align="center">
<img src="./assets/fudannlp_logo.png" height=50><img src="./assets/bytedance.jpg" height=50><img src="./assets/shanghai_innovation_institute_logo.png" height=50>
</div>


