# 项目整体框架

## `AgentGym/agentenv/agentenv/envs/webarena.py`
为 AgentGym 通用 env 抽象 注册 WebArena 环境客户端与任务类型。
关键类：

* `WebarenaEnvClient(BaseEnvClient)`：
  * 持有 env_server_base（即 env_addr），在 __init__ 中向环境服务端发送 POST /create 建立一个 env 实例，拿到 env_idx。
  * 实现 observe()：通过 GET /observation 拉取当前 observation。
  * 实现 step(action)：通过 POST /step 发送 LLM 输出的动作，并解析返回的 observation / reward / terminated。
  * reset(idx) / close()：对应 env server 的 /reset 与 /close 接口，在 episode 结束或退出时重置 / 关闭环境实例。

* `WebarenaTask(BaseTask)`：
  * 指定 env_client_cls = WebarenaEnvClient，env_name = "Webarena"。
  * Verl 端根据 task_name=webarena 选择该 Task，从而在训练时真正连上 WebArena HTTP 环境。




## `AgentGym/agentenv_webarena`
* `agentenv_webarena/environment.py`
    * 将底层 WebArena 环境封装成 environment 接口（reset / step / observation 等），对外提供统一 API
    * 内部会管理浏览器、任务加载、奖励计算等细节

* `agentenv_webarena/server.py`
    * 环境 HTTP 服务端，暴露 /create、/observation、/step、/reset、/close 等 REST 接口
    * 训练脚本中的 actor_rollout_ref.agentgym.env_addr=${env_server_url} 就是指向这个服务，例如 http://172.19.132.183:8000

* `agentenv_webarena/launch.py`
    * 以命令行方式启动 WebArena 环境服务的入口（通常 README 会让你用它来启动服务）

### `webarena`
WebArena 原始代码仓（作为子模块/依赖放在这里）

* `.auth`
    * 部署网站的 cookies 验证

* `brower_env/auto_login.py`【主流程】
    * 准备 cookies 验证文件，完成网站自动登录，生成 .auth 目录

* `agent/agent.py`【WebArena 内置智能体定义】
  * 定义 Agent 抽象类，以及两类核心 Agent：
    * TeacherForcingAgent：从配置 JSON 中读取参考动作序列（reference_action_sequence），按顺序回放，用于上限评估 / debug
    * PromptAgent：根据浏览器 Trajectory + intent + meta_data 调用 PromptConstructor 构造 prompt，调用底层 LLM（OpenAI / HF 等），并将响应解析为浏览器 Action
  * construct_agent(args)：按照命令行参数（agent_type、instruction_path、provider、model 等）构建具体 Agent 实例

* `agent/prompts`
【提示词模板与构造器】
  * raw/：以 Python 文件形式维护多种 prompt 模板（CoT / direct、是否允许 NA、不同模型风格等）
  * to_json.py：将 raw/ 中的 Python 模板转换为 jsons/ 下的 JSON 版本，便于运行时读取
  * prompt_constructor.py：定义 PromptConstructor 及其子类，负责：
    * 从 instruction json 读取系统提示、few-shot 示例等
    * 在每一轮根据 observation / 轨迹拼接成最终输入 LLM 的长 prompt
    * 提供 extract_action(...) 从 LLM 的自然语言输出中抽取可执行动作字符串

* `environment_docker`
完整的 Docker 化部署（包括 Dockerfile / docker-compose 等），可将 WebArena 环境一键部署到云服务器（例如阿里云）。

* `evaluation_harness/evaluation`
评估器与奖励聚合的实现

* `run.py`
【端到端评测入口脚本】
  * 作为官方 WebArena benchmark 的 CLI 入口：
    * 解析命令行参数（渲染开关、观测类型、最大步数、模型提供方/名称、日志目录等）。
    * 使用 agent/agent.py 中的 construct_agent 构建具体 Agent。
    * 利用 ScriptBrowserEnv 与 evaluation_harness 对一批任务配置文件进行回放与评分。
  * 不经过 AgentGym-RL / Verl，也可以单独运行 WebArena 基准评测。




## `AgentGym-RL-main/AgentGym-RL`
* `AgentItemId/webarena_train.json`
    * 训练数据到环境 item ID 的映射表，用于把数据集中的样本映射到具体的 WebArena 任务实例（方便复现 / 诊断）

* `scripts`
脚本集

### `Verl`
Verl 强化学习训练框架

#### `agent_trainer`
【重要：训练器核心】

* `main_ppo.py`
    * PPO / GRPO 的主训练入口，`webarena_train.sh` 里 `python3 -m verl.agent_trainer.main_ppo` 调用的就是它。
    * 解析 Hydra 配置（ppo_trainer.yaml + 命令行覆盖）；构建数据集 / dataloader；调用 utils.agentgym.client.init_env_client 创建 WebArena 环境客户端；构建 actor / critic / optimizer；启动 Ray 进程、分布式训练 Loop。

* `ppo/ray_trainer.py`
【训练的整体过程】
  * 分布式训练主控：管理 Ray 集群 / worker 的创建与销毁。
  * 负责分发 rollout 任务、收集经验、调用 core_algos 更新参数，并周期性保存 checkpoint / 日志。

* `ppo/core_algos.py`
【核心RL算法】
  * 实现 PPO / GRPO / RLOO / REINFORCE++ 等核心 RL 算法的损失与优势估计。
  * 定义策略更新的细节（clip、KL 正则、value function 损失等）。

* `ppo/metric_utils.py`
【监控与统计指标】
  * 封装训练过程的各种统计：KL、value loss、policy loss、优势分布、reward 统计等。
  * 统一对接 logger / wandb，便于多实验对比。

* `config/ppo_trainer.yaml`
    * PPO 训练的默认配置文件（batch 大小、KL 控制、rounds 策略、并行度等）。
    * `webarena_train.sh` 用命令行 override 其中部分字段（如 algorithm.rounds_ctrl.*、data.*、actor_rollout_ref.* 等），来适配 WebArena 的具体超参。

#### `workers`
* `agent_actor`
Actor 策略网络工作线程
  * base.py：定义 BasePPOActor 抽象，约定 actor 如何接收 rollout 数据、计算 log_prob / entropy、执行梯度更新。
  * dp_actor.py：基于数据并行（data parallel）的 Actor 实现，可在多张 GPU 之间分布策略模型、同步梯度。

* `agent_critic`
Critic 价值网络工作线程
  * base.py：定义 BasePPOCritic 抽象（compute_values / update_critic）。
  * dp_critic.py：基于数据并行的 Critic 实现，拟合 value function，用于优势估计与降低方差。

* `reward_manager`
奖励管理器
  * naive.py：最简单的奖励聚合器，从环境或 reward_model 收集原始奖励并做基础汇总（如 episode 累积）。
  * prime.py：为 Prime 系列任务（代码 / 数学）提供复杂的奖励路由与组合逻辑（多阶段评分、规则 + 模型混合等）。

* `reward_model`
奖励模型
  * base.py：奖励模型的抽象基类，定义给定输入（如模型输出 / 轨迹）返回标量奖励的接口。
  * megatron/reward_model.py：基于 Megatron 的大规模 reward model 实现，支持分布式部署与高吞吐推理。

* `rollout`
采样 / rollout 子系统
  * base.py：rollout worker 抽象，统一“如何驱动策略与环境交互，产出 trajectories”。
  * hf_rollout.py：基于 Hugging Face transformers 的 rollout 实现。
  * agent_vllm_rollout/vllm_rollout.py：基于 vLLM 的高吞吐 rollout，实现批量推理、attention mask / position ids / loss mask 等构造，是 AgentGym-RL 中多轮交互的关键组件。
  * naive/naive_rollout.py：简化版 rollout，便于小规模实验或调试。
  * schemas.py：定义 rollout 数据结构（observation / action / reward / done 等）的模式，保证多模块之间的数据格式兼容。
  * tokenizer.py：为 rollout 阶段提供统一的分词封装（HF / vLLM 共用）。

#### `utils`

* `agentgym/client.py`
    * Verl 侧的环境客户端路由器，根据 task_name 选择具体 env client。
    * envclient_classes 字典中包含 "webarena": WebarenaEnvClient 等映射。
    * init_env_client(args) 根据 args.task_name 和 args.env_addr 不断重试连接，直到对应环境服务端准备就绪。
    * 就是这里把 “Hydra 参数里的 actor_rollout_ref.agentgym.task_name=webarena + env_addr” 变成真正的 Python 客户端实例。

* `reward_score`
奖励评分工具 ✅
    * 针对 GSM8K、数学、Prime-Code / Prime-Math 等任务定义细粒度评分逻辑（如数值对比、表达式归一化、代码执行验证）。
    * WebArena 训练中不会直接用到，但为自定义奖励函数提供了参考实现范式。
