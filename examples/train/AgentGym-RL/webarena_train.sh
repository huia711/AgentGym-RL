set -x
# === AgentGym-RL WebArena 训练脚本 ===
# 前置条件：
# - 你已在终端/其他会话启动 WebArena，例如：`webarena --host 0.0.0.0 --port 8010`
# - 将下方 `server_host` 与 `server_port` 配置为 panel 可访问的真实地址
# - 本脚本在 conda 环境 `agentgym-rl` 中执行训练
export VLLM_USE_MODELSCOPE=0  # 启用 ModelScope 镜像
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

# 训练任务与仓库根目录
task_name="webarena"
repo_root="/root/AgentGym-RL-main"
 # WebArena 环境服务器（外部已启动）地址（与终端启动的地址保持一致）
server_host="172.19.132.183"
server_port=8000
env_server_url="http://${server_host}:${server_port}"
# 若需脚本内启动环境服务器，使用的环境名称
env_conda_env="agentenv-webarena"
# 训练数据文件（使用绝对路径，避免 Hydra 切换工作目录影响）
# 使用过滤后的数据（去除需要地图的任务）
train_file_path="${repo_root}/AgentGym-RL-Data-ID/train/${task_name}_train_filtered.json"
# WebArena 健康检查路径（当前根路径 / 即可返回 200）
env_health_endpoint="/"

# cd AgentGym-RL
# 激活训练环境（仅用于训练，不影响外部已启动的环境服务器）
source ~/.bashrc
conda activate agentgym-rl
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export WANDB_BASE_URL=https://api.bandw.top
# 可选：Hugging Face 镜像与缓存设置（网络不通时可注释/调整）
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

wandb login 9cf5c4f192e7de9282e4dd31caf8259d4d0f0958

pure_agent_model_name="Qwen2.5-3B-Instruct"
agent_model_path="/root/AgentGym-RL-main/models/Qwen2.5-3B-Instruct"

kl_coef=0.001
policy_learning_rate=5e-6  # 1e-5
rollout_sample_num=4
train_batch_size=8  # 平衡显存和训练稳定性（避免全部失败导致崩溃）
ppo_mini_batch_size=4  # PPO mini batch 与 train batch 相同
ppo_micro_batch_size_per_gpu=1
ppo_inner_epochs=2

total_epoches=25

# 实验与模型保存路径（使用绝对路径，避免 Hydra 工作目录切换）
model_save_dir="${repo_root}/saves"
mkdir -p ${model_save_dir}
exp_name="3B-1029"
model_save_path=${model_save_dir}/${exp_name}

mkdir -p ${model_save_path}

# 健康检查：尝试连接外部 WebArena 服务
echo "等待WebArena环境服务器启动..."
for _ in $(seq 1 60); do
    if curl -fs "${env_server_url}${env_health_endpoint}" >/dev/null 2>&1; then
        echo "环境服务器已就绪: ${env_server_url}"
        break
    fi
    sleep 2
done

if ! curl -fs "${env_server_url}${env_health_endpoint}" >/dev/null 2>&1; then
    echo "环境服务器在指定时间内未准备就绪，退出训练" >&2
    exit 1
fi

# 清理旧的 Ray 会话
echo "清理旧的 Ray 会话..."
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true

# 启动训练（VERL PPO）。注意：`actor_rollout_ref.agentgym.env_addr` 使用上方的外部环境地址。
# Ray 本地模式配置：防止尝试连接远程节点
export RAY_ADDRESS=""
export RAY_DEDUP_LOGS=0
export RAY_BACKEND_LOG_LEVEL=debug

HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online python3 -m verl.agent_trainer.main_ppo  \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=15 \
    data.train_file=${train_file_path} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=750 \
    data.max_response_length=2048 \
    actor_rollout_ref.agentgym.task_name=${task_name} \
    actor_rollout_ref.agentgym.env_addr=${env_server_url} \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.agentgym.max_retries=20 \
    actor_rollout_ref.model.path=${agent_model_path} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
    actor_rollout_ref.rollout.n=${rollout_sample_num} \
    actor_rollout_ref.rollout.max_model_len=3072 \
    actor_rollout_ref.rollout.max_tokens=384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.ppo_epochs=${ppo_inner_epochs} \
    actor_rollout_ref.actor.optim.lr=${policy_learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.default_local_dir=${model_save_path} \
    trainer.project_name=AgentGym-RL \
    trainer.experiment_name=${exp_name} \
    trainer.save_freq=25 \
    trainer.total_epochs=${total_epoches} \
    critic.model.fsdp_config.grad_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.param_offload=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1
status=$?
exit $status