#!/usr/bin/env bash
set -euo pipefail

# WebArena 训练前健康检查与 API 自检
# 用法：
#   bash /root/webarena_preflight.sh http://127.0.0.1:8000

ENV_ADDR="${1:-http://127.0.0.1:8000}"
AUTH_DIR="./AgentGym/agentenv-webarena/webarena/.auth"
CONFIG_DIR="./AgentGym/agentenv-webarena/webarena/config_files"

echo "[PRECHECK] ENV_ADDR=${ENV_ADDR}"
echo "[PRECHECK] AUTH_DIR=${AUTH_DIR}"
echo "[PRECHECK] CONFIG_DIR=${CONFIG_DIR}"

echo "[STEP 1] 健康检查 - HTTP /"
if curl -fsS "${ENV_ADDR}/" >/dev/null; then
  echo "  OK: ${ENV_ADDR}/ 可达"
else
  echo "  FAIL: ${ENV_ADDR}/ 不可达，请先在另一个终端启动：conda activate agentenv-webarena && webarena --host 0.0.0.0 --port 8000" >&2
  exit 1
fi

echo "[STEP 2] 登录态文件检查 - ${AUTH_DIR}"
missing=()
need=("shopping_state.json" "shopping_admin_state.json" "gitlab_state.json")
mkdir -p "${AUTH_DIR}" || true
for f in "${need[@]}"; do
  if [[ -s "${AUTH_DIR}/${f}" ]]; then
    echo "  OK: ${f} 存在(${AUTH_DIR}/${f})"
  else
    echo "  MISS: ${f} 不存在(${AUTH_DIR}/${f})"
    missing+=("${f}")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "  建议：先运行 bash /root/generate_webarena_cookies.sh 生成/更新登录态"
fi

echo "[STEP 3] API 自检 - /create -> /reset -> /observation -> /close"
python3 - <<'PY'
import os, json, sys, time
import requests

env_addr=os.environ.get('ENV_ADDR','http://127.0.0.1:8000')
test_ids=[230,330]  # 可根据需要追加 369 观察失败样本

def jprint(tag, obj):
    try:
        print(tag, json.dumps(obj, ensure_ascii=False))
    except Exception:
        print(tag, str(obj))

try:
    r=requests.post(f"{env_addr}/create", timeout=60)
    r.raise_for_status()
    env_idx=r.json()["env_idx"]
    print(f"  OK: create -> env_idx={env_idx}")
except Exception as e:
    print(f"  FAIL: /create 错误: {e}")
    sys.exit(2)

ok_cnt=0
for item_id in test_ids:
    try:
        rr=requests.post(f"{env_addr}/reset", json={"env_idx":env_idx,"seed":0,"idx":item_id}, timeout=120)
        rr.raise_for_status()
        data=rr.json()
        jprint(f"  reset[{item_id}] ->", {"sites":data.get("sites"), "object":data.get("object"), "observation_head":str(data.get("observation",""))[:80]})
        if data.get("observation")!="TimeoutError":
            ok_cnt+=1
        ro=requests.get(f"{env_addr}/observation", params={"env_idx":env_idx}, timeout=30)
        ro.raise_for_status()
        obs=ro.json()
        jprint(f"  observe[{item_id}] ->", {"text_head":str(obs)[:80]})
    except Exception as e:
        print(f"  WARN: id={item_id} 自检失败: {e}")

try:
    rc=requests.post(f"{env_addr}/close", json={"env_idx":env_idx}, timeout=30)
    rc.raise_for_status()
    print("  OK: close")
except Exception as e:
    print(f"  WARN: /close 错误: {e}")

print(f"  自检通过条目数: {ok_cnt}/{len(test_ids)}")
PY

echo "[SUMMARY] 若存在 MISS 登录态或自检 WARN，请先修复后再启动训练。"
echo "[DONE] 预检结束"


