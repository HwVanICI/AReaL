#!/bin/bash
export HCCL_IF_BASE_PORT=63000
export HCCL_NPU_SOCKET_PORT_RANGE="62100-62350"
unset https_proxy http_proxy proxy
MASTER_NODE_PORT="${MASTER_NODE_PORT:-6699}"

source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

_detect_n_gpus_per_node() {
  local n
  if [ -n "${MA_NUM_GPUS}" ]; then
    printf '%s' "${MA_NUM_GPUS}"
    return
  fi
  if [ -n "${ASCEND_RT_VISIBLE_DEVICES}" ]; then
    printf '%s' "$(printf '%s' "${ASCEND_RT_VISIBLE_DEVICES}" | awk -F, '{print NF}')"
    return
  fi
  n=$(ls -d /dev/davinci[0-9]* 2>/dev/null | wc -l)
  if [ "${n:-0}" -gt 0 ]; then
    printf '%s' "$n"
    return
  fi
  n=$(npu-smi info -l 2>/dev/null | awk '/Chip Count/{s+=$NF} END{print s+0}')
  printf '%s' "${n:-0}"
}

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-$(_detect_n_gpus_per_node)}"
export N_GPUS_PER_NODE
if ! [[ "${N_GPUS_PER_NODE}" =~ ^[0-9]+$ ]] || [ "${N_GPUS_PER_NODE}" -lt 1 ]; then
  echo "ERROR: 无法确定每节点 NPU 数（N_GPUS_PER_NODE='${N_GPUS_PER_NODE}'），请显式设置 MA_NUM_GPUS" >&2
  exit 1
fi

MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-900}"
CHECK_INTERVAL="${CHECK_INTERVAL:-3}"
LOG_FILE="${LOG_FILE:-latest.log}"     # 作业脚本可覆盖

# 返回值：master 继续往下跑；worker 在这里直接退出
cluster_init() {
  mkdir -p ${FILE_ROOT_PATH}/logs/root/${EXP_NAME}/${TRIAL_NAME}
  NODE_NUM="${NODE_NUM:-1}"
  IS_MASTER=0                 # 只有主节点会置 1；worker 在下面直接 exit
  local master_addr="${VC_WORKER_HOSTS%%,*}"
  local current_ip
  # 裸机场景没有 VC_WORKER_HOSTS，可由作业脚本预先指定 MASTER_NODE_IP
  MASTER_NODE_IP="${MASTER_NODE_IP:-$(python -c "import socket; print(socket.gethostbyname('${master_addr}'))")}"
  current_ip=$(hostname -I | awk '{print $1}')
  [ -z "$current_ip" ] && { echo "无法提取当前节点IP"; return 1; }
  echo "current ip: $current_ip, master ip: $MASTER_NODE_IP"

  if [ "$current_ip" != "$MASTER_NODE_IP" ]; then
    unset ASCEND_RT_VISIBLE_DEVICES
    echo "start workers, connect to master: $MASTER_NODE_IP:$MASTER_NODE_PORT"
    sleep 30s
    ray start --address="${MASTER_NODE_IP}:${MASTER_NODE_PORT}" --block
    echo "Worker received shutdown signal."
    exit 0                     # 直接结束整个作业脚本
  fi
  IS_MASTER=1

  if [ "$NODE_NUM" -gt 0 ]; then   # 保持原语义：1 节点也要起 ray head
    unset ASCEND_RT_VISIBLE_DEVICES
    echo "start Ray master node..."
    ray stop --force
    ray start --head --port="$MASTER_NODE_PORT"
    local elapsed=0 cur
    while true; do
      cur=$(ray status | grep -c "node_" || echo 0)
      [[ "$cur" =~ ^[0-9]+$ ]] || cur=0
      echo "current_node_num: $cur"
      [ "$cur" -ge "$NODE_NUM" ] && { echo "all $NODE_NUM workers node is ready."; break; }
      if [ "$elapsed" -ge "$MAX_WAIT_SECONDS" ]; then
        echo "timeout($MAX_WAIT_SECONDS s), current_node_num: $cur"
        ray stop --force; exit 1
      fi
      sleep "$CHECK_INTERVAL"
      elapsed=$((elapsed + CHECK_INTERVAL))
    done
  fi
}

# 统一的启动 + 日志过滤
run_job() {
  local script="$1"; shift
  python -u "$script" "$@" 2>&1 \
    | grep -v "apex.normalization.fused_layer_norm" \
    | tee "$LOG_FILE"
}

# 收尾 + 错误判定
finish_job() {
  sleep 60s
  local log
  log=$(tail -n 500 "$LOG_FILE")
  ray stop --force
  if ! echo "$log" | grep -q "TrainController destroyed" \
     && echo "$log" | grep -v "async_llm" \
        | grep -iqE "Traceback|timeouterror|unknown applicaiton exception"; then
    exit 1
  fi
  exit 0
}
