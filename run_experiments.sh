#!/bin/bash
set -e
set -o pipefail

# ======================
# 實驗設定
# ======================

# clients_list=(5 10 16)
clients_list=(5)
# value_weight_list=(0 1)
value_weight_list=(1)
environments=(
  "CartPoleSwingUpFixInitState-v2"
  "PendulumFixPos-v1"
  "MountainCarFixPos-v1"
  "HopperFixLength-v0"
  "HalfCheetahFixLength-v0"
)
total_cpu=10

# ======================
# 產生時間字串 YYYY_MM_DD_HH_MM
# ======================
time_str=$(date +"%Y_%m_%d_%H_%M")

# 關閉 ray 重複輸出
export RAY_DEDUP_LOGS=0

# ======================
# 主迴圈
# ======================
for clients in "${clients_list[@]}"; do
  cpu_per_client=$((total_cpu / clients))
  if [ $cpu_per_client -lt 1 ]; then
    cpu_per_client=1
  fi

  echo "Debug: Clients=$clients, CPU per client=$cpu_per_client"

  for env in "${environments[@]}"; do
    # 預設回合數
    rounds=10

    case $env in
      "PendulumFixPos-v0") rounds=50 ;;
      "PendulumFixPos-v1") rounds=200 ;;
      "MountainCarFixPos-v0") rounds=200 ;;
      "MountainCarFixPos-v1") rounds=200 ;;
      "CartPoleSwingUpFixInitState-v1") rounds=150 ;;
      "CartPoleSwingUpFixInitState-v2") rounds=150 ;;
      "HopperFixLength-v0") rounds=600 ;;
      "HalfCheetahFixLength-v0") rounds=600 ;;
    esac

    for vw in "${value_weight_list[@]}"; do
      save_dir="multiagent/${time_str}_c${clients}_${env}_VW${vw}"
      mkdir -p "$save_dir"

      echo "============================================="
      echo "Environment: $env"
      echo "Rounds: $rounds"
      echo "Clients: $clients"
      echo "Value Weight: $vw"
      echo "Save Directory: $save_dir"
      echo "============================================="

      # 啟動伺服器 (背景執行)
      echo "[Server] 啟動 VWserver.py"
      python VWserver.py -r "$rounds" -c "$clients" --log_dir "$save_dir" &

      server_pid=$!
      echo "[Server] PID=$server_pid"
      sleep 5

      echo "[Clients] 啟動中..."
      for ((i=0; i<clients; i++)); do
        echo "  -> Client $i"
        python FERclient_FixState.py -i "$i" -e "$env" --value_weight "$vw" --log_dir "$save_dir" --n_cpu "$cpu_per_client" &
      done

      echo "[Monitor] 等待伺服器完成..."
      python check_server_end.py --log_dir "$save_dir"

      echo "[Monitor] 檢查完成，結束伺服器"
      kill $server_pid 2>/dev/null || true

      echo "[Experiment] $env (VW=$vw) 完成 ✅"
      echo
      echo "---------------------------------------------"
      echo
      sleep 5
    done
  done
done

echo "所有實驗完成 ✅"
