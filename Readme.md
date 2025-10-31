# Federated Reinforcement Learning with Value-Weighted Aggregation

## 概述 (Overview)

本專案將強化學習 (Reinforcement Learning) 與聯邦學習 (Federated Learning) 結合,使用 **回報值 (Return)** 作為 FedAvg 演算法中的聚合權重,以改進模型訓練效果。

### 核心特色

- **強化學習框架**: 使用 Stable-Baselines3 的 SAC (Soft Actor-Critic) 模型
- **聯邦學習框架**: 基於 Flower (flwr) 實現
- **多種聚合策略比較**:
  - 均勻權重 (Uniform)
  - Softmax 值加權 (Value-Weighted Softmax)
  - 最小值調整歸一化 (Min-Adjusted Normalization)

### 訓練環境

本專案支援多個強化學習環境,並針對聯邦學習場景進行了**客製化調整** (Custom Modifications),使其更適合本演算法的特性與需求。

#### 支援環境列表

| 環境名稱 | 類型 | 說明 |
|---------|------|------|
| **CartPoleSwingUp** | 經典控制 | 倒立擺搖擺問題 |
| **Pendulum** | 經典控制 | 單擺平衡控制 |
| **MountainCar** | 經典控制 | 山地小車爬坡 |
| **Hopper** | 機器人控制 | 單足跳躍機器人 |
| **HalfCheetah** | 機器人控制 | 半身獵豹奔跑 |

#### 客製化環境實現

所有環境的客製化版本位於 **`Env/envs/`** 目錄下。

> 💡 **提示**: 若要使用這些客製化環境,請確保從 `Env.envs` 模組導入,而非直接使用 Gym/Gymnasium 的標準環境。
---

## 環境需求 (Requirements)

- **Python**: 3.10
- **PyTorch**: 2.9.0+cu126

### 安裝步驟
```bash
pip install -r requirements.txt
```

---

## 使用說明 (Usage)

### 1. 啟動伺服器 (Server)

**執行檔案**: `VWserver.py`

#### 參數說明 (Parameters)

| 參數 (Argument) | 簡寫 | 說明 (Description) | 類型 (Type) | 預設值 (Default) |
|----------------|------|-------------------|------------|----------------|
| `--port` | `-p` | 本地端口 (Local port) | `str` | `8080` |
| `--rounds` | `-r` | 總訓練輪數 (Total training rounds) | `int` | `300` |
| `--clients` | `-c` | 客戶端數量 (Number of clients) | `int` | `2` |
| `--log_dir` | | 伺服器與客戶端日誌目錄 (Server & client log directory) | `str` | `None` |
| `--value_weighted` | `-v` | 值加權模式 (Value weighting mode) | `int` | `0` |

**值加權模式說明**:
- `0`: 均勻權重 (Uniform weighting)
- `1`: Softmax 值加權 (Softmax value-weighted)
- `2`: 最小值調整歸一化 (Min-adjusted normalization)

#### 啟動範例
```bash
python VWserver.py -p 8080 -r 300 -c 2 -v 1
```

---

### 2. 啟動客戶端 (Client)

**執行檔案**: `FERclient_FixState.py`

#### 參數說明 (Parameters)

| 參數 (Argument) | 簡寫 | 說明 (Description) | 類型 (Type) | 預設值 (Default) | 必填 (Required) |
|----------------|------|-------------------|------------|----------------|----------------|
| `--log_name` | `-l` | 自訂日誌名稱 (Custom log name) | `str` | `auto` | |
| `--environment` | `-e` | 訓練環境名稱 (Training environment) | `str` | | ✓ |
| `--index` | `-i` | 客戶端索引 (Client index) | `int` | `0` | ✓ |
| `--port` | `-p` | 本地端口 (Local port) | `str` | `8080` | |
| `--time_step` | `-m` | 訓練時間步數 (Training timesteps) | `int` | `5000` | |
| `--kl_coef` | | KL 散度係數 (KL divergence coefficient) | `float` | `0` | |
| `--value_weight` | | 值加權模式 (Value weighting mode) | `int` | `0` | |
| `--log_dir` | | 日誌目錄 (Log directory) | `str` | `None` | |
| `--n_cpu` | | CPU 核心數 (Number of CPUs) | `int` | `1` | |

**值加權模式說明**:

| 參數值 | 聚合權重策略 (Aggregation Strategy) |
|-------|----------------------------------|
| `0` | 每個客戶端權重相同 (Uniform) |
| `1` | 使用值加權並對所有客戶端權重進行 Softmax |
| `2` | 使用值加權並對所有客戶端權重進行最小值調整歸一化 |

#### 啟動範例
```bash
# 客戶端 1
python FERclient_FixState.py -e CartPoleSwingUp -i 0 -p 8080 -m 5000 --value_weight 1

# 客戶端 2
python FERclient_FixState.py -e Pendulum -i 1 -p 8080 -m 5000 --value_weight 1
```

#### 訓練結果

訓練結果預設存儲於 `multiagent` 資料夾中。

---

## 模型評估與記錄 (Evaluation & Recording)

**執行檔案**: `record.py`

### 參數說明 (Parameters)

| 參數 (Argument) | 簡寫 | 說明 (Description) | 類型 (Type) | 必填 (Required) |
|----------------|------|-------------------|------------|----------------|
| `--log_model` | `-l` | 待評估的模型路徑 (Model path to evaluate) | `str` | |
| `--environment` | `-e` | 環境名稱 (Environment name) | `str` | ✓ |
| `--record` | | 錄製影片 (Record video) | `flag` | |
| `--snapshot` | | 顯示快照 (Display snapshots) | `flag` | |
| `--evaluate` | | 評估模型 (Evaluate model) | `flag` | |
| `--display` | | 顯示可視化環境 (Display visual environment) | `flag` | |

### 使用範例
```bash
# 評估模型並顯示環境
python record.py -l ./multiagent/.../ -e CartPoleSwingUp --evaluate --display

# 錄製模型執行影片
python record.py -l ./multiagent/.../ -e Pendulum --record
```

---

## 結果視覺化 (Visualization)

### 比較不同聚合策略

使用 `utils/smooth_chart_multi_dir.py` 比較相同環境下不同聚合策略的訓練結果。
```bash
python utils/smooth_chart_multi_dir.py
```

### 比較同一伺服器下不同客戶端

使用 `utils/smooth_chart_one_server.py` 比較同一伺服器中不同客戶端的訓練表現。
```bash
python utils/smooth_chart_one_server.py
```

---

## 核心演算法

### 1. 回報值收集 (Return Collection)

**位置**: `utils/CustomSAC.py` → `collect_rollouts` 函數

當 `value_weight > 0` 時,每當環境產生 `done` 信號,系統會將該回合的累積回報值 (Return) 記錄至 `last_R` 列表中。
```python
actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

# 執行動作並獲取回饋
new_obs, rewards, dones, infos = env.step(actions)

for n in range(self.n_envs):
    self.current_R[n] += rewards[n]
    if dones[n]:
        # 記錄本回合的累積回報
        self.last_R[n].append(self.current_R[n])
        self.current_R[n] = 0
        
        # 記錄至日誌
        if self.last_R[n]:
            avg_return = sum(self.last_R[n]) / len(self.last_R[n])
            self.logger.record("rollout/Return", avg_return)
```

### 2. 聚合策略實現 (Aggregation Strategies)

**位置**: `utils/merge_alg.py`

#### 策略 1: 均勻權重 (Uniform)
```python
if option == AggregationStrategy.UNIFORM:
    # 所有客戶端權重相等
    multiple_weights = [1/len(results) for _, fit_res in results]
```

#### 策略 2: Softmax 值加權 (Value-Weighted Softmax)
```python
elif option == AggregationStrategy.VW_SOFTMAX:
    all_values = [fit_res.num_examples for (_, fit_res) in results]
    max_value = max(all_values)
    
    # 數值穩定的 Softmax 計算
    exp_values = [exp(value - max_value) for value in all_values]
    exp_values = [round(x, 6) for x in exp_values]
    values_exp_sum = sum(exp_values)
    
    multiple_weights = [value / values_exp_sum for value in exp_values]
```

#### 策略 3: 最小值調整歸一化 (Min-Adjusted Normalization)
```python
elif option == AggregationStrategy.VW_MIN_ADJUSTED:
    all_values = [fit_res.num_examples for (_, fit_res) in results]
    min_value = min(all_values)
    
    # 減去最小值並歸一化
    adjusted_values = [value - min_value for value in all_values]
    adjusted_sum = sum(adjusted_values) + 1e-4
    
    h = len(adjusted_values)
    multiple_weights = [
        (value / adjusted_sum + (1 / h)) / 2 
        for value in adjusted_values
    ]
```