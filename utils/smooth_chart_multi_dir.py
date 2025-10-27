import argparse
import os
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
import matplotlib.pyplot as plt
from tsmoothie import LowessSmoother
import numpy as np

# 新增：消除環境偏差的統計方法
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_names", nargs='+', help="data log names", type=str, required=True) # \CartPoleSwingUpFixInitState-v1_ppo\
parser.add_argument("-n", "--custom_names", nargs='+', help="custom names for each log", type=str)
parser.add_argument("-s", "--save_dir", help="directory to save plots", type=str, required=True)
parser.add_argument("-o", "--remove_outliers", action="store_true", help="remove outliers from the data")
parser.add_argument("--iqr_factor", type=float, default=1.5, help="IQR factor for outlier removal (default: 1.5)")
parser.add_argument("-p", "--prefixes", type=str, default="0,1,2,3,4", help="Comma-separated list of prefixes of subdirectories to process (default: 0,1,2,3)")
args = parser.parse_args()

def process_folder(folder_path, prefix):
    event_loader_list = []
    true_folder_path = [f.path for f in os.scandir(folder_path) if f.is_dir() and f.name.startswith(prefix)]
    assert len(true_folder_path) == 1, "Prefix more than 1"
    
    subdirectories = [f.path for f in os.scandir(true_folder_path[0]) if f.is_dir()]
    
    for subdir in subdirectories:
        files = [f.path for f in os.scandir(subdir) if f.is_file()]
        if len(files) == 1:
            file_path = files[0]
            event_loader_list.append(EventFileLoader(file_path))
        else:
            print(f"Error: Found {len(files)} files in {subdir}. Expected only one.")
    
    # metrics = ["rollout/ep_rew_mean", "train/approx_kl", "train/entropy_loss", "train/loss", "train/old_entropy", "train/policy_gradient_loss", "train/value_loss", "train/std"]
    metrics = ["rollout/ep_rew_mean", "rollout/Return", "train/actor_loss", "train/critic_loss", "train/ent_coef"]
    metrics_dict = {m: ([], []) for m in metrics}
    
    for event_file in event_loader_list:
        for event in event_file.Load():
            if len(event.summary.value) > 0:
                for m in metrics:
                    if event.summary.value[0].tag == m:
                        metrics_dict[m][0].append(event.step)
                        metrics_dict[m][1].append(event.summary.value[0].tensor.float_val[0])
    
    for k, (x, y) in metrics_dict.items():
        combined = list(zip(x, y))
        sorted_combined = sorted(combined, key=lambda item: item[0])
        metrics_dict[k] = tuple(zip(*sorted_combined))
    
    return metrics_dict

def custom_smoother(y, smoother_y, coef = 0.3):
    low, up = [], []
    last_range = 0
    for data, target in zip(y, smoother_y):
        data_range = abs(data - target)
        last_range = last_range + coef * (data_range - last_range)
        up.append(target + last_range)
        low.append(target - last_range)
    return low, up

def remove_outliers(x, y, iqr_factor):
    x = np.array(x)
    y = np.array(y)
    
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - iqr_factor * IQR
    upper_bound = Q3 + iqr_factor * IQR
    
    mask = (y >= lower_bound) & (y <= upper_bound)
    
    return x[mask], y[mask]

def remove_environment_bias(data_dict, method='z_score_within_env'):
    """
    消除不同環境間的偏差
    
    Args:
        data_dict: {custom_name: [(x, y), (x, y), ...]} 每個實驗方法在各環境的數據
        method: 偏差消除方法
            - 'z_score_within_env': 每個環境內進行Z-score標準化
            - 'min_max_within_env': 每個環境內進行Min-Max標準化
            - 'robust_within_env': 每個環境內進行Robust標準化（對outlier較不敏感）
            - 'percent_change': 轉換為相對變化百分比
            - 'baseline_normalization': 以初始值為基準進行標準化
            - 'rank_normalization': 轉換為排名標準化
    """
    normalized_data = {}
    
    for custom_name, env_data_list in data_dict.items():
        normalized_env_data = []
        
        for env_idx, (x, y) in enumerate(env_data_list):
            y = np.array(y)
            
            if method == 'z_score_within_env':
                # 每個環境內的Z-score標準化 (y - mean) / std
                if np.std(y) > 0:
                    y_normalized = (y - np.mean(y)) / np.std(y)
                else:
                    y_normalized = y - np.mean(y)
                    
            elif method == 'min_max_within_env':
                # 每個環境內的Min-Max標準化 (y - min) / (max - min)
                y_min, y_max = np.min(y), np.max(y)
                if y_max - y_min > 0:
                    y_normalized = (y - y_min) / (y_max - y_min)
                else:
                    y_normalized = np.zeros_like(y)
                    
            elif method == 'robust_within_env':
                # 每個環境內的Robust標準化 (y - median) / IQR
                median = np.median(y)
                q75, q25 = np.percentile(y, [75, 25])
                iqr = q75 - q25
                if iqr > 0:
                    y_normalized = (y - median) / iqr
                else:
                    y_normalized = y - median
                    
            elif method == 'percent_change':
                # 轉換為相對於第一個值的百分比變化
                if len(y) > 0 and y[0] != 0:
                    y_normalized = ((y - y[0]) / abs(y[0])) * 100
                else:
                    y_normalized = np.zeros_like(y)
                    
            elif method == 'baseline_normalization':
                # 以初始值為基準的標準化
                if len(y) > 0:
                    # 使用前10%的值作為baseline（更穩定）
                    baseline_length = max(1, len(y) // 10)
                    baseline = np.mean(y[:baseline_length])
                    if baseline != 0:
                        y_normalized = y / baseline
                    else:
                        y_normalized = np.ones_like(y)
                else:
                    y_normalized = y
                    
            elif method == 'rank_normalization':
                # 排名標準化：將數值轉換為排名百分位
                y_normalized = stats.rankdata(y) / len(y)
                
            else:
                y_normalized = y
            
            normalized_env_data.append((x, y_normalized))
        
        normalized_data[custom_name] = normalized_env_data
    
    return normalized_data

def create_summary_with_bias_removal():
    """創建消除偏差後的統計比較圖"""
    print("Creating bias-removed summary comparison plots...")
    
    # 收集所有環境下每個log_name的metrics數據
    raw_metrics_data = {}  # {metric: {log_name: [(x, y), (x, y), ...]}}
    
    for prefix in prefixes:
        for log_name, custom_name in zip(args.log_names, custom_names):
            try:
                metrics_data = process_folder(log_name, prefix + "_")
                for metric, (x, y) in metrics_data.items():
                    if args.remove_outliers:
                        x, y = remove_outliers(x, y, args.iqr_factor)
                    
                    # 初始化數據結構
                    if metric not in raw_metrics_data:
                        raw_metrics_data[metric] = {}
                    if custom_name not in raw_metrics_data[metric]:
                        raw_metrics_data[metric][custom_name] = []
                    
                    # 存儲每個環境的數據
                    raw_metrics_data[metric][custom_name].append((x, y))
            except Exception as e:
                print(f"{e}, metric: {metric}, error at: {log_name}, {prefix}")
    
    # 創建多種偏差消除方法的summary資料夾
    bias_removal_methods = {
        'z_score_within_env': 'Z-Score Normalization',
        'robust_within_env': 'Robust Normalization', 
        'percent_change': 'Percent Change',
        'baseline_normalization': 'Baseline Normalization',
        'no_normalization': 'No Normalization (Original)'
    }
    
    for method_key, method_name in bias_removal_methods.items():
        print(f"Processing with {method_name}...")
        
        summary_dir = os.path.join(args.save_dir, f"summary_{method_key}")
        os.makedirs(summary_dir, exist_ok=True)
        
        # 為每個metric創建統計比較圖
        for metric, log_data in raw_metrics_data.items():
            # 消除環境偏差
            if method_key != 'no_normalization':
                normalized_log_data = remove_environment_bias(log_data, method_key)
            else:
                normalized_log_data = log_data
            
            plt.figure(figsize=(12, 7))
            
            for custom_name, env_data_list in normalized_log_data.items():
                if not env_data_list:
                    continue
                    
                # 找到所有數據的最小長度，確保對齊
                min_length = min(len(y) for x, y in env_data_list)
                
                # 收集所有環境的y值進行統計
                all_y_values = []
                x_values = None
                
                for x, y in env_data_list:
                    if x_values is None:
                        x_values = x[:min_length]
                    # 確保所有數據長度一致
                    y_trimmed = y[:min_length]
                    all_y_values.append(y_trimmed)
                
                if not all_y_values:
                    continue
                    
                # 轉換為numpy array進行統計計算
                all_y_values = np.array(all_y_values)
                
                # 計算平均值和標準誤差
                mean_y = np.mean(all_y_values, axis=0)
                std_y = np.std(all_y_values, axis=0)
                stderr_y = std_y / np.sqrt(len(all_y_values))  # 標準誤差
                
                # 平滑處理
                smoother = LowessSmoother(smooth_fraction=0.05, iterations=1)
                smoother.smooth(mean_y)
                smoothed_mean = smoother.smooth_data[0]
                
                # 也平滑標準誤差
                smoother_stderr = LowessSmoother(smooth_fraction=0.05, iterations=1)
                smoother_stderr.smooth(stderr_y)
                smoothed_stderr = smoother_stderr.smooth_data[0]
                
                # 繪製平均值線
                plt.plot(x_values, smoothed_mean, linewidth=2.5, label=f"{custom_name}")
                
                # 繪製95%信賴區間 (±1.96 * stderr)
                ci_95 = 1.96 * smoothed_stderr
                plt.fill_between(x_values, 
                                smoothed_mean - ci_95, 
                                smoothed_mean + ci_95, 
                                alpha=0.2)
            
            plt.xlabel("steps")
            
            # 根據正規化方法調整y軸標籤
            if method_key == 'z_score_within_env':
                plt.ylabel(f"{metric} (Z-score normalized)")
            elif method_key == 'robust_within_env':
                plt.ylabel(f"{metric} (Robust normalized)")
            elif method_key == 'percent_change':
                plt.ylabel(f"{metric} (% change from initial)")
            elif method_key == 'baseline_normalization':
                plt.ylabel(f"{metric} (normalized to baseline)")
            else:
                plt.ylabel(metric)
                
            plt.title(f"{metric} - {method_name}\n(95% CI across {len(prefixes)} environments)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # 儲存圖片
            safe_metric_name = metric.replace('/', '_')
            save_path = os.path.join(summary_dir, f"{safe_metric_name}_summary.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Plots with {method_name} saved to {summary_dir}")

if __name__ == "__main__":
    prefixes = args.prefixes.split(',')
    custom_names = args.custom_names if args.custom_names else args.log_names
    
    if len(custom_names) != len(args.log_names):
        raise ValueError("The number of custom names must match the number of log directories.")
    
    whole_metrics_data = []

    for prefix in prefixes:
        all_metrics_data = {}
        
        for log_name, custom_name in zip(args.log_names, custom_names):
            try:
                metrics_data = process_folder(log_name, prefix + "_")
                for metric, (x, y) in metrics_data.items():
                    if args.remove_outliers:
                        x, y = remove_outliers(x, y, args.iqr_factor)
                    if metric not in all_metrics_data:
                        all_metrics_data[metric] = []
                    all_metrics_data[metric].append((custom_name, x, y))
            except Exception as e:
                print(f"{e}, metric: {metric}, error at: {log_name}, {prefix}")
        
        for metric, data_list in all_metrics_data.items():
            plt.figure(figsize=(11, 6))
            
            for custom_name, x, y in data_list:
                smoother = LowessSmoother(smooth_fraction=0.05, iterations=1)
                smoother.smooth(y)
                low, up = custom_smoother(y, smoother.smooth_data[0])
                
                plt.plot(x, smoother.smooth_data[0], linewidth=2, label=custom_name)
                plt.fill_between(x, low, up, alpha=0.1)
            
            plt.xlabel("steps")
            plt.ylabel(metric)
            plt.title(f"{metric}" if args.remove_outliers else f"{metric} (Prefix: {prefix})")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            save_subdir = os.path.join(args.save_dir, prefix)
            os.makedirs(save_subdir, exist_ok=True)
            
            safe_metric_name = metric.replace('/', '_')
            outlier_info = f"_no_outliers_iqr{args.iqr_factor}" if args.remove_outliers else ""
            # save_path = os.path.join(save_subdir, f"{safe_metric_name}{outlier_info}.png")
            save_path = os.path.join(save_subdir, f"{safe_metric_name}.png")
            plt.savefig(save_path)
            # print(f"Saved plot for {metric} (Prefix: {prefix}) to {save_path}")
            plt.close()

        # print(f"All plots for prefix {prefix} have been saved to {os.path.join(args.save_dir, prefix)}")

    # print("Processing complete for all prefixes.")

    # 新增：統計所有環境的metrics比較
    print("Creating summary comparison plots...")

    # 收集所有環境下每個log_name的metrics數據
    summary_metrics_data = {}  # {metric: {log_name: [(x, y), (x, y), ...]}}

    for prefix in prefixes:
        for log_name, custom_name in zip(args.log_names, custom_names):
            try:
                metrics_data = process_folder(log_name, prefix + "_")
                for metric, (x, y) in metrics_data.items():
                    if args.remove_outliers:
                        x, y = remove_outliers(x, y, args.iqr_factor)
                    
                    # 初始化數據結構
                    if metric not in summary_metrics_data:
                        summary_metrics_data[metric] = {}
                    if custom_name not in summary_metrics_data[metric]:
                        summary_metrics_data[metric][custom_name] = []
                    
                    # 存儲每個環境的數據
                    summary_metrics_data[metric][custom_name].append((x, y))
            except Exception as e:
                print(f"{e}, metric: {metric}, error at: {log_name}, {prefix}")

    # 創建summary資料夾
    summary_dir = os.path.join(args.save_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    # 為每個metric創建統計比較圖
    for metric, log_data in summary_metrics_data.items():
        plt.figure(figsize=(12, 7))
        
        for custom_name, env_data_list in log_data.items():
            # 統計處理：計算所有環境的平均值和標準差
            if not env_data_list:
                continue
                
            # 找到所有數據的最小長度，確保對齊
            min_length = min(len(y) for x, y in env_data_list)
            
            # 收集所有環境的y值進行統計
            all_y_values = []
            x_values = None
            
            for x, y in env_data_list:
                if x_values is None:
                    x_values = x[:min_length]
                # 確保所有數據長度一致
                y_trimmed = y[:min_length]
                all_y_values.append(y_trimmed)
            
            if not all_y_values:
                continue
                
            # 轉換為numpy array進行統計計算
            import numpy as np
            all_y_values = np.array(all_y_values)
            
            # 計算平均值和標準差
            mean_y = np.mean(all_y_values, axis=0)
            std_y = np.std(all_y_values, axis=0)
            
            # 平滑處理
            smoother = LowessSmoother(smooth_fraction=0.05, iterations=1)
            smoother.smooth(mean_y)
            smoothed_mean = smoother.smooth_data[0]
            
            # 計算信賴區間（平均值 ± 標準差）
            # 也可以平滑標準差
            smoother_std = LowessSmoother(smooth_fraction=0.05, iterations=1)
            smoother_std.smooth(std_y)
            smoothed_std = smoother_std.smooth_data[0]
            
            # 繪製平均值線
            plt.plot(x_values, smoothed_mean, linewidth=2.5, label=f"{custom_name} (avg across envs)")
            
            # 繪製信賴區間
            plt.fill_between(x_values, 
                            smoothed_mean - smoothed_std, 
                            smoothed_mean + smoothed_std, 
                            alpha=0.2, label=f"{custom_name} ±1σ")
        
        plt.xlabel("steps")
        plt.ylabel(metric)
        plt.title(f"{metric} - Summary Comparison Across All Environments")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # 儲存圖片
        safe_metric_name = metric.replace('/', '_')
        outlier_info = f"_no_outliers_iqr{args.iqr_factor}" if args.remove_outliers else ""
        save_path = os.path.join(summary_dir, f"{safe_metric_name}_summary.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot for {metric} to {save_path}")
        plt.close()

    print(f"Summary plots saved to {summary_dir}")

    # 執行偏差消除的統計分析
    create_summary_with_bias_removal()
    print("Bias removal analysis completed!")

    