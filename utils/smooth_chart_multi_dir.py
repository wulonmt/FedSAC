import argparse
import os
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
import matplotlib.pyplot as plt
from tsmoothie import LowessSmoother
import numpy as np

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

if __name__ == "__main__":
    prefixes = args.prefixes.split(',')
    custom_names = args.custom_names if args.custom_names else args.log_names
    
    if len(custom_names) != len(args.log_names):
        raise ValueError("The number of custom names must match the number of log directories.")
    
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