from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
import matplotlib.pyplot as plt
from tsmoothie import ConvolutionSmoother, LowessSmoother
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_name", help="data log name", type=str, required = True)
args = parser.parse_args()

if __name__ == "__main__":
    subdirectories = [f.path for f in os.scandir(args.log_name) if f.is_dir()]

    event_loader_list = []

    for subdir in subdirectories:
        files = [f.path for f in os.scandir(subdir) if f.is_file()]
        if len(files) == 1:
            file_path = files[0]
            event_loader_list.append(EventFileLoader(file_path))
        else:
            print(f"Error: Found {len(files)} files in {subdir}. Expected only one.")


    wall_times = []
    step_numbers = []
    metrics = ["rollout/ep_rew_mean", "train/approx_kl", "train/entropy_loss", "train/loss", "train/old_entropy", "train/policy_gradient_loss", "train/value_loss"]
    metrics_dict = {m:([], []) for m in metrics}

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
        
    for k, (x, y) in metrics_dict.items():
        smoother = LowessSmoother(smooth_fraction=0.05, iterations=1)
        smoother.smooth(y)
        low, up = smoother.get_intervals('sigma_interval')
        plt.figure(figsize=(11,6))
        plt.plot(x, smoother.smooth_data[0], linewidth=3, color='blue')
        plt.plot(x, smoother.data[0], '.k')
        plt.xlabel("steps")
        plt.title(k)
        plt.fill_between(x, low[0], up[0], alpha=0.3)
        
    plt.show()
