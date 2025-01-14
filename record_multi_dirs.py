import gymnasium as gym
from matplotlib import pyplot as plt
import argparse
import Env
from stable_baselines3 import SAC
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from utils.init_pos_config import get_init_pos, get_init_list, assert_alarm

import cv2


from collections import OrderedDict
import torch as th
import os
from typing import Dict, List, Optional
from glob import glob
import time

class RewardDisplayWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cumulative_reward = 0
        self.current_reward = 0
        
    def reset(self, **kwargs):
        self.cumulative_reward = 0
        self.current_reward = 0
        return super().reset(**kwargs)
        
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        self.current_reward = reward
        self.cumulative_reward += reward
            
        return obs, reward, done, truncated, info
        
    def render(self):
        img = super().render()
        
        img = np.ascontiguousarray(img, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 準備文字
        text_curr = f'Current Reward: {self.current_reward:.2f}'
        text_cum = f'Accumulated Reward: {self.cumulative_reward:.2f}'
        
        # 獲取文字大小
        (curr_width, curr_height), _ = cv2.getTextSize(text_curr, font, 0.5, 1)
        (cum_width, cum_height), _ = cv2.getTextSize(text_cum, font, 0.5, 1)
        
        # 使用最大寬度
        max_width = max(curr_width, cum_width)
        total_height = curr_height + cum_height + 10
        
        # 添加背景
        overlay = img.copy()
        cv2.rectangle(
            overlay, 
            (5, 5), 
            (10 + max_width, 15 + total_height), 
            (0, 0, 0), 
            -1
        )
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        # 添加文字
        cv2.putText(
            img, 
            text_curr, 
            (10, 20), 
            font, 
            0.5, 
            (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )
        
        cv2.putText(
            img, 
            text_cum, 
            (10, 20 + curr_height + 5),
            font, 
            0.5, 
            (255, 255, 255), 
            1, 
            cv2.LINE_AA
        )
        
        return img

def make_wrapped_env(env_name, i=0):
    def _init():
        env = gym.make(env_name, render_mode="rgb_array", **get_init_pos(env_name, i))
        return RewardDisplayWrapper(env)
    return _init

def load_params_to_sac(env, npz_path, verbose=False):
    """
    從 .npz 文件加載參數到 SAC 模型
    
    Args:
        env: gym 環境
        npz_path: .npz 文件路徑
        verbose: 是否打印調試信息
    
    Returns:
        loaded_model: 加載參數後的 SAC 模型
    """
    # 加載 .npz 文件中的參數
    parameters = np.load(npz_path)
    
    # 創建新的 SAC 模型
    model = SAC("MlpPolicy", env)
    
    # 創建參數字典並加載到模型中
    params_dict = zip(model.policy.state_dict().keys(), parameters.values())
    state_dict = OrderedDict({k: th.tensor(v) for k, v in params_dict})
    
    if verbose:
        print("Loading parameters with following keys:")
        for k in state_dict.keys():
            print(f"- {k}: {state_dict[k].shape}")
    
    # 使用 strict=True 確保所有參數都被正確加載
    model.policy.load_state_dict(state_dict, strict=True)
    
    return model

def load_npz_from_folders(folder_list_path: str, base_path: Optional[str] = None) -> Dict[str, np.lib.npyio.NpzFile]:
    """
    從文本文件中讀取資料夾列表，並加載每個資料夾中的第一個 .npz 文件
    
    Args:
        folder_list_path (str): 包含資料夾名稱列表的文本文件路徑
        base_path (str, optional): 基礎路徑，如果提供，資料夾路徑會相對於這個基礎路徑
    
    Returns:
        Dict[str, np.lib.npyio.NpzFile]: 以資料夾名為鍵，加載的 npz 文件為值的字典
    """
    # 檢查文本文件是否存在
    if not os.path.exists(folder_list_path):
        raise FileNotFoundError(f"找不到文件列表：{folder_list_path}")
    
    # 讀取資料夾列表
    with open(folder_list_path, 'r') as f:
        folders = [line.strip().strip("\"") for line in f if line.strip()]
    
    # 存儲加載的 npz 文件
    loaded_files = {}
    failed_folders = []
    
    # 處理每個資料夾
    for folder in folders:
        # 構建完整路徑
        if base_path:
            full_folder_path = os.path.join(base_path, folder)
        else:
            full_folder_path = folder
            
        try:
            if not os.path.exists(full_folder_path):
                raise FileNotFoundError(f"資料夾不存在：{full_folder_path}")
            
            # 使用 glob 找到資料夾中的所有 .npz 文件
            npz_files = glob(os.path.join(full_folder_path, "*.npz"))
            
            if not npz_files:
                raise FileNotFoundError(f"在資料夾中找不到 .npz 文件：{full_folder_path}")
            
            if len(npz_files) > 1:
                print(f"警告：{folder} 包含多個 .npz 文件，將使用：{os.path.basename(npz_files[0])}")
            
            # 加載第一個找到的 npz 文件
            npz_path = npz_files[0]
            environment = folder.split("_")[-2]
            loaded_files[folder] = (environment, npz_path)
            # print(f"成功加載：{npz_path}")
            
        except Exception as e:
            print(f"加載失敗 {folder}: {str(e)}")
            failed_folders.append((folder, str(e)))
            continue
    
    # 打印摘要
    print("\n加載摘要:")
    print(f"成功加載: {len(loaded_files)}/{len(folders)} 個文件")
    if failed_folders:
        print("\n加載失敗的資料夾:")
        for folder, error in failed_folders:
            print(f"- {folder}: {error}")
    
    return loaded_files

def record(env_name, model_path, save_dir, video_length = None, episodes = 3):
    model = load_params_to_sac(env_name, model_path)
    
    for i in range(len(get_init_list(env_name))):
        vec_env = DummyVecEnv([make_wrapped_env(env_name, i=i)])
        
        if video_length is None:
            max_steps = vec_env.envs[0].spec.max_episode_steps
            video_length = max_steps * episodes
        
        vec_env = VecVideoRecorder(vec_env, save_dir + "\\videos",
                    record_video_trigger=lambda x: x == 0,
                    name_prefix=f"env_index_{i}",
                    video_length=video_length,)
        
        obs = vec_env.reset()
        
        for _ in range(video_length + 1):
            action, _states = model.predict(obs, deterministic=True)
            obs, _, _, _ = vec_env.step(action)
        vec_env.close()

def test():
    index = 3
    env_name = "CartPoleSwingUpFixInitState-v1"
    log_model = ".\\multiagent\\2025_01_13_00_30_c5_CartPoleSwingUpFixInitState-v1_VW0\\final_model_round_150.npz"
    log_dir = "\\".join(log_model.split("\\")[:-1])
    assert_alarm(env_name)
    
    record(env_name, log_model, log_dir)
    
        
def record_multi_dirs():
    npz_files = load_npz_from_folders(
        folder_list_path="record_dirs.txt",
    )

    for folder, (env_name, npz) in npz_files.items():
        # 處理數據
        print(f"Folder: {folder}")
        record(env_name, npz, folder)

if __name__ == "__main__":
    # test()
    record_multi_dirs()