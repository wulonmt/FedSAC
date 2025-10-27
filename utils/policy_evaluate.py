import torch
import numpy as np
from stable_baselines3 import SAC
from CustomSAC import CustomSAC
from stable_baselines3.common.env_util import make_vec_env
import sys

def print_model_details(model):
    """印出 SAC 模型的詳細資訊"""
    
    print("=" * 80)
    print("SAC 模型詳細資訊")
    print("=" * 80)
    
    # 1. 基本資訊
    print("\n【基本資訊】")
    print(f"演算法: {model.__class__.__name__}")
    print(f"Policy 類型: {model.policy.__class__.__name__}")
    print(f"裝置: {model.device}")
    
    # 2. 超參數
    print("\n【超參數】")
    print(f"學習率 (learning_rate): {model.learning_rate}")
    print(f"Buffer 大小 (buffer_size): {model.buffer_size}")
    print(f"學習開始步數 (learning_starts): {model.learning_starts}")
    print(f"Batch 大小 (batch_size): {model.batch_size}")
    print(f"Tau (軟更新係數): {model.tau}")
    print(f"Gamma (折扣因子): {model.gamma}")
    print(f"Train 頻率: {model.train_freq}")
    print(f"Gradient 步數: {model.gradient_steps}")
    print(f"Target update interval: {model.target_update_interval}")
    print(f"Target entropy: {model.target_entropy}")
    
    # 3. 網路架構
    print("\n【Actor 網路架構】")
    print(model.actor)
    
    print("\n【Critic 網路架構】")
    print(model.critic)
    
    print("\n【Critic Target 網路架構】")
    print(model.critic_target)
    
    # 4. 各網路參數數量
    print("\n【網路參數統計】")
    
    actor_params = sum(p.numel() for p in model.actor.parameters())
    print(f"Actor 參數數量: {actor_params:,}")
    
    critic_params = sum(p.numel() for p in model.critic.parameters())
    print(f"Critic 參數數量: {critic_params:,}")
    
    critic_target_params = sum(p.numel() for p in model.critic_target.parameters())
    print(f"Critic Target 參數數量: {critic_target_params:,}")
    
    total_params = actor_params + critic_params + critic_target_params
    print(f"總參數數量: {total_params:,}")
    
    # 5. 模型大小估算 (MB)
    print("\n【模型大小估算】")
    model_size_mb = total_params * 4 / (1024 ** 2)  # 假設 float32
    print(f"模型大小: {model_size_mb:.2f} MB")
    
    # 6. 詳細的層級資訊
    print("\n【Actor 詳細層級資訊】")
    print_layer_details(model.actor)
    
    print("\n【Critic 詳細層級資訊】")
    print_layer_details(model.critic)
    
    # 7. Optimizer 資訊
    print("\n【Optimizer 資訊】")
    if hasattr(model, 'actor') and hasattr(model.actor, 'optimizer'):
        print(f"Actor Optimizer: {model.actor.optimizer}")
    if hasattr(model, 'critic') and hasattr(model.critic, 'optimizer'):
        print(f"Critic Optimizer: {model.critic.optimizer}")
    
    # 8. 動作空間和觀察空間
    print("\n【環境空間資訊】")
    print(f"觀察空間: {model.observation_space}")
    print(f"動作空間: {model.action_space}")
    
    # 9. Replay Buffer 資訊
    print("\n【Replay Buffer 資訊】")
    if hasattr(model, 'replay_buffer') and model.replay_buffer is not None:
        print(f"Buffer 類型: {model.replay_buffer.__class__.__name__}")
        print(f"Buffer 容量: {model.replay_buffer.buffer_size}")
        print(f"當前大小: {model.replay_buffer.size()}")
        print(f"指標位置: {model.replay_buffer.pos}")
    
    print("\n" + "=" * 80)

def print_layer_details(network):
    """印出網路各層的詳細資訊"""
    for name, module in network.named_modules():
        if len(list(module.children())) == 0:  # 只印葉節點
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name}: {module.__class__.__name__}, 參數數: {params:,}")
                # 印出形狀資訊
                for param_name, param in module.named_parameters(recurse=False):
                    print(f"    └─ {param_name}: {tuple(param.shape)}")

def main():
    print("正在建立環境和模型...\n")
    
    # 建立環境 (使用 Pendulum 作為範例)
    env = make_vec_env("Pendulum-v1", n_envs=1)
    
    # 建立 SAC 模型
    # model = SAC(
    #     "MlpPolicy",
    #     env,
    #     learning_rate=3e-4,
    #     buffer_size=1000000,
    #     learning_starts=100,
    #     batch_size=256,
    #     tau=0.005,
    #     gamma=0.99,
    #     train_freq=1,
    #     gradient_steps=1,
    #     verbose=1
    # )
    
    model = CustomSAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        verbose=1
    )

    # 印出詳細資訊
    print_model_details(model)
    
    # 額外資訊: 測試模型預測
    print("\n【測試模型預測】")
    obs = env.reset()
    action, _states = model.predict(obs, deterministic=True)
    print(f"輸入觀察維度: {obs.shape}")
    print(f"輸出動作維度: {action.shape}")
    print(f"動作值範例: {action}")
    
    env.close()

if __name__ == "__main__":
    main()