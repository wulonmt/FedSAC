import gymnasium as gym

# 印出所有註冊的環境名稱
envs = gym.envs.registry

print(envs.keys())