import numpy as np

INIT_POS = {
    "MountainCarFixPos-v0": [
        {"init_x": 0, "x_limit": 0.15},
        {"init_x": -0.3, "x_limit": 0.15},
        {"init_x": -0.6, "x_limit": 0.15},
        {"init_x": 0.3, "x_limit": 0.15},
        {"init_x": 0, "x_limit": 0.15}
    ],
    "PendulumFixPos-v0": [
        {"init_theta": np.pi*3/4, "init_thetadot": 1},
        {"init_theta": -np.pi*3/4, "init_thetadot": 1},
        {"init_theta": np.pi/2, "init_thetadot": 1},
        {"init_theta": -np.pi/2, "init_thetadot": 1},
        {"init_theta": np.pi, "init_thetadot": 1}
    ],
    "CartPoleSwingUpFixInitState-v1": [
        {"init_x": 2, "init_angle": np.pi/2},
        {"init_x": -2, "init_angle": np.pi/2},
        {"init_x": 2, "init_angle": -np.pi/2},
        {"init_x": -2, "init_angle": -np.pi/2},
        {"init_x": 0, "init_angle": np.pi}
    ],
    "HopperFixLength-v0": [
        {'thigh_scale': 1.5, 'leg_scale': 1},
        {'thigh_scale': 1, 'leg_scale': 1.5},
        {'thigh_scale': 1.5, 'leg_scale': 1.5},
        {'thigh_scale': 0.6, 'leg_scale': 1},
        {'thigh_scale': 1, 'leg_scale': 1}
    ],
    "HalfCheetahFixLength-v0": [
        {'bthigh_scale': 1.5, 'fthigh_scale': 1.0},
        {'bthigh_scale': 1.0, 'fthigh_scale': 1.5},
        {'bthigh_scale': 1.5, 'fthigh_scale': 1.5},
        {'bthigh_scale': 0.7, 'fthigh_scale': 0.7},
        {'bthigh_scale': 1.0, 'fthigh_scale': 1.0}
    ]
}

def get_init_pos(env_name, index):
    """
    獲取指定環境和索引的初始位置參數
    
    :param env_name: 環境名稱
    :param index: 初始位置索引
    :return: 初始位置參數字典
    """
    pos_len = len(INIT_POS[env_name])
    return INIT_POS[env_name][min(index, pos_len - 1)]

def get_init_list(env_name):
    """
    獲取指定環境列表
    
    :param env_name: 環境名稱
    :return: 環境初始位置列表
    """
    return INIT_POS[env_name]

def get_param_names(env_name):
    """
    獲取指定環境的參數名稱列表
    
    :param env_name: 環境名稱
    :return: 參數名稱列表
    """
    return list(INIT_POS[env_name][0].keys())

def is_valid_env(env_name):
    """
    檢查環境名稱是否有效
    
    :param env_name: 環境名稱
    :return: 如果環境名稱有效則返回 True，否則返回 False
    """
    return env_name in INIT_POS

def get_available_envs():
    """
    獲取所有可用的環境名稱
    
    :return: 可用環境名稱的列表
    """
    return list(INIT_POS.keys())

def assert_alarm(env_name):
    assert is_valid_env(env_name), f"Only environments {', '.join(get_available_envs())} are available"