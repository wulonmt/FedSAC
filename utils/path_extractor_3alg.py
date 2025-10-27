import os

def extract_experiment_name(path):
    # 從路徑中獲取最後一個資料夾名稱
    folder_name = os.path.basename(path.strip())
    
    # 尋找包含實驗參數的部分（從 "c" 開始）
    parts = folder_name.split('_')
    for i, part in enumerate(parts):
        if part.startswith('c'):
            return '_'.join(parts[i:])
    
    return folder_name

def get_name_by_suffix(path, vw1_count):
    """根據路徑結尾決定名稱"""
    if path.endswith('_VW0'):
        return 'uniform'
    elif path.endswith('_VW1'):
        if vw1_count == 0:
            return 'softmax'
        else:
            return 'rweight'
    else:
        # 如果沒有特定結尾，使用原本的邏輯
        return extract_experiment_name(path)

def process_paths(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # 確保行數是 3 的倍數
    if len(lines) % 3 != 0:
        print("Warning: Input file should have a number of lines divisible by 3")
        return
    
    # 生成批次檔可以使用的格式
    pairs = []
    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break
        
        vw1_count = 0  # 用來追蹤 VW1 出現的次數
        
        dir1 = lines[i].strip().strip("\"").split("FedER_SAC\\")[-1]
        dir2 = lines[i + 1].strip().strip("\"").split("FedER_SAC\\")[-1]
        dir3 = lines[i + 2].strip().strip("\"").split("FedER_SAC\\")[-1]
        
        # 提取實驗名稱
        base_name = extract_experiment_name(dir1).split('_VW')[0]  # 移除 VW 部分
        env = base_name.split('_')[-1]
        
        # 根據路徑結尾決定名稱
        name1 = get_name_by_suffix(lines[i].strip().strip("\""), vw1_count)
        if lines[i].strip().strip("\"").endswith('_VW1'):
            vw1_count += 1
        
        name2 = get_name_by_suffix(lines[i + 1].strip().strip("\""), vw1_count)
        if lines[i + 1].strip().strip("\"").endswith('_VW1'):
            vw1_count += 1
        
        name3 = get_name_by_suffix(lines[i + 2].strip().strip("\""), vw1_count)
        
        # 將這組資訊加入清單
        pairs.append(f"{dir1}\\{env} {dir2}\\{env} {dir3}\\{env} {base_name} {name1} {name2} {name3}")
    
    # 寫入檔案供批次檔使用
    with open('compare_pairs_3alg.txt', 'w') as f:
        f.write(';'.join(pairs))
    
    print(f"處理完成！共處理了 {len(pairs)} 組路徑")
    print(f"結果已寫入 compare_pairs_3alg.txt")

if __name__ == "__main__":
    process_paths("compare_dirs_3alg.txt")