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

def process_paths(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # 確保行數是偶數
    if len(lines) % 2 != 0:
        print("Warning: Input file should have an even number of lines")
        return
    
    # 生成批次檔可以使用的格式
    pairs = []
    for i in range(0, len(lines), 2):
        if i + 1 >= len(lines):
            break
            
        dir1 = lines[i].strip().strip("\"").split("FedER_SAC\\")[-1]
        dir2 = lines[i + 1].strip().strip("\"").split("FedER_SAC\\")[-1]
        # print(dir1, dir2)
        
        # 提取實驗名稱
        base_name = extract_experiment_name(dir1).split('_VW')[0]  # 移除 VW 部分
        name2 = extract_experiment_name(dir1)  # 完整名稱用於 server1
        name3 = extract_experiment_name(dir2)  # 完整名稱用於 server2
        env = base_name.split('_')[-1]
        
        # 將這組資訊加入清單
        pairs.append(f"{dir1}\{env} {dir2}\{env} {base_name} {name2} {name3}")
    
    # 寫入臨時檔案供批次檔使用
    with open('compare_pairs.txt', 'w') as f:
        f.write(';'.join(pairs))

if __name__ == "__main__":
    process_paths("compare_dirs.txt")