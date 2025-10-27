import subprocess
import os
import time
import datetime

def process_compare_pairs():
    """讀取compare_pairs_3alg.txt並執行相應的Python腳本"""
    
    try:
        subprocess.run('python ./utils/path_extractor_3alg.py')
        with open('compare_pairs_3alg.txt', 'r', encoding='utf-8') as file:
            content = file.read().strip()
        
        # 分割不同的比較組合（使用分號分隔）
        compare_groups = content.split(';')
        
        for group in compare_groups:
            group = group.strip()
            if not group:
                continue
                
            # 分割每組的7個參數
            params = group.split()
            
            if len(params) >= 7:
                dir1, dir2, dir3, result1, result2, result3, result4 = params[:7]
                now = datetime.datetime.now().strftime("%Y%m%d_%H_%M_%S")
                
                print("Processing directories:")
                print(f"  {dir1}")
                print(f"  {dir2}")
                print(f"  {dir3}")
                print("-" * 23)
                print("Generating results:")
                print(f"  {result1}")
                print(f"  {result2}")
                print(f"  {result3}")
                print(f"  {result4}")
                print()
                
                iqr_factor = 2
                print(f"{iqr_factor = }")
                
                # 執行四個命令：一個比較三個資料夾，三個單獨分析
                commands = [
                    ['python', './utils/smooth_chart_multi_dir.py', '-l', dir1, dir2, dir3, '-n', result2, result3, result4, '-s', f'results/{now}_{result1}', '--iqr_factor', str(iqr_factor)],
                    ['python', './utils/smooth_chart_one_server.py', '-l', dir1, '-s', f'results/{now}_{result2}'],
                    ['python', './utils/smooth_chart_one_server.py', '-l', dir2, '-s', f'results/{now}_{result3}'],
                    ['python', './utils/smooth_chart_one_server.py', '-l', dir3, '-s', f'results/{now}_{result4}']
                ]
                
                for cmd in commands:
                    try:
                        print(f"Executing: {' '.join(cmd)}")
                        subprocess.run(cmd, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"Error executing command: {e}")
                    except FileNotFoundError:
                        print(f"File not found: {cmd[0]}")
                
                print()
                print("-" * 24)
                print()
                
                # 添加短暫延遲
                time.sleep(2)
            else:
                print(f"Warning: Insufficient parameters in group: {group}")
                
    except FileNotFoundError:
        print("Error: compare_pairs_3alg.txt not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    process_compare_pairs()