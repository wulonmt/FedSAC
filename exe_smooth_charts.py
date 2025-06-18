import subprocess
import os
import time

def process_compare_pairs():
    """讀取compare_pairs.txt並執行相應的Python腳本"""
    
    try:
        with open('compare_pairs.txt', 'r', encoding='utf-8') as file:
            content = file.read().strip()
        
        # 分割不同的比較組合（使用分號分隔）
        compare_groups = content.split(';')
        
        for group in compare_groups:
            group = group.strip()
            if not group:
                continue
                
            # 分割每組的5個參數
            params = group.split()
            
            if len(params) >= 5:
                dir1, dir2, result1, result2, result3 = params[:5]
                
                print("Processing directories:")
                print(f"  {dir1}")
                print(f"  {dir2}")
                print("-" * 23)
                print("Generating results:")
                print(f"  {result1}")
                print(f"  {result2}")
                print(f"  {result3}")
                print()
                
                # 執行三個命令
                commands = [
                    ['python', './utils/smooth_chart_multi_dir.py', '-l', dir1, dir2, '-n', 'value_weighted', 'uniform', '-s', f'results/{result1}'],
                    ['python', './utils/smooth_chart_one_server.py', '-l', dir1, '-s', f'results/{result2}'],
                    ['python', './utils/smooth_chart_one_server.py', '-l', dir2, '-s', f'results/{result3}']
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
        print("Error: compare_pairs.txt not found")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    process_compare_pairs()