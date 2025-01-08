import argparse
import os
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Check if server task is completed')
    parser.add_argument('--log_dir', type=str, required=True, help='Directory to check for client_weights.csv')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 檢查檔案的完整路徑
    file_path = os.path.join(args.log_dir, 'client_weights.csv')
    
    print(f"Starting to check for file: {file_path}")
    
    while True:
        if os.path.exists(file_path):
            print(f"Found client_weights.csv in {args.log_dir}")
            break
        time.sleep(10)
        
    time.sleep(10)
        
if __name__ == '__main__':
    main()