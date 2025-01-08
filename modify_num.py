import argparse
import toml

def paser_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--clients", help="Number of clients", type=int, required = True)

    return parser.parse_args()

def main():
    args = paser_argument()
    # 讀取 TOML 檔案
    config = toml.load('pyproject.toml')
    
    clients = args.clients
    # 手動設定相依的值
    config['tool']['flwr']['federations']['local-simulation-gpu']['options']['num-supernodes'] = clients
    config['tool']['flwr']['federations']['local-simulation-gpu']['options']['backend']["client-resources"]["num-cpus"] = round(10 / clients, 1)
    config['tool']['flwr']['federations']['local-simulation-gpu']['options']['backend']["client-resources"]["num-gpus"] = round(1 / clients, 1)

    # 如果需要，可以再寫回檔案
    with open('pyproject.toml', 'w') as f:
        toml.dump(config, f)

if __name__ == "__main__":
    main()