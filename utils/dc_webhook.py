import json
import requests
import os
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

# 從環境變數中取得 Webhook URL
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def send_discord_webhook(message: str, webhook_url: str = WEBHOOK_URL):
    # 檢查 webhook_url 是否存在
    if not webhook_url:
        print("警告: Discord Webhook URL 未設定，訊息無法發送")
        return False
    
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "content": message
    }
    
    try:
        response = requests.post(webhook_url, headers=headers, data=json.dumps(data))
        if response.status_code == 204:
            print("Discord Webhook sent successfully")
            return True
        else:
            print("Discord Webhook send error:", response.status_code, response.text)
            return False
    except requests.exceptions.RequestException as e:
        print(f"Discord Webhook 發送失敗: {e}")
        return False