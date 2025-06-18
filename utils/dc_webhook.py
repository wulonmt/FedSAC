import json
import requests
import os
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

# 從環境變數中取得 Webhook URL
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

def send_discord_webhook(message: str, webhook_url: str = WEBHOOK_URL):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "content": message
    }
    response = requests.post(webhook_url, headers=headers, data=json.dumps(data))
    if response.status_code == 204:
        print("Discord Webhook sent successfully")
    else:
        print("Discord Webhook send error:", response.status_code, response.text)
