#!/usr/bin/env python3

import subprocess
import sys
import os

def main():
    """
    簡單的啟動腳本 - 啟動 KPI OOB 監控 Dashboard
    """
    print("🚀 啟動 KPI 時序資料異常監控 Dashboard...")
    print("📊 應用程式將在瀏覽器中開啟: http://localhost:8501")
    print("⏹️  按 Ctrl+C 停止應用程式")
    print("-" * 50)
    
    try:
        # 檢查是否安裝了 streamlit
        subprocess.run([sys.executable, "-c", "import streamlit"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ 未找到 Streamlit，正在安裝必要的依賴套件...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    # 啟動 Streamlit 應用程式
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n✅ 應用程式已停止")
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")
        print("💡 請確保已安裝所有依賴套件: pip install -r requirements.txt")

if __name__ == "__main__":
    main()