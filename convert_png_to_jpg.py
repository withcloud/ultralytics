#!/usr/bin/env python3
import os
import glob
from tqdm import tqdm
import cv2
from pathlib import Path

def convert_png_to_jpg(directory):
    """
    將目錄中的所有PNG圖片轉換為JPG格式，並刪除原始PNG檔
    """
    print(f"處理目錄: {directory}")
    
    # 確保目錄存在
    if not os.path.exists(directory):
        print(f"錯誤: 目錄 {directory} 不存在")
        return 0
    
    # 獲取所有PNG檔案
    png_files = glob.glob(os.path.join(directory, "*.png"))
    print(f"找到 {len(png_files)} 個PNG檔案")
    
    converted_count = 0
    error_count = 0
    
    # 處理每個PNG檔案
    for png_file in tqdm(png_files, desc="轉換PNG到JPG"):
        try:
            # 讀取PNG圖片
            img = cv2.imread(png_file)
            if img is None:
                print(f"警告: 無法讀取圖片 {png_file}")
                error_count += 1
                continue
            
            # 創建JPG的檔案路徑 (更改副檔名)
            jpg_file = os.path.splitext(png_file)[0] + ".jpg"
            
            # 保存為JPG格式 (使用95%的質量)
            cv2.imwrite(jpg_file, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # 刪除原始PNG檔
            os.remove(png_file)
            
            converted_count += 1
        except Exception as e:
            print(f"處理 {png_file} 時發生錯誤: {str(e)}")
            error_count += 1
    
    return converted_count, error_count

# 主程序
if __name__ == "__main__":
    # 設定要處理的目錄
    train_dir = "datasets/yoga82/images/train"
    val_dir = "datasets/yoga82/images/val"
    
    # 處理訓練集
    print("\n正在處理訓練集...")
    train_converted, train_errors = convert_png_to_jpg(train_dir)
    
    # 處理驗證集
    print("\n正在處理驗證集...")
    val_converted, val_errors = convert_png_to_jpg(val_dir)
    
    # 輸出結果報告
    print("\n===== 轉換結果摘要 =====")
    print(f"訓練集成功轉換: {train_converted} 個檔案")
    print(f"訓練集轉換失敗: {train_errors} 個檔案")
    print(f"驗證集成功轉換: {val_converted} 個檔案")
    print(f"驗證集轉換失敗: {val_errors} 個檔案")
    print(f"總共轉換: {train_converted + val_converted} 個檔案")
    print("==========================") 