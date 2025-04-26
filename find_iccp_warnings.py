#!/usr/bin/env python3
import os
import glob
import subprocess
from tqdm import tqdm

def find_iccp_warning_images(directory):
    """
    查找包含iCCP警告的圖片文件
    使用PIL庫來加載圖片，並捕獲標準錯誤輸出
    """
    print(f"檢查目錄: {directory}")
    
    # 確保目錄存在
    if not os.path.exists(directory):
        print(f"錯誤: 目錄 {directory} 不存在")
        return []
    
    # 獲取圖片文件列表
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    
    print(f"找到 {len(image_files)} 個圖片檔案")
    
    # 對每個文件進行測試
    problematic_files = []
    
    for img_path in tqdm(image_files, desc="檢查圖片"):
        # 創建一個簡單的Python命令來打開和加載圖片
        cmd = f"python -c \"from PIL import Image; Image.open('{img_path}').load()\" 2>&1"
        
        # 執行命令並捕獲所有輸出
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        output = process.communicate()[0]
        
        # 檢查輸出中是否包含iCCP警告
        if "iCCP: known incorrect sRGB profile" in output:
            problematic_files.append(img_path)
            print(f"發現問題文件: {img_path}")
    
    return problematic_files

# 主程序
if __name__ == "__main__":
    # 設定要檢查的目錄
    train_dir = "datasets/yoga82/images/train"
    val_dir = "datasets/yoga82/images/val"
    
    # 執行檢查
    print("正在檢查訓練集...")
    train_problems = find_iccp_warning_images(train_dir)
    
    print("\n正在檢查驗證集...")
    val_problems = find_iccp_warning_images(val_dir)
    
    # 輸出結果
    print("\n===== 結果摘要 =====")
    print(f"訓練集中發現 {len(train_problems)} 個問題文件")
    print(f"驗證集中發現 {len(val_problems)} 個問題文件")
    
    # 保存結果到文件
    output_file = "iccp_warning_files.txt"
    with open(output_file, "w") as f:
        f.write("===== 訓練集問題文件 =====\n")
        for file in train_problems:
            f.write(f"{file}\n")
        
        f.write("\n===== 驗證集問題文件 =====\n")
        for file in val_problems:
            f.write(f"{file}\n")
    
    print(f"\n結果已保存到 {output_file}") 