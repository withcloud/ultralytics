import os
import glob
from tqdm import tqdm
import sys
import subprocess
from pathlib import Path

def find_problematic_images(directory):
    """查找引發 libpng iCCP 警告的圖片"""
    print(f"檢查目錄: {directory}")
    
    # 確保目錄存在
    if not os.path.exists(directory):
        print(f"錯誤：目錄 {directory} 不存在")
        return []
    
    # 獲取所有圖片檔案
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    
    print(f"找到 {len(image_files)} 個圖片檔案")
    
    problematic_files = []
    
    # 創建一個臨時Python腳本來讀取圖片並觸發警告
    temp_script = """
import sys
from PIL import Image
import warnings

# 讀取圖片
try:
    img = Image.open(sys.argv[1])
    img.load()
except Exception as e:
    print(f"錯誤讀取圖片: {e}", file=sys.stderr)
    sys.exit(1)
"""
    
    # 創建臨時腳本檔案
    with open("temp_check_image.py", "w") as f:
        f.write(temp_script)
    
    # 逐一處理每個圖片
    for img_path in tqdm(image_files, desc="檢查圖片"):
        # 通過Python運行腳本，並捕獲stderr
        cmd = f"python temp_check_image.py '{img_path}' 2>&1"
        output = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 檢查輸出中是否包含iCCP警告
        if "iCCP: known incorrect sRGB profile" in output.stderr or "iCCP: known incorrect sRGB profile" in output.stdout:
            problematic_files.append(img_path)
            print(f"發現問題文件: {img_path}")
    
    # 刪除臨時腳本
    os.remove("temp_check_image.py")
    
    return problematic_files

# 檢查訓練和驗證數據集
train_dir = "datasets/yoga82/images/train"
val_dir = "datasets/yoga82/images/val"

train_problems = find_problematic_images(train_dir)
val_problems = find_problematic_images(val_dir)

# 顯示結果
print("\n===== 結果摘要 =====")
print(f"訓練集問題檔案數量: {len(train_problems)}")
print(f"驗證集問題檔案數量: {len(val_problems)}")

# 儲存結果到檔案
with open("iccp_warning_files.txt", "w") as f:
    f.write("=== 訓練集問題檔案 ===\n")
    for file in train_problems:
        f.write(f"{file}\n")
    
    f.write("\n=== 驗證集問題檔案 ===\n")
    for file in val_problems:
        f.write(f"{file}\n")

print(f"結果已保存到 iccp_warning_files.txt") 