import os
import glob
from tqdm import tqdm
import subprocess

def check_images_for_iccp_warning(directory):
    """檢查目錄中的圖片是否有iCCP警告"""
    print(f"檢查目錄: {directory}")
    
    # 獲取所有圖片文件
    image_files = glob.glob(os.path.join(directory, "*.jpg")) + \
                  glob.glob(os.path.join(directory, "*.png")) + \
                  glob.glob(os.path.join(directory, "*.jpeg"))
    
    print(f"找到 {len(image_files)} 個圖片檔案")
    
    problematic_files = []
    
    for img_path in tqdm(image_files, desc="檢查圖片"):
        # 使用 ImageMagick 的 identify 命令檢查圖片
        cmd = ["identify", "-verbose", img_path]
        
        try:
            # 執行命令並捕獲輸出
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            # 檢查是否有 iCCP 簽名
            if "Profile-icc" in process.stdout or "iCCP" in process.stdout:
                problematic_files.append(img_path)
                print(f"找到可能有問題的檔案: {img_path}")
        except subprocess.TimeoutExpired:
            print(f"處理超時: {img_path}")
        except Exception as e:
            print(f"處理錯誤 {img_path}: {str(e)}")
    
    return problematic_files

# 檢查訓練集
train_dir = "datasets/yoga82/images/train"
train_problems = check_images_for_iccp_warning(train_dir)

# 檢查驗證集
val_dir = "datasets/yoga82/images/val"
val_problems = check_images_for_iccp_warning(val_dir)

# 輸出結果
print("\n===== 結果摘要 =====")
print(f"訓練集問題檔案: {len(train_problems)}")
for file in train_problems:
    print(f"  - {file}")

print(f"\n驗證集問題檔案: {len(val_problems)}")
for file in val_problems:
    print(f"  - {file}")

# 將結果保存到文件
with open("problematic_png_files.txt", "w") as f:
    f.write("=== 訓練集問題檔案 ===\n")
    for file in train_problems:
        f.write(f"{file}\n")
    
    f.write("\n=== 驗證集問題檔案 ===\n")
    for file in val_problems:
        f.write(f"{file}\n")

print(f"結果已保存到 problematic_png_files.txt") 