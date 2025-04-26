#!/usr/bin/env python3
import os
import glob
from tqdm import tqdm
from collections import Counter
import imghdr

def check_image_types(directory):
    """
    檢查目錄中的圖片類型，確認是否都是jpg檔
    """
    print(f"檢查目錄: {directory}")
    
    # 確保目錄存在
    if not os.path.exists(directory):
        print(f"錯誤: 目錄 {directory} 不存在")
        return None
    
    # 獲取所有圖片檔案
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp', '*.tiff', '*.tif']:
        image_files.extend(glob.glob(os.path.join(directory, ext)))
    
    print(f"找到 {len(image_files)} 個圖片檔案")
    
    # 檢查每個檔案的類型
    file_types = {}
    file_extensions = Counter()
    
    for img_path in tqdm(image_files, desc="檢查檔案類型"):
        # 獲取文件擴展名
        extension = os.path.splitext(img_path)[1].lower()
        file_extensions[extension] += 1
        
        # 使用imghdr檢測真實的圖片類型
        try:
            img_type = imghdr.what(img_path)
            if img_type:
                file_types[img_path] = img_type
            else:
                file_types[img_path] = "unknown"
        except Exception as e:
            print(f"檢查 {img_path} 時發生錯誤: {str(e)}")
            file_types[img_path] = "error"
    
    # 尋找不是jpg的文件
    non_jpg_files = []
    for file_path, file_type in file_types.items():
        if file_type not in ['jpeg', 'jpg']:
            non_jpg_files.append((file_path, file_type))
    
    return {
        'total_files': len(image_files),
        'extensions': file_extensions,
        'image_types': Counter(file_types.values()),
        'non_jpg_files': non_jpg_files
    }

# 主程序
if __name__ == "__main__":
    # 設定要檢查的目錄
    train_dir = "datasets/yoga82/images/train"
    val_dir = "datasets/yoga82/images/val"
    
    # 檢查訓練集
    print("\n正在檢查訓練集...")
    train_results = check_image_types(train_dir)
    
    # 檢查驗證集
    print("\n正在檢查驗證集...")
    val_results = check_image_types(val_dir)
    
    # 輸出結果報告
    print("\n===== 檢查結果摘要 =====")
    if train_results:
        print(f"訓練集檔案總數: {train_results['total_files']}")
        print(f"檔案擴展名統計: {dict(train_results['extensions'])}")
        print(f"圖片類型統計: {dict(train_results['image_types'])}")
        print(f"非JPG檔案數量: {len(train_results['non_jpg_files'])}")
        
        if train_results['non_jpg_files']:
            print("\n訓練集中的非JPG檔案:")
            for file_path, file_type in train_results['non_jpg_files']:
                print(f"  - {file_path} (實際類型: {file_type})")
    
    if val_results:
        print(f"\n驗證集檔案總數: {val_results['total_files']}")
        print(f"檔案擴展名統計: {dict(val_results['extensions'])}")
        print(f"圖片類型統計: {dict(val_results['image_types'])}")
        print(f"非JPG檔案數量: {len(val_results['non_jpg_files'])}")
        
        if val_results['non_jpg_files']:
            print("\n驗證集中的非JPG檔案:")
            for file_path, file_type in val_results['non_jpg_files']:
                print(f"  - {file_path} (實際類型: {file_type})")
    
    # 檢查結論
    all_jpg = True
    if train_results and train_results['non_jpg_files']:
        all_jpg = False
    if val_results and val_results['non_jpg_files']:
        all_jpg = False
    
    print("\n===== 檢查結論 =====")
    if all_jpg:
        print("✅ 所有圖片都是JPG檔案格式")
    else:
        print("❌ 發現非JPG格式的圖片檔案")
    print("=======================") 