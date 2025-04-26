#!/usr/bin/env python3
import os
import glob
from tqdm import tqdm
import cv2
import imghdr
from collections import Counter

def convert_non_jpg_to_jpg(directory):
    """
    檢查目錄中的圖片，將實際類型不是jpg的檔案轉換為真正的jpg檔案
    特別注意檢查.jpg檔案的實際類型
    """
    print(f"處理目錄: {directory}")
    
    # 確保目錄存在
    if not os.path.exists(directory):
        print(f"錯誤: 目錄 {directory} 不存在")
        return None
    
    # 獲取所有圖片檔案
    image_files = []
    
    # 特別尋找.jpg和.jpeg檔案進行檢查
    jpg_files = glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.jpeg"))
    print(f"找到 {len(jpg_files)} 個JPG檔案")
    
    # 其他圖片格式
    other_files = []
    for ext in ['*.png', '*.gif', '*.bmp', '*.webp', '*.tiff', '*.tif']:
        other_files.extend(glob.glob(os.path.join(directory, ext)))
    print(f"找到 {len(other_files)} 個其他格式圖片檔案")
    
    # 合併所有圖片檔案
    image_files = jpg_files + other_files
    print(f"總共找到 {len(image_files)} 個圖片檔案")
    
    # 檢查並轉換每個檔案
    file_types = {}
    converted_count = 0
    error_count = 0
    
    # 計數器
    false_jpg_count = 0  # 副檔名是.jpg但實際不是jpg的檔案數
    
    for img_path in tqdm(image_files, desc="檢查並轉換檔案"):
        # 獲取副檔名
        extension = os.path.splitext(img_path)[1].lower()
        
        # 使用imghdr檢測真實的圖片類型
        try:
            img_type = imghdr.what(img_path)
            file_types[img_path] = img_type if img_type else "unknown"
            
            # 檢查副檔名為.jpg/.jpeg但實際不是jpg/jpeg的檔案
            if extension in ['.jpg', '.jpeg'] and img_type not in ['jpeg', 'jpg']:
                false_jpg_count += 1
                print(f"警告: {img_path} 副檔名為{extension}但實際類型為{img_type}")
            
            # 如果實際類型不是jpg/jpeg，轉換它
            if img_type not in ['jpeg', 'jpg']:
                # 讀取圖片
                img = cv2.imread(img_path)
                if img is None:
                    print(f"警告: 無法讀取圖片 {img_path}")
                    error_count += 1
                    continue
                
                # 創建新的JPG文件路徑，確保副檔名是.jpg
                new_path = os.path.splitext(img_path)[0] + ".jpg"
                
                # 保存為JPG格式 (使用95%的質量)
                cv2.imwrite(new_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                # 如果新路徑與原路徑不同，刪除原始檔案
                if new_path != img_path:
                    os.remove(img_path)
                    print(f"轉換: {img_path} -> {new_path}")
                else:
                    print(f"重寫: {img_path} (實際類型為{img_type})")
                
                converted_count += 1
        except Exception as e:
            print(f"處理 {img_path} 時發生錯誤: {str(e)}")
            error_count += 1
    
    # 返回處理結果
    return {
        'total_files': len(image_files),
        'jpg_files': len(jpg_files),
        'other_files': len(other_files),
        'false_jpg_files': false_jpg_count,
        'converted': converted_count,
        'errors': error_count,
        'image_types': Counter(file_types.values())
    }

# 主程序
if __name__ == "__main__":
    # 設定要處理的目錄
    train_dir = "datasets/yoga82/images/train"
    val_dir = "datasets/yoga82/images/val"
    
    # 處理訓練集
    print("\n正在處理訓練集...")
    train_results = convert_non_jpg_to_jpg(train_dir)
    
    # 處理驗證集
    print("\n正在處理驗證集...")
    val_results = convert_non_jpg_to_jpg(val_dir)
    
    # 輸出結果報告
    print("\n===== 轉換結果摘要 =====")
    if train_results:
        print(f"訓練集檔案總數: {train_results['total_files']}")
        print(f"JPG檔案數: {train_results['jpg_files']}")
        print(f"假JPG檔案數(副檔名是jpg但實際不是): {train_results['false_jpg_files']}")
        print(f"轉換檔案數: {train_results['converted']}")
        print(f"失敗數: {train_results['errors']}")
        print(f"檔案類型統計: {dict(train_results['image_types'])}")
    
    if val_results:
        print(f"\n驗證集檔案總數: {val_results['total_files']}")
        print(f"JPG檔案數: {val_results['jpg_files']}")
        print(f"假JPG檔案數(副檔名是jpg但實際不是): {val_results['false_jpg_files']}")
        print(f"轉換檔案數: {val_results['converted']}")
        print(f"失敗數: {val_results['errors']}")
        print(f"檔案類型統計: {dict(val_results['image_types'])}")
    
    print("\n總共轉換: {0} 個檔案".format(
        (train_results['converted'] if train_results else 0) + 
        (val_results['converted'] if val_results else 0)
    ))
    print("==========================")
    
    # 再次運行檢查腳本以確認所有文件現在都是JPG
    print("\n正在再次檢查所有檔案類型...")
    os.system("python3 check_image_types.py") 