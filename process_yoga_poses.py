#!/usr/bin/env python3
# Script to process yoga82 dataset images with YOLOv11x-pose model and save results in specific format
import os
import glob
from pathlib import Path
from tqdm import tqdm
import torch
import cv2
import numpy as np
from ultralytics import YOLO

# Define paths
INPUT_DIR = "datasets/yoga82/images/train"
OUTPUT_DIR = "datasets/yoga82/labels/train"
PRED_DIR = "datasets/yoga82/preds/train"

TEST_MODE = True

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# 關鍵點連接線的定義 (人體骨架)
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13],  # 腿
    [12, 13], [6, 12], [7, 13],  # 軀幹
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11],  # 手臂
    [2, 3], [1, 2], [1, 3], [2, 4], [3, 5],  # 面部
    [4, 6], [5, 7]  # 肩膀連接
]

# 關鍵點名稱
KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

# 顏色定義
RED = (0, 0, 255)        # 紅色 (BGR)
GREEN = (0, 255, 0)      # 綠色
BLUE = (255, 0, 0)       # 藍色
YELLOW = (0, 255, 255)   # 黃色
PURPLE = (255, 0, 255)   # 紫色
ORANGE = (0, 165, 255)   # 橙色
BLACK = (0, 0, 0)        # 黑色
WHITE = (255, 255, 255)  # 白色

# 置信度閾值，用於繪製關鍵點連線
KEYPOINT_THRESHOLD = 0.4  # 提高關鍵點置信度閾值，低於此值將不會繪製

def draw_pose(image, keypoints, box=None, class_id=0, conf=None, color_box=GREEN, color_keypoint=RED, color_skeleton=BLUE, thickness=2, radius=4):
    """
    繪製人體姿勢關鍵點和骨架
    
    Args:
        image: OpenCV圖像
        keypoints: 關鍵點數據 [x, y, conf] 格式
        box: 邊界框 [x, y, w, h] 格式 (如果提供)
        class_id: 類別ID
        conf: 類別置信度
        color_box: 邊界框顏色
        color_keypoint: 關鍵點顏色
        color_skeleton: 骨架連接線顏色
        thickness: 線條粗細
        radius: 關鍵點半徑
    """
    h, w = image.shape[:2]
    
    # 繪製邊界框（如果提供）
    if box is not None:
        # 將正規化的邊界框座標轉換回像素座標
        x, y, width, height = box
        x_min = int((x - width/2) * w)
        y_min = int((y - height/2) * h)
        x_max = int((x + width/2) * w)
        y_max = int((y + height/2) * h)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color_box, thickness)
        
        # 在邊界框上方繪製類別和置信度
        if conf is not None:
            label = f"person {conf:.2f}"
            # 計算文字大小
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            # 繪製背景矩形
            cv2.rectangle(image, (x_min, y_min - t_size[1] - 5), (x_min + t_size[0], y_min), color_box, -1)
            # 使用抗鋸齒繪製文字
            cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 1, cv2.LINE_AA)
    
    # 將正規化的關鍵點轉換為像素坐標
    valid_kpts_dict = {}  # 使用字典儲存有效的關鍵點，索引為鍵
    for i, kpt in enumerate(keypoints):
        x, y, conf = kpt
        # 避免異常值，確保座標在合理範圍內
        if conf > KEYPOINT_THRESHOLD and 0 <= x <= 1 and 0 <= y <= 1:
            # 將正規化座標轉換為像素座標
            x_px = int(x * w)
            y_px = int(y * h)
            
            # 確保不是(0,0)座標附近 (過濾可能的錯誤檢測)
            if not (x_px < 5 and y_px < 5):
                valid_kpts_dict[i] = (x_px, y_px, conf)
    
    # 繪製骨架連接線
    for p1_idx, p2_idx in SKELETON:
        # 調整為0索引
        p1_idx -= 1
        p2_idx -= 1
        
        # 檢查兩個關鍵點是否都存在於有效字典中
        if p1_idx in valid_kpts_dict and p2_idx in valid_kpts_dict:
            p1 = valid_kpts_dict[p1_idx][:2]  # 只取坐標，忽略置信度
            p2 = valid_kpts_dict[p2_idx][:2]
            
            # 再次檢查確保不是連到(0,0)附近
            if not ((p1[0] < 5 and p1[1] < 5) or (p2[0] < 5 and p2[1] < 5)):
                cv2.line(image, p1, p2, color_skeleton, thickness, cv2.LINE_AA)
                
                # 顯示連線兩端點的索引（如果是測試模式）
                if TEST_MODE:
                    # 在連線中間位置添加標籤
                    mid_x = (p1[0] + p2[0]) // 2
                    mid_y = (p1[1] + p2[1]) // 2
                    conn_label = f"{p1_idx+1}-{p2_idx+1}"
                    
                    # 如果需要標記連線，可以取消下面的註釋
                    # cv2.putText(image, conn_label, (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, YELLOW, 1, cv2.LINE_AA)
    
    # 繪製關鍵點
    for idx, (x, y, conf) in valid_kpts_dict.items():
        # 繪製關鍵點圓圈
        cv2.circle(image, (x, y), radius, color_keypoint, -1, cv2.LINE_AA)
        
        # 添加關鍵點索引標籤
        if TEST_MODE:
            # 索引從0開始，顯示時加1與COCO標準一致
            cv2.putText(image, str(idx+1), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
    
    return image

def main():
    print("Loading YOLOv11x-pose model...")
    # Load the YOLOv11x-pose model
    model = YOLO("yolo11x-pose.pt")
    
    # Get all image files
    image_files = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    print(f"Found {len(image_files)} images to process.")
    
    # For TEST_MODE, just process one image
    if TEST_MODE:
        print("TEST_MODE is enabled. Only processing 1000 images.")
        image_files = image_files[:1000]
    
    # Create a list to track images with multiple poses
    multi_pose_images = []
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Get the base filename without extension
        base_name = Path(img_path).stem
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
        pred_path = os.path.join(PRED_DIR, f"{base_name}.jpg")
        
        # Skip if output file already exists (except in TEST_MODE)
        if os.path.exists(output_path) and not TEST_MODE:
            continue
        
        # 讀取原始圖片
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Error: Cannot read image {img_path}")
            continue
        
        # 創建一個副本用於繪製
        visualization_img = original_img.copy()
        
        # 運行推理
        results = model.predict(
            img_path, 
            save=False,  # 不自動保存
            verbose=False,
            conf=0.25,  # 置信度閾值
        )
        
        # 從results獲取圖片尺寸
        img_height, img_width = results[0].orig_shape
        
        # In TEST_MODE, print detection results to log
        if TEST_MODE:
            print(f"\n===== Detection Results for {img_path} =====")
            print(f"Image dimensions: {img_width}x{img_height}")
            for i, r in enumerate(results):
                print(f"Result {i+1}:")
                if r.boxes is not None:
                    print(f"  Found {len(r.boxes)} bounding boxes")
                    # 打印邊界框的置信度
                    for j, box in enumerate(r.boxes):
                        print(f"    Box {j+1}: class={box.cls.item()}, confidence={box.conf.item():.4f}")
                if r.keypoints is not None:
                    print(f"  Found {len(r.keypoints)} sets of keypoints")
                    
                    # Print detailed keypoint information for the first pose
                    if len(r.keypoints) > 0:
                        print("\n  Detailed Keypoints for Pose 1:")
                        kpts = r.keypoints[0].data[0]
                        for j, (name, kpt) in enumerate(zip(KEYPOINT_NAMES, kpts)):
                            # 顯示原始和正規化後的座標
                            norm_x = float(kpt[0]) / img_width
                            norm_y = float(kpt[1]) / img_height
                            print(f"    {j+1}. {name}: x={kpt[0]:.4f} (norm: {norm_x:.4f}), y={kpt[1]:.4f} (norm: {norm_y:.4f}), confidence={kpt[2]:.4f}")
            print("=" * 50)
        
        # Track number of poses detected in this image
        pose_count = 0
        
        # Process each detection and draw on visualization image
        # Create output file
        with open(output_path, "w") as f:
            # Process each detection
            for r in results:
                boxes = r.boxes
                keypoints = r.keypoints
                
                if keypoints is None or len(keypoints) == 0:
                    continue
                
                # Process each detection with keypoints
                for i in range(len(keypoints)):
                    if i >= len(boxes):
                        continue
                    
                    # Increment pose count
                    pose_count += 1
                        
                    # Get bounding box coordinates (normalized)
                    box = boxes[i].xywhn[0].tolist()
                    
                    # 獲取類別ID和置信度
                    class_id = int(boxes[i].cls.item()) if boxes[i].cls is not None else 0
                    confidence = float(boxes[i].conf.item()) if boxes[i].conf is not None else None
                    
                    # Get keypoints data
                    kpts = keypoints[i].data[0]
                    
                    # 準備正規化的關鍵點數據用於繪製
                    normalized_kpts = []
                    for kpt in kpts:
                        # 正規化 x,y 座標
                        norm_x = float(kpt[0]) / img_width
                        norm_y = float(kpt[1]) / img_height
                        normalized_kpts.append([norm_x, norm_y, kpt[2]])
                    
                    # 使用不同顏色繪製每個姿勢
                    color_idx = i % 3  # 在三種顏色間循環
                    if color_idx == 0:
                        box_color = GREEN
                        kpt_color = RED
                        skeleton_color = BLUE
                    elif color_idx == 1:
                        box_color = YELLOW
                        kpt_color = PURPLE
                        skeleton_color = ORANGE
                    else:
                        box_color = BLUE
                        kpt_color = GREEN
                        skeleton_color = RED
                    
                    # 在可視化圖像上繪製姿勢
                    draw_pose(
                        visualization_img, 
                        normalized_kpts, 
                        box=box, 
                        class_id=class_id,
                        conf=confidence,
                        color_box=box_color,
                        color_keypoint=kpt_color, 
                        color_skeleton=skeleton_color,
                        thickness=2,
                        radius=4
                    )
                    
                    # Start line with class id (0 for person)
                    line = ["0"]
                    
                    # Add bounding box coordinates (normalized)
                    line.extend([f"{v:.6f}" for v in box])
                    
                    # Add keypoints (x, y, confidence) - 正規化座標
                    for kpt in kpts:
                        if kpt[2] > 0:  # Visible keypoint
                            # 正規化 x,y 座標
                            norm_x = float(kpt[0]) / img_width
                            norm_y = float(kpt[1]) / img_height
                            line.extend([f"{norm_x:.6f}", f"{norm_y:.6f}", f"{kpt[2]:.6f}"])
                        else:  # Not visible
                            line.extend(["0.000000", "0.000000", "0.000000"])
                    
                    # Write line to file
                    f.write(" ".join(line) + "\n")
        
        # 將姿勢標註後的圖像保存到PRED_DIR
        cv2.imwrite(pred_path, visualization_img)
        
        if TEST_MODE:
            print(f"Visualization saved to: {pred_path}")
        
        # Check if multiple poses were detected
        if pose_count > 1:
            multi_pose_images.append(img_path)
    
    # Write the list of images with multiple poses to output.txt
    with open("output.txt", "w") as f:
        for img_path in multi_pose_images:
            f.write(f"{img_path}\n")
    
    print(f"Processing complete. Results saved to {OUTPUT_DIR}")
    print(f"Visualizations saved to {PRED_DIR}")
    print(f"Found {len(multi_pose_images)} images with multiple poses. List saved to output.txt")

if __name__ == "__main__":
    main() 