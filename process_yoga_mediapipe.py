#!/usr/bin/env python3
# Script to process yoga82 dataset images with MediaPipe pose model and save results in specific format
import os
import glob
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import mediapipe as mp
import csv

# Define paths
INPUT_DIR = "datasets/yoga82/images/train"
OUTPUT_DIR = "datasets/yoga82/labels_mediapipe/train"
PRED_DIR = "datasets/yoga82/preds_mediapipe/train"
CSV_OUTPUT = "datasets/yoga82/results_mediapipe.csv"

TEST_MODE = False

# 新增人體置信度過濾閾值
POSE_CONFIDENCE_THRESHOLD = 0.9  # 只處理平均置信度超過此值的檢測結果

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)

# COCO 關鍵點索引與 MediaPipe 關鍵點的映射關係
# COCO 格式有 17 個關鍵點:
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
#
# MediaPipe Pose 有 33 個關鍵點，我們需要映射到 COCO 的 17 個關鍵點
MEDIAPIPE_TO_COCO = {
    0: 0,   # nose
    2: 1,   # left_eye
    5: 2,   # right_eye
    7: 3,   # left_ear
    8: 4,   # right_ear
    11: 5,  # left_shoulder
    12: 6,  # right_shoulder
    13: 7,  # left_elbow
    14: 8,  # right_elbow
    15: 9,  # left_wrist
    16: 10, # right_wrist
    23: 11, # left_hip
    24: 12, # right_hip
    25: 13, # left_knee
    26: 14, # right_knee
    27: 15, # left_ankle
    28: 16  # right_ankle
}

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
KEYPOINT_THRESHOLD = 0.7  # 關鍵點置信度閾值，低於此值將不會繪製

def convert_mediapipe_to_coco(landmarks, image_width, image_height):
    """
    將 MediaPipe 的姿勢關鍵點轉換為 COCO 格式
    
    Args:
        landmarks: MediaPipe 檢測到的關鍵點
        image_width: 圖像寬度
        image_height: 圖像高度
        
    Returns:
        coco_keypoints: COCO 格式的關鍵點 (每個關鍵點為 [x, y, visibility])
    """
    coco_keypoints = []
    
    # 為每個 COCO 關鍵點創建一個條目
    for coco_idx in range(17):
        # 對於每個 COCO 索引，查找對應的 MediaPipe 索引
        mp_idx = None
        for mp_key, coco_val in MEDIAPIPE_TO_COCO.items():
            if coco_val == coco_idx:
                mp_idx = mp_key
                break
        
        # 如果找到了對應的 MediaPipe 索引，轉換坐標
        if mp_idx is not None and mp_idx < len(landmarks):
            # MediaPipe 坐標是相對坐標 (0-1)
            landmark = landmarks[mp_idx]
            
            # 取得正規化後的 x, y 坐標
            x = landmark.x
            y = landmark.y
            
            # 使用可見性作為置信值 (MediaPipe 的 visibility 在 0-1 之間)
            visibility = landmark.visibility
            
            # 確保坐標在範圍內
            x = max(0, min(1, x))
            y = max(0, min(1, y))
            
            coco_keypoints.append([x, y, visibility])
        else:
            # 如果找不到對應的 MediaPipe 索引，設置為不可見
            coco_keypoints.append([0, 0, 0])
    
    return coco_keypoints

def draw_pose(image, keypoints, box=None, class_id=0, conf=None, visibility_ratio=None, color_box=GREEN, color_keypoint=RED, color_skeleton=BLUE, thickness=2, radius=4):
    """
    繪製人體姿勢關鍵點和骨架
    
    Args:
        image: OpenCV圖像
        keypoints: 關鍵點數據 [x, y, conf] 格式
        box: 邊界框 [x, y, w, h] 格式 (如果提供)
        class_id: 類別ID
        conf: 類別置信度
        visibility_ratio: 可見關鍵點比例
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
            # 使用4位小數顯示置信度
            if visibility_ratio is not None:
                label = f"MediaPipe {conf:.4f} (vis: {visibility_ratio:.2f})"
            else:
                label = f"MediaPipe {conf:.4f}"
                
            # 計算文字大小
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            # 繪製背景矩形
            cv2.rectangle(image, (x_min, y_min - t_size[1] - 5), (x_min + t_size[0], y_min), color_box, -1)
            # 使用抗鋸齒繪製文字
            cv2.putText(image, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
            
            # 調試輸出
            if TEST_MODE:
                print(f"Confidence value passed to draw_pose: {conf}")
                print(f"Label displayed on image: {label}")
    
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
    
    # 繪製關鍵點
    for idx, (x, y, conf) in valid_kpts_dict.items():
        # 繪製關鍵點圓圈
        cv2.circle(image, (x, y), radius, color_keypoint, -1, cv2.LINE_AA)
        
        # 添加關鍵點索引標籤
        if TEST_MODE:
            # 索引從0開始，顯示時加1與COCO標準一致
            cv2.putText(image, str(idx+1), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1, cv2.LINE_AA)
    
    return image

def compute_bounding_box(keypoints):
    """
    根據關鍵點計算邊界框
    
    Args:
        keypoints: 關鍵點列表 [x, y, visibility]
        
    Returns:
        bounding_box: [x_center, y_center, width, height] 格式的邊界框
    """
    valid_points = []
    for point in keypoints:
        x, y, visibility = point
        if visibility > KEYPOINT_THRESHOLD:
            valid_points.append((x, y))
    
    if not valid_points:
        return None
    
    # 計算最小和最大坐標
    x_coords, y_coords = zip(*valid_points)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # 計算中心點和尺寸
    width = x_max - x_min
    height = y_max - y_min
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    # 添加一些padding，確保邊界框包含所有關鍵點
    width *= 1.1
    height *= 1.1
    
    # 確保邊界框在範圍內
    x_center = max(width/2, min(1 - width/2, x_center))
    y_center = max(height/2, min(1 - height/2, y_center))
    width = min(width, 1.0)
    height = min(height, 1.0)
    
    return [x_center, y_center, width, height]

def save_results_to_csv(results_data, output_path, pass_threshold_count, total_count):
    """
    將處理結果儲存為CSV檔案
    
    Args:
        results_data: 包含圖片檔名、路徑、置信值和其他資訊的列表
        output_path: CSV檔案輸出路徑
        pass_threshold_count: 通過置信度閾值的數量
        total_count: 總處理圖片數量
    """
    with open(output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 寫入標題行
        csv_writer.writerow([
            'filename', 
            'image_path', 
            'confidence_score', 
            'min_keypoint_conf', 
            'max_keypoint_conf',
            'visible_keypoints',
            'visibility_ratio',
            'detection_threshold',
            'model_complexity'
        ])
        # 寫入數據
        for data in results_data:
            csv_writer.writerow(data)
    
    print(f"CSV file saved to: {output_path}")
    print("Note: MediaPipe doesn't provide explicit box confidence scores like YOLO.")
    print("      The confidence_score is calculated as the average visibility of ALL keypoints (including invisible ones).")
    print(f"Confidence threshold: {POSE_CONFIDENCE_THRESHOLD}")
    print(f"Total processed images: {total_count}")
    print(f"Images passing threshold: {pass_threshold_count} ({pass_threshold_count/total_count*100:.2f}% pass rate)")

def main():
    print("Initializing MediaPipe Pose model...")
    # 初始化 MediaPipe Pose 模型
    detection_threshold = 0.5  # 保存使用的檢測閾值
    model_complexity = 2       # 保存使用的模型複雜度
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,        # 圖片模式
        model_complexity=model_complexity,            # 最高精度模型
        enable_segmentation=False,     # 不需要分割
        min_detection_confidence=detection_threshold   # 最小檢測閾值
    )
    
    # Get all image files
    image_files = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    print(f"Found {len(image_files)} images to process.")
    
    # For TEST_MODE, just process one image
    if TEST_MODE:
        print("TEST_MODE is enabled. Only processing 1000 images.")
        image_files = image_files[:1000]
    
    # Create a list to track images with multiple poses
    multi_pose_images = []
    
    # Create a list to store results for CSV
    csv_results = []
    
    # 追蹤統計資訊
    total_processed = 0
    passed_threshold = 0
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Get the base filename without extension
        base_name = Path(img_path).stem
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
        pred_path = os.path.join(PRED_DIR, f"{base_name}.jpg")
        
        # Skip if output file already exists (except in TEST_MODE)
        if os.path.exists(output_path) and not TEST_MODE:
            continue
        
        # 更新總處理數
        total_processed += 1
        
        # 讀取原始圖片
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Error: Cannot read image {img_path}")
            continue
        
        # MediaPipe 需要 RGB 格式圖像
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img_rgb.shape[:2]
        
        # 創建一個副本用於繪製
        visualization_img = original_img.copy()
        
        # 處理圖像
        results = pose.process(img_rgb)
        
        # 確保檢測到了姿勢關鍵點
        if not results.pose_landmarks:
            print(f"No pose detected in {img_path}")
            continue
        
        # 將 MediaPipe 關鍵點轉換為 COCO 格式
        coco_keypoints = convert_mediapipe_to_coco(
            results.pose_landmarks.landmark,
            img_width,
            img_height
        )
        
        # 計算邊界框
        bbox = compute_bounding_box(coco_keypoints)
        if bbox is None:
            print(f"Failed to compute bounding box for {img_path}")
            continue
        
        # 計算有關關鍵點可見性的統計數據
        # 原有方法：只考慮可見的關鍵點
        # visibilities = [kpt[2] for kpt in coco_keypoints if kpt[2] > 0]
        # 修改後：考慮所有關鍵點，包括不可見的（置信度為0）
        visibilities = [kpt[2] for kpt in coco_keypoints]
        
        if len(visibilities) == 0:
            print(f"No keypoints in {img_path}")
            continue
            
        # 計算可見關鍵點的數據
        visible_points = [v for v in visibilities if v > 0]
        if len(visible_points) == 0:
            print(f"No visible keypoints in {img_path}")
            continue
            
        min_visibility = min(visible_points)
        max_visibility = max(visible_points)
        visible_count = len(visible_points)
        
        # 平均置信度 - 使用所有關鍵點計算，包括置信度為0的
        confidence = sum(visibilities) / len(visibilities)
        
        # 計算可見關鍵點比例
        visibility_ratio = visible_count / len(visibilities)
        
        # 添加結果到CSV數據 - 增加了更多資訊，無論置信度高低都記錄
        csv_results.append([
            base_name, 
            img_path, 
            f"{confidence:.6f}",
            f"{min_visibility:.6f}",
            f"{max_visibility:.6f}",
            visible_count,
            f"{visibility_ratio:.6f}",
            detection_threshold,
            model_complexity
        ])
        
        # 檢查人體置信度是否超過閾值
        if confidence < POSE_CONFIDENCE_THRESHOLD:
            if TEST_MODE:
                print(f"Skipping {img_path} - confidence {confidence:.4f} below threshold {POSE_CONFIDENCE_THRESHOLD}")
            continue
        
        # 更新通過閾值的計數
        passed_threshold += 1
        
        # In TEST_MODE, print detection results to log
        if TEST_MODE:
            print(f"\n===== Detection Results for {img_path} =====")
            print(f"Image dimensions: {img_width}x{img_height}")
            print(f"Computed bounding box: {bbox}")
            print(f"Estimated confidence: {confidence:.4f} (Passed threshold: {POSE_CONFIDENCE_THRESHOLD})")
            
            # Print detailed keypoint information
            print("\n  Detailed Keypoints:")
            for j, (name, kpt) in enumerate(zip(KEYPOINT_NAMES, coco_keypoints)):
                print(f"    {j+1}. {name}: x={kpt[0]:.4f}, y={kpt[1]:.4f}, visibility={kpt[2]:.4f}")
            print("=" * 50)
        
        # 在可視化圖像上繪製姿勢
        draw_pose(
            visualization_img, 
            coco_keypoints, 
            box=bbox, 
            conf=confidence,
            visibility_ratio=visibility_ratio,
            color_box=GREEN,
            color_keypoint=RED, 
            color_skeleton=BLUE,
            thickness=2,
            radius=4
        )
        
        # 在可視化圖像上添加更多信息
        info_text = f"MediaPipe conf: {confidence:.4f}"
        cv2.putText(visualization_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
        
        info_text2 = f"Visible kpts: {visible_count}/{len(visibilities)} ({visibility_ratio:.2f})"
        cv2.putText(visualization_img, info_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
        
        # 標記閾值信息
        threshold_text = f"Threshold: {POSE_CONFIDENCE_THRESHOLD}"
        cv2.putText(visualization_img, threshold_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLUE, 2, cv2.LINE_AA)
        
        # 創建輸出文件
        with open(output_path, "w") as f:
            # 類別 ID (0 for person)
            line = ["0"]
            
            # 添加邊界框座標 (正規化)
            line.extend([f"{v:.6f}" for v in bbox])
            
            # 添加關鍵點 (x, y, visibility)
            for kpt in coco_keypoints:
                x, y, visibility = kpt
                if visibility > 0:  # 可見關鍵點
                    line.extend([f"{x:.6f}", f"{y:.6f}", f"{visibility:.6f}"])
                else:  # 不可見關鍵點
                    line.extend(["0.000000", "0.000000", "0.000000"])
            
            # 寫入文件
            f.write(" ".join(line) + "\n")
        
        # 保存可視化圖像
        cv2.imwrite(pred_path, visualization_img)
        
        if TEST_MODE:
            print(f"Visualization saved to: {pred_path}")
        
        # 暫時不考慮多姿勢情況，MediaPipe 每張圖只檢測一個姿勢
    
    # 儲存CSV結果
    save_results_to_csv(csv_results, CSV_OUTPUT, passed_threshold, total_processed)
    
    # 釋放資源
    pose.close()
    
    print(f"\n===== 處理結果摘要 =====")
    print(f"總共找到的圖片: {len(image_files)}")
    print(f"成功處理的圖片: {total_processed}")
    print(f"通過閾值 ({POSE_CONFIDENCE_THRESHOLD}) 的圖片: {passed_threshold}")
    print(f"通過率: {passed_threshold/total_processed*100:.2f}%")
    print(f"標籤文件保存至: {OUTPUT_DIR}")
    print(f"可視化圖片保存至: {PRED_DIR}")
    print(f"統計數據CSV保存至: {CSV_OUTPUT}")
    print("=" * 30)

if __name__ == "__main__":
    main() 