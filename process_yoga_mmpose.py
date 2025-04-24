#!/usr/bin/env python3
# Script to process yoga82 dataset images with MMPose model and save results in specific format
import os
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmpose.evaluation.functional import nms
from mmdet.apis import inference_detector, init_detector
from mmpose.utils import adapt_mmdet_pipeline

# Register all MMPose modules
register_all_modules()

# Register all MMDetection modules 
from mmdet.utils import register_all_modules as register_all_mmdet_modules
register_all_mmdet_modules()

# Define paths
INPUT_DIR = "datasets/yoga82/images/train"
OUTPUT_DIR = "datasets/yoga82/labels_mmpose/train"
PRED_DIR = "datasets/yoga82/preds_mmpose/train"

# Change to False for processing the entire dataset
TEST_MODE = True

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

# Model configuration
DET_CONFIG = '/Users/region/withcloud/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'  # MMDetection 配置文件
DET_CHECKPOINT = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'  # RTMPose项目的人体检测模型
POSE_CONFIG = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
POSE_CHECKPOINT = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

# Detection parameters
BBOX_THR = 0.3  # 边界框置信度阈值
NMS_THR = 0.3   # NMS 阈值
DET_CAT_ID = 0  # 人体类别ID (COCO中)

# COCO 關鍵點索引定義
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

# 關鍵點連接線的定義 (人體骨架)
SKELETON = [
    # 腿部
    [15, 13], [13, 11],  # 左腿
    [16, 14], [14, 12],  # 右腿
    
    # 躯干
    [11, 12],  # 胯部连接
    [5, 11], [6, 12],  # 肩膀到胯部
    [5, 6],  # 肩膀连接
    
    # 手臂
    [5, 7], [7, 9],  # 左臂
    [6, 8], [8, 10],  # 右臂
    
    # 面部
    [0, 1], [0, 2],  # 鼻子到眼睛
    [1, 3], [2, 4],  # 眼睛到耳朵
    [0, 5], [0, 6]   # 鼻子到肩膀
]

# 關鍵點名稱
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# 置信度閾值，用於繪製關鍵點連線
KPT_CONF_THRESHOLD = 0.15  # 關鍵點置信度閾值，低於此值將不會繪製

# 顏色定義
RED = (0, 0, 255)        # 紅色 (BGR)
GREEN = (0, 255, 0)      # 綠色
BLUE = (255, 0, 0)       # 藍色
YELLOW = (0, 255, 255)   # 黃色
PURPLE = (255, 0, 255)   # 紫色
ORANGE = (0, 128, 255)   # 橙色
BLACK = (0, 0, 0)        # 黑色
WHITE = (255, 255, 255)  # 白色

def draw_pose(image, keypoints, box=None, conf=None, color_box=GREEN, color_keypoint=RED, color_skeleton=BLUE, thickness=2, radius=4):
    """
    在图像上绘制姿势估计结果
    
    Args:
        image: 输入图像 (BGR格式)
        keypoints: 归一化的关键点 [x, y, conf] 列表
        box: 归一化的边界框 [x_center, y_center, width, height]
        conf: 边界框的置信度
        color_box: 边界框的颜色
        color_keypoint: 关键点的颜色
        color_skeleton: 骨架的颜色
        thickness: 线条粗细
        radius: 关键点圆点半径
    """
    h, w = image.shape[:2]
    
    # 绘制边界框
    if box is not None:
        x_center, y_center, width, height = box
        x1 = int((x_center - width/2) * w)
        y1 = int((y_center - height/2) * h)
        x2 = int((x_center + width/2) * w)
        y2 = int((y_center + height/2) * h)
        
        # 绘制矩形
        cv2.rectangle(image, (x1, y1), (x2, y2), color_box, thickness)
        
        # 绘制置信度标签
        if conf is not None:
            label = f"{conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
            cv2.rectangle(image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), color_box, -1)
            cv2.putText(image, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, thickness-1)
    
    # 将归一化坐标转换为图像坐标
    image_keypoints = []
    for kpt in keypoints:
        x, y, conf = kpt
        image_keypoints.append([int(x * w), int(y * h), conf])
    
    # 绘制骨架线条
    for connection in SKELETON:
        idx1, idx2 = connection[0], connection[1]  # 直接使用索引值
        
        # 检查关键点可见性
        if idx1 < len(image_keypoints) and idx2 < len(image_keypoints):
            if image_keypoints[idx1][2] > KPT_CONF_THRESHOLD and image_keypoints[idx2][2] > KPT_CONF_THRESHOLD:
                pt1 = (image_keypoints[idx1][0], image_keypoints[idx1][1])
                pt2 = (image_keypoints[idx2][0], image_keypoints[idx2][1])
                cv2.line(image, pt1, pt2, color_skeleton, thickness)
    
    # 绘制关键点
    for i, kpt in enumerate(image_keypoints):
        x, y, conf = kpt
        if conf > KPT_CONF_THRESHOLD:  # 只绘制置信度较高的关键点
            cv2.circle(image, (x, y), radius, color_keypoint, -1)
            
            # 如果是测试模式，显示关键点索引
            if TEST_MODE:
                cv2.putText(image, f"{i}", (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, BLACK, 1)
    
    return image

def parse_args():
    parser = argparse.ArgumentParser(description='Process yoga images with MMPose')
    parser.add_argument('--input', default='datasets/yoga82/images/train', help='Input directory containing images')
    parser.add_argument('--output', default='datasets/yoga82/labels_mmpose/train', help='Output directory for label files')
    parser.add_argument('--pred', default='datasets/yoga82/preds_mmpose/train', help='Directory for visualization images')
    parser.add_argument('--test', action='store_true', help='Enable test mode (process only 10 images)')
    parser.add_argument('--bbox-thr', type=float, default=0.3, help='Bounding box threshold')
    parser.add_argument('--nms-thr', type=float, default=0.3, help='NMS threshold')
    parser.add_argument('--kpt-thr', type=float, default=0.15, help='Keypoint confidence threshold')
    parser.add_argument('--device', default='', help='Device to run inference (cuda:0, cpu)')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 更新全局设置
    global INPUT_DIR, OUTPUT_DIR, PRED_DIR, TEST_MODE, BBOX_THR, NMS_THR, KPT_CONF_THRESHOLD
    
    INPUT_DIR = args.input
    OUTPUT_DIR = args.output
    PRED_DIR = args.pred
    TEST_MODE = args.test
    BBOX_THR = args.bbox_thr
    NMS_THR = args.nms_thr
    KPT_CONF_THRESHOLD = args.kpt_thr
    
    # 设置设备
    device = args.device
    if not device:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    
    print("Initializing MMDetection and MMPose models...")
    
    # 检查配置文件是否存在
    if not os.path.exists(DET_CONFIG):
        print(f"Error: Det config file {DET_CONFIG} not found")
        return
    
    # 初始化目标检测模型
    detector = init_detector(
        DET_CONFIG, 
        DET_CHECKPOINT, 
        device=device
    )
    print("Detector initialized successfully")
    
    # 适配MMDetection管道以在MMPose中使用
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    
    # 初始化姿勢估測模型
    pose_estimator = init_model(
        POSE_CONFIG, 
        POSE_CHECKPOINT, 
        device=device
    )
    print("Pose estimator initialized successfully")
    
    # Get all image files
    image_files = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    print(f"Found {len(image_files)} images to process.")
    
    # For TEST_MODE, just process limited images
    if TEST_MODE:
        print("TEST_MODE is enabled. Only processing 10 images.")
        image_files = image_files[:10]
    
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
        
        img_height, img_width = original_img.shape[:2]
        
        # 創建一個副本用於繪製
        visualization_img = original_img.copy()
        
        try:
            # 使用目标检测模型检测人体
            det_result = inference_detector(detector, original_img)
            pred_instance = det_result.pred_instances.cpu().numpy()
            
            # 过滤检测结果
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == DET_CAT_ID,
                                         pred_instance.scores > BBOX_THR)]
            
            if len(bboxes) == 0:
                print(f"No person detected in {img_path}")
                
                # 使用整个图像作为默认边界框
                default_bbox = np.array([[0, 0, img_width, img_height, 1.0]])
                bboxes = default_bbox[:, :4]
            else:
                # 应用NMS过滤重叠框
                bboxes = bboxes[nms(bboxes, NMS_THR), :4]
                if TEST_MODE:
                    print(f"Detected {len(bboxes)} persons in {img_path}")
            
            # 進行姿勢估測
            pose_results = inference_topdown(pose_estimator, original_img, bboxes)
            
            if not pose_results:
                print(f"No pose detected in {img_path}")
                continue
            
            # 寫入檢測結果
            with open(output_path, "w") as f:
                for i, pose_result in enumerate(pose_results):
                    # 獲取關鍵點 - 获取预测的关键点
                    keypoints = pose_result.pred_instances.keypoints[0]
                    keypoint_scores = pose_result.pred_instances.keypoint_scores[0]
                    
                    # 检查数据类型并转换为numpy数组
                    if hasattr(keypoints, 'cpu'):
                        keypoints = keypoints.cpu().numpy()
                    if hasattr(keypoint_scores, 'cpu'):
                        keypoint_scores = keypoint_scores.cpu().numpy()
                    
                    # 获取对应的边界框
                    bbox = bboxes[i]
                    bbox_score = pred_instance.scores[i] if i < len(pred_instance.scores) else 1.0
                    
                    # 將邊界框轉換為YOLO格式 [x_center, y_center, width, height]
                    x1, y1, x2, y2 = bbox
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    x_center = (x1 + (x2 - x1) / 2) / img_width
                    y_center = (y1 + (y2 - y1) / 2) / img_height
                    bbox_yolo = [x_center, y_center, width, height]
                    
                    # 將關鍵點轉換為正規化座標
                    normalized_keypoints = []
                    for j, (kpt, kpt_score) in enumerate(zip(keypoints, keypoint_scores)):
                        x, y = kpt
                        x_norm = x / img_width
                        y_norm = y / img_height
                        # 確保座標在0-1範圍內
                        x_norm = max(0, min(1, x_norm))
                        y_norm = max(0, min(1, y_norm))
                        normalized_keypoints.append([x_norm, y_norm, kpt_score])
                    
                    # 在可視化圖像上繪製姿勢
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
                    
                    # 绘制边界框和关键点
                    draw_pose(
                        visualization_img, 
                        normalized_keypoints, 
                        box=bbox_yolo, 
                        conf=bbox_score,
                        color_box=box_color,
                        color_keypoint=kpt_color, 
                        color_skeleton=skeleton_color,
                        thickness=2,
                        radius=4
                    )
                    
                    # In TEST_MODE, print detection results to log
                    if TEST_MODE:
                        print(f"\n===== Detection Results for {img_path} =====")
                        print(f"Image dimensions: {img_width}x{img_height}")
                        print(f"Person {i+1}: bbox confidence: {bbox_score:.4f}")
                        print(f"Bbox: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
                        print(f"Normalized: x_center={bbox_yolo[0]:.4f}, y_center={bbox_yolo[1]:.4f}, width={bbox_yolo[2]:.4f}, height={bbox_yolo[3]:.4f}")
                        
                        # Print detailed keypoint information
                        print("\n  Detailed Keypoints:")
                        for j, (name, kpt) in enumerate(zip(KEYPOINT_NAMES, normalized_keypoints)):
                            print(f"    {j+1}. {name}: x={kpt[0]:.4f}, y={kpt[1]:.4f}, confidence={kpt[2]:.4f}")
                        print("=" * 50)
                    
                    # 寫入YOLO格式的姿勢數據
                    # Start line with class id (0 for person)
                    line = ["0"]
                    
                    # Add bounding box coordinates (normalized)
                    line.extend([f"{v:.6f}" for v in bbox_yolo])
                    
                    # Add keypoints (x, y, confidence)
                    for kpt in normalized_keypoints:
                        if kpt[2] > 0:  # Visible keypoint
                            line.extend([f"{kpt[0]:.6f}", f"{kpt[1]:.6f}", f"{kpt[2]:.6f}"])
                        else:  # Not visible
                            line.extend(["0.000000", "0.000000", "0.000000"])
                    
                    # Write line to file
                    f.write(" ".join(line) + "\n")
            
            # 將姿勢標註後的圖像保存到PRED_DIR
            cv2.imwrite(pred_path, visualization_img)
            
            if TEST_MODE:
                print(f"Visualization saved to: {pred_path}")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Processing complete. Results saved to {OUTPUT_DIR}")
    print(f"Visualizations saved to {PRED_DIR}")

if __name__ == "__main__":
    main() 