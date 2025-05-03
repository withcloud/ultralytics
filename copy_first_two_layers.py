# copy_first_two_layers.py
import torch
from ultralytics import YOLO

def copy_initial_layers():
    # 加載模型
    yolo_model = YOLO('yolo11n-pose.pt')
    gde_model = YOLO('super_phase5_1_v2/weights/best.pt')
    
    # 獲取模型權重
    yolo_state_dict = yolo_model.model.state_dict()
    gde_state_dict = gde_model.model.state_dict()
    
    # 僅複製前兩層(0和1)的權重
    copied_count = 0
    initial_layers = []
    
    # 建立要複製的層清單
    for name in gde_state_dict.keys():
        if any(part in name for part in ['model.0.', 'model.1.']):
            initial_layers.append(name)
    
    # 複製權重
    for name in initial_layers:
        if name in yolo_state_dict and gde_state_dict[name].shape == yolo_state_dict[name].shape:
            gde_state_dict[name].copy_(yolo_state_dict[name])
            copied_count += 1
            print(f"複製層: {name}, 形狀: {gde_state_dict[name].shape}")
    
    print(f"共複製了 {copied_count} 個參數")
    
    # 載入修改後的權重
    gde_model.model.load_state_dict(gde_state_dict)
    
    # 保存模型
    gde_model.save('gde_yolo_initial_layers.pt')
    print("模型已保存為 gde_yolo_initial_layers.pt")
    
    return gde_model

# 執行
model = copy_initial_layers()

# 評估模型
metrics = model.val(data="coco-pose.yaml")
print(f"mAP@50: {metrics.box.map50}")
print(f"Pose Keypoint mAP: {metrics.pose.map}")
