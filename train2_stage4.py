from ultralytics import YOLO

# Load the model from the third training stage
model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train43/weights/last.pt")  # Using the last weights from third stage

# Train the model with optimized fine-tuning parameters
results = model.train(
    data="coco-pose.yaml",
    epochs=100,
    imgsz=1280,        # Keep high resolution
    batch=128,         # Balanced batch size for high resolution
    save_period=1,     # Save every epoch as requested
    cache="disk",      # Use disk caching
    optimizer="AdamW", # Continue with AdamW optimizer
    lr0=0.00005,       # Further reduced learning rate for fine-tuning
    lrf=0.01,          # Final LR factor
    cos_lr=True,       # Cosine LR scheduler
    device="0,1",      # Use both GPUs
    patience=25,       # Early stopping patience
    box=12.0,          # Increased box loss weight
    pose=18.0,         # Increased pose loss weight
    kobj=4.0,          # Increased keypoint objectness weight
    multi_scale=True,  # Enable multi-scale training for better generalization
    close_mosaic=10    # Close mosaic in final epochs
)
