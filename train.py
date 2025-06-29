from ultralytics import YOLO
import os

# 写入数据集配置
os.makedirs("data", exist_ok=True)

# 加载模型（使用预训练的YOLOv8n）
model = YOLO("data/yolov8n.pt")  # 会自动下载模型

# 训练参数配置
results = model.train(
    data="data/dataset.yaml",  # 数据集配置
    epochs=50,
    batch=8,
    imgsz=640,
    device="cpu",  # 使用GPU（如"cpu"则用CPU）
    name="train",
    optimizer="Adam",  # 可选优化器
    lr0=0.001,        # 初始学习率
    exist_ok=True,  # 允许覆盖
)

# 验证模型
metrics = model.val()  # 在验证集上评估
print(f"mAP50-95: {metrics.box.map}")  # 打印精度