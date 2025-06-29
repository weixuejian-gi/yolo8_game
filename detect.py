from ultralytics import YOLO
import cv2

# # 1. 加载训练好的模型
# model = YOLO("runs/detect/train/weights/best.pt")  # 替换为你的路径

# # 2. 单张图片预测
# results = model.predict("data/images/val/test2.png", exist_ok=True, save=True, conf=0.1)

model = YOLO('data/yolov8n.pt')
results = model.predict('data/images/val/test3.png', exist_ok=True, save=True, conf=0.1) 
# results = model.predict('data/images/val/test.jpg', save=True, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) 

# 3. 可视化结果（可选）
for r in results:
    im_array = r.plot()  # 绘制检测框
    cv2.imshow("YOLOv8 Detection", im_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 4. 获取检测信息
for result in results:
    print("检测到的对象：")
    for box in result.boxes:
        print(f"类别: {result.names[box.cls[0].item()]} | 置信度: {box.conf[0].item():.2f}")
        print(f"坐标: {box.xyxy[0].tolist()}")  # [x1,y1,x2,y2]