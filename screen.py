import cv2
import numpy as np
from mss import mss
from ultralytics import YOLO

# 加载模型（替换成你的自定义模型或官方模型）
model = YOLO("data/yolov8n.pt")  # 或者 "best.pt"

# 设置屏幕捕获区域（全屏或自定义区域）
monitor = {
    "top": 0,
    "left": 0,
    "width": 1920,  # 修改为你的屏幕分辨率
    "height": 1080,
}

# 创建单个窗口（设置窗口名称，并允许调整大小）
cv2.namedWindow("YOLOv8 Real-Time Screen Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Real-Time Screen Detection", 800, 600)  # 初始窗口大小

with mss() as sct:
    while True:
        # 1. 截取屏幕
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 转换颜色通道

        # 2. YOLOv8 推理（stream=True 优化实时性能）
        results = model(img, stream=True)  

        # 3. 绘制检测结果
        for r in results:
            img = r.plot()  # 直接在图像上绘制检测框

        # 4. 更新窗口（替换上一帧）
        cv2.imshow("YOLOv8 Real-Time Screen Detection", img)

        # 5. 退出条件（按 'q' 或 ESC 退出）
        key = cv2.waitKey(1)
        if key in (ord("q"), 27):  # 27 是 ESC 键
            break

# 释放资源
cv2.destroyAllWindows()