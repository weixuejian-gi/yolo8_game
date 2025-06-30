import cv2
import numpy as np
import pyautogui
from mss import mss
from ultralytics import YOLO
import time
import math

class ElementController:
    def __init__(self):
        # 初始化设置
        pyautogui.PAUSE = 0.05  # 控制响应速度
        self.model = YOLO('data/yolov8n.pt')  # 替换为你的训练好的模型
        
        # 屏幕捕获设置
        self.monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        
        # 定义你的两个特定元素的类别ID (根据训练模型调整)
        self.TARGET_CLASS_1 = 0  # 第一个目标类别ID
        self.TARGET_CLASS_2 = 1  # 第二个目标类别ID
        
        # 控制键设置 (根据实际应用调整)
        self.CONTROLS = {
            'element1': {'up': 'w', 'down': 's', 'left': 'a', 'right': 'd'},
            'element2': {'up': 'i', 'down': 'k', 'left': 'j', 'right': 'l'}
        }
        
        # 控制参数
        self.MIN_DISTANCE = 50  # 认为已经靠近的距离阈值(像素)
        self.CONFIDENCE_THRESHOLD = 0.05  # 检测置信度阈值
    
    def get_detections(self, screenshot):
        """使用YOLO模型检测屏幕中的目标元素"""
        # 将BGRA转换为BGR（去掉Alpha通道）
        screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        results = self.model(screenshot_bgr)  # 使用转换后的图像
        elements = {'class1': [], 'class2': []}
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < self.CONFIDENCE_THRESHOLD:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1
                
                detection = {
                    'position': (center_x, center_y),
                    'confidence': conf,
                    'size': (width, height),
                    'bbox': (x1, y1, x2, y2)
                }
                
                if cls == self.TARGET_CLASS_1:
                    elements['class1'].append(detection)
                elif cls == self.TARGET_CLASS_2:
                    elements['class2'].append(detection)
        
        return elements
    
    def select_specific_elements(self, elements):
        """从检测到的多个元素中选择两个特定目标"""
        # 这里实现你的特定选择逻辑
        # 示例1: 选择置信度最高的两个目标
        if len(elements['class1']) > 0 and len(elements['class2']) > 0:
            # 对每个类别按置信度排序
            elements['class1'].sort(key=lambda x: -x['confidence'])
            elements['class2'].sort(key=lambda x: -x['confidence'])
            return elements['class1'][0], elements['class2']
        
        if len(elements['class1']) > 0:
            return elements['class1'][0], None
        
        else:
            return None, None
    
    def calculate_movement(self, pos1, pos2):
        """计算两个元素应该移动的方向"""
        x1, y1 = pos1
        x2, y2 = pos2
        
        # 计算中间点
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # 元素1需要移动的方向向量
        dx1 = mid_x - x1
        dy1 = mid_y - y1
        
        return (dx1, dy1)
    
    def control_element(self, control_set, dx, dy):
        """根据方向向量发送键盘指令"""
        # 确定主要移动方向
        if abs(dx) > abs(dy):
            key = control_set['right'] if dx > 0 else control_set['left']
        else:
            key = control_set['down'] if dy > 0 else control_set['up']
        
        # 发送控制指令
        pyautogui.keyDown(key)
        time.sleep(0.05)
        pyautogui.keyUp(key)
    
    def move_elements_together(self, element1, element2):
        """控制两个元素相互靠近"""
        pos1 = element1['position']
        pos2 = element2['position']
        
        # 计算当前距离
        distance = math.dist(pos1, pos2)
        if distance < self.MIN_DISTANCE:
            return True
        
        # 计算移动方向
        (dx1, dy1) = self.calculate_movement(pos1, pos2)
        
        # 控制元素1移动
        self.control_element(self.CONTROLS['element1'], dx1, dy1)
        
        return False
    
    def visualize(self, screenshot, element1, element2):
        """可视化检测结果和控制过程"""
        if element1:
            pos1 = element1['position']
            # 绘制元素边界框
            x1, y1, x2, y2 = element1['bbox']
            cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if element2:
            pos2 = element2[0]['position']
            for element in element2:
                x1, y1, x2, y2 = element['bbox']
                cv2.rectangle(screenshot, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if element1 and element2:
            # 绘制中心点和连接线
            cv2.circle(screenshot, pos1, 5, (0, 255, 0), -1)
            cv2.circle(screenshot, pos2, 5, (0, 0, 255), -1)
            cv2.line(screenshot, pos1, pos2, (255, 0, 0), 2)
            
            # 显示距离
            distance = math.dist(pos1, pos2)
            cv2.putText(screenshot, f"Distance: {distance:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return screenshot
    
    def run(self):
        """主控制循环"""
        with mss() as sct:
            while True:
                # 捕获屏幕
                screenshot = np.array(sct.grab(self.monitor))
                # 检测所有目标元素
                elements = self.get_detections(screenshot)
                # 选择两个特定元素
                element1, element2 = self.select_specific_elements(elements)
                
                if element1 and element2:
                    # 控制元素相互靠近
                    done = self.move_elements_together(element1, element2[0])
                    if done:
                        print("两个目标元素已经足够接近!")
                        # 可以在这里添加靠近后的操作
                        # break  # 或者继续控制其他元素
                
                # 可视化结果
                screenshot = self.visualize(screenshot, element1, element2)
                cv2.imshow('Element Controller', screenshot)
                # 5. 退出条件（按 'b' 或 ESC 退出）
                key = cv2.waitKey(1)
                if key in (ord("b"), 27):  # 27 是 ESC 键
                    break

if __name__ == "__main__":
    controller = ElementController()
    controller.run()