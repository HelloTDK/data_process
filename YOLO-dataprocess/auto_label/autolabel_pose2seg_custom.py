import os
import json
import cv2
import numpy as np
import torch
import random
from ultralytics import YOLO
from pathlib import Path

# 设置目录
images_dir = "images"
labels_dir = "labels"
os.makedirs(labels_dir, exist_ok=True)

# 加载YOLO-Pose模型
model = YOLO("yolo-pose.pt")

# 生成随机颜色
def random_color():
    return f"#{random.randint(0, 255):02X}{random.randint(0, 255):02X}{random.randint(0, 255):02X}"

# 创建config.json
config = {
    "labelList": [
        {
            "labelId": "person",
            "labelName": "person",
            "labelColor": random_color()
        }
    ],
    "shape": 1,
    "classes": ["person"]
}

with open("config.json", "w") as f:
    json.dump(config, f)

# 处理图像文件
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)
    
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像: {img_path}")
        continue
    
    # 获取图像尺寸
    height, width = img.shape[:2]
    
    # 使用YOLO-Pose进行推理
    results = model(img)
    
    label_data = {
        "labelList": [],
        "itemType": 0,
        "background": False,
        "itemStatus": 1,
        "markType": 0,
        "checked": True,
        "addTrain": 0
    }
    
    for result in results:
        if result.keypoints is None:
            continue
            
        keypoints = result.keypoints.data
        boxes = result.boxes.data
        
        for i, (kpts, box) in enumerate(zip(keypoints, boxes)):
            if box[5] == 0:  # 确保是人类
                kpts = kpts.cpu().numpy()
                box = box.cpu().numpy()
                
                # 获取边界框
                x1, y1, x2, y2 = box[:4]
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                
                # 计算宽度和高度
                w = x2 - x1
                h = y2 - y1
                
                # 使用关键点创建多边形
                valid_kpts = kpts[kpts[:, 2] > 0]
                
                if len(valid_kpts) >= 4:
                    # 选择一些关键点作为多边形顶点
                    # 这里简单选择4个点，实际应用中可能需要更复杂的逻辑
                    polygon_points = []
                    indices = [0, 5, 11, 15]  # 例如：鼻子，左肩，左髋，左脚踝
                    
                    for idx in indices:
                        if idx < len(kpts) and kpts[idx, 2] > 0:
                            polygon_points.append([float(kpts[idx, 0]), float(kpts[idx, 1])])
                        else:
                            # 如果关键点不可用，使用边界框的角点
                            corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                            polygon_points = corners[:4]
                            break
                else:
                    # 如果没有足够的有效关键点，使用边界框的角点
                    polygon_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                
                # 计算多边形面积
                area = 0
                for j in range(len(polygon_points)):
                    x1, y1 = polygon_points[j]
                    x2, y2 = polygon_points[(j + 1) % len(polygon_points)]
                    area += (x1 * y2 - x2 * y1)
                area = abs(area) / 2
                
                person_data = {
                    "polygon_points": polygon_points,
                    "markArea": area,
                    "name": "person",
                    "bnd_points": [x1, y1, w, h],
                    "key_point": []
                }
                
                label_data["labelList"].append(person_data)
    
    # 保存标签文件
    label_file = os.path.splitext(img_file)[0] + ".json"
    label_path = os.path.join(labels_dir, label_file)
    
    with open(label_path, "w") as f:
        json.dump(label_data, f)
    
    print(f"处理完成: {img_file} -> {label_file}")

print("所有图像处理完成，标签已保存到labels目录")
