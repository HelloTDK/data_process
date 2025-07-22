from ultralytics import YOLO
import math
import os
import time
import cv2
import json
import numpy as np

"""
    功能描述：自动标注，通过YOLO模型对图像进行自动标注，并保存为我们自定义的数据格式
    输入：
        model_path: 模型路径
        imgs_path: 图像路径
        labels_path: 标注路径
        labels: 标注类别
    输出：
        json文件
    
"""

def autolabels(model_path,imgs_path,labels_path,labels=None):
    # 加载模型
    model = YOLO(model_path)
    # 获取图像列表
    img_names = os.listdir(imgs_path)
    # 遍历所有图像
    os.makedirs(labels_path,exist_ok=True)
    for i,img_name in enumerate(img_names):
        # 如果不是图片则跳过
        if img_name.lower().endswith(('.jpg','.png','jpeg')):
            t1 = time.time()
            img_path = os.path.join(imgs_path,img_name)
            im = cv2.imread(img_path)
            infer_results = ultralytics_infer(model,im,labels)
            # json_info = json.dumps(infer_results)
            label_name = img_name.replace('.jpg','.json')
            label_path = os.path.join(labels_path,label_name)
            if infer_results:
                with open(label_path,'w') as file:
                    json.dump(infer_results,file)
        

def ultralytics_infer(model,source,lables=None):
    # 获取所有类别
    class_names = model.names
    # 模型推理
    results = model.predict(source)
    infer_results = []
    # 遍历模型检测结果
    for result in results:
        if result.masks is not None:
            masks = result.masks.xy
        else:
            masks = [None]*len(result.boxes)
        for i,box in enumerate(result.boxes):
            class_id = int(box.cls[0].item())
            name = class_names[class_id]
            if lables:
                if name not in lables:
                    continue
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            w = x2 - x2
            h = y2 - y1
            if masks is not None:
                polygon_points = masks[i].tolist() if masks[i] is not None else []
            confidence = box.conf[0].item()
            infer_result = {
            "class_id":class_id,
            "name" : name,
            "bnd_points":[x1,y1,w,h]
            # "polygon_points": polygon_points
            }
            if polygon_points:
                infer_result["polygon_points"] = polygon_points
            infer_results.append(infer_result)
        return infer_results

if __name__ == "__main__":

    model_path = r"D:\train_results\yolo\person_detect\yolov8_person9.27\weights\best.pt"
    imgs_path = r"D:\Data\人体识别\2\8.30-9.26"
    labels_path = r"D:\Data\人体识别\2\8.30-9.26_labels"
    autolabels(model_path,imgs_path,labels_path)