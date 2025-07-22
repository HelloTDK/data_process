import os
import json
import base64
import cv2
from pathlib import Path
from tqdm import tqdm

def get_image_binary(image_path):
    """
    读取图像并转换为base64编码
    """
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

def convert_yolo_to_labelme(img_dir, label_dir, save_dir, class_names):
    """
    将YOLO格式数据转换为LabelMe格式
    
    Args:
        img_dir (str): 图片目录路径
        label_dir (str): YOLO格式标签目录路径
        save_dir (str): 保存LabelMe格式数据的路径
        class_names (list): 类别名称列表，索引对应YOLO中的类别ID
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取所有标签文件
    label_files = list(Path(label_dir).glob('*.txt'))
    
    for label_file in tqdm(label_files, desc="Converting to LabelMe format"):
        # 获取对应的图片文件
        img_file = label_file.stem + '.jpg'  # 默认jpg格式，如果有其他格式需要修改
        img_path = os.path.join(img_dir, img_file)
        
        # 检查图片是否存在
        if not os.path.exists(img_path):
            # 尝试其他常见图片格式
            for ext in ['.png', '.jpeg', '.bmp']:
                img_path = os.path.join(img_dir, label_file.stem + ext)
                if os.path.exists(img_path):
                    img_file = label_file.stem + ext
                    break
            else:
                print(f"Warning: Image for {label_file.name} not found, skipping...")
                continue
        
        # 读取图片获取尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot read image {img_file}, skipping...")
            continue
        
        img_height, img_width = img.shape[:2]
        
        # 获取图片的二进制数据
        image_data = get_image_binary(img_path)
        
        # 创建LabelMe格式的数据结构
        labelme_data = {
            "version": "5.4.1",
            "flags": {},
            "shapes": [],
            "imagePath": img_file,
            "imageData": image_data,
            "imageHeight": img_height,
            "imageWidth": img_width
        }
        
        # 读取YOLO格式的标注
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 转换每个标注
        for line in lines:
            data = line.strip().split()
            if len(data) != 5:  # YOLO格式：class_id center_x center_y width height
                continue
            
            class_id = int(data[0])
            if class_id >= len(class_names):
                print(f"Warning: Class ID {class_id} exceeds class_names length, skipping...")
                continue
            
            # YOLO坐标是归一化的，需要转换回像素坐标
            center_x = float(data[1]) * img_width
            center_y = float(data[2]) * img_height
            width = float(data[3]) * img_width
            height = float(data[4]) * img_height
            
            # 计算矩形的左上角和右下角坐标
            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2
            
            # 创建矩形标注
            shape = {
                "label": class_names[class_id],
                "points": [
                    [x1, y1],  # 左上角
                    [x2, y2]   # 右下角
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            }
            
            labelme_data["shapes"].append(shape)
        
        # 保存为JSON文件
        json_file = os.path.join(save_dir, label_file.stem + '.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

def main():
    # 设置路径
    base_dir = r"D:\Data\car_plate\plate_det\detect_plate_datasets\train_data1\train_data1\train_data"  # 请修改为您的数据集路径
    img_dir = os.path.join(base_dir, "images")
    label_dir = os.path.join(base_dir, "labels")
    save_dir = os.path.join(base_dir, "labelme")
    
    # 设置类别名称，请按照YOLO类别ID的顺序填写
    class_names = [
        "single_plate",  # 类别ID 0
        "double_plate",  # 类别ID 1
        # 添加更多类别...
    ]
    
    # 转换数据集
    convert_yolo_to_labelme(img_dir, label_dir, save_dir, class_names)
    print("转换完成！")

if __name__ == "__main__":
    main()
