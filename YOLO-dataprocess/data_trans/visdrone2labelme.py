import os
import json
import base64
import cv2
from tqdm import tqdm

def get_image_binary(image_path):
    """
    读取图像并转换为base64编码
    """
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

def convert_visdrone_to_labelme(img_path, txt_path, save_path):
    """
    将VisDrone数据集转换为LabelMe格式
    Args:
        img_path: VisDrone图片路径
        txt_path: VisDrone标注文件路径
        save_path: 保存LabelMe格式数据的路径
    """
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # VisDrone类别映射
    categories = {
        1: "pedestrian",
        2: "people",
        3: "bicycle",
        4: "car",
        5: "van",
        6: "truck",
        7: "tricycle",
        8: "awning-tricycle",
        9: "bus",
        10: "motor"
    }

    # 获取所有标注文件
    annotation_files = [f for f in os.listdir(txt_path) if f.endswith('.txt')]

    for ann_file in tqdm(annotation_files, desc="Converting to LabelMe format"):
        img_file = ann_file.replace('.txt', '.jpg')
        
        # 读取图片
        img_path_full = os.path.join(img_path, img_file)
        if not os.path.exists(img_path_full):
            print(f"Warning: Image {img_file} not found, skipping...")
            continue

        # 获取图片尺寸
        img = cv2.imread(img_path_full)
        if img is None:
            print(f"Warning: Cannot read image {img_file}, skipping...")
            continue
            
        img_height, img_width = img.shape[:2]

        # 获取图片的二进制数据
        image_data = get_image_binary(img_path_full)

        # 读取标注文件
        with open(os.path.join(txt_path, ann_file), 'r') as f:
            lines = f.readlines()

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

        # 转换标注信息
        for line in lines:
            data = line.strip().split(',')
            if len(data) < 8:
                continue

            x, y, w, h = map(float, data[:4])
            score = float(data[4])
            category_id = int(data[5])
            
            # 跳过score为0的标注
            if score == 0:
                continue
                print(f"score: {score} is 0")
            # 跳过不在类别映射中的标注
            if category_id not in categories:
                continue
                print(f"category_id: {category_id} not in categories")
            # 跳过score小于0.5的标注

            # 创建矩形标注
            shape = {
                "label": categories[category_id],
                "points": [
                    [x, y],  # 左上角
                    [x + w, y + h]  # 右下角
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",
                "flags": {}
            }
            
            labelme_data["shapes"].append(shape)

        # 保存为JSON文件
        json_file = os.path.join(save_path, ann_file.replace('.txt', '.json'))
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

def main():
    # 设置路径
    base_path = r"D:\Data\UAV\SkyToSurface\VisDrone2019\VisDrone2019-DET-val"  # 根据实际路径修改
    img_path = os.path.join(base_path, "images")
    txt_path = os.path.join(base_path, "annotations")
    save_path = os.path.join(base_path, "labelme")
    # save_path = "VisDrone2019-LabelMe"

    # 转换数据集
    convert_visdrone_to_labelme(img_path, txt_path, save_path)

if __name__ == "__main__":
    main()
