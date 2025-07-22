import os
import shutil
from tqdm import tqdm
import cv2

def convert_visdrone_to_yolo(img_path, txt_path, save_path, class_mapping):
    """
    将VisDrone数据集转换为YOLO格式
    Args:
        img_path: VisDrone图片路径
        txt_path: VisDrone标注文件路径
        save_path: 保存YOLO格式数据的路径
        class_mapping: 类别映射字典
    """
    # 创建保存目录
    os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels'), exist_ok=True)

    # 获取所有标注文件
    annotation_files = [f for f in os.listdir(txt_path) if f.endswith('.txt')]

    for ann_file in tqdm(annotation_files, desc="Converting annotations"):
        img_file = ann_file.replace('.txt', '.jpg')
        
        # 读取图片获取尺寸
        img_path_full = os.path.join(img_path, img_file)
        if not os.path.exists(img_path_full):
            print(f"Warning: Image {img_file} not found, skipping...")
            continue
            
        img = cv2.imread(img_path_full)
        if img is None:
            print(f"Warning: Cannot read image {img_file}, skipping...")
            continue
            
        img_height, img_width = img.shape[:2]

        # 复制图片到新目录
        shutil.copy2(
            img_path_full,
            os.path.join(save_path, 'images', img_file)
        )

        # 读取标注文件
        with open(os.path.join(txt_path, ann_file), 'r') as f:
            lines = f.readlines()

        # 转换标注格式
        yolo_lines = []
        for line in lines:
            data = line.strip().split(',')
            if len(data) < 8:  # 确保数据格式正确
                continue

            x, y, w, h = map(float, data[:4])
            score = float(data[4])
            category_id = int(data[5])
            
            # 跳过score为0的标注
            if score == 0:
                continue
                
            # 跳过不在类别映射中的标注
            if category_id not in class_mapping:
                continue

            # 转换为YOLO格式：<class> <x_center> <y_center> <width> <height>
            x_center = (x + w/2) / img_width
            y_center = (y + h/2) / img_height
            width = w / img_width
            height = h / img_height

            # 确保坐标在[0,1]范围内
            x_center = min(max(x_center, 0), 1)
            y_center = min(max(y_center, 0), 1)
            width = min(max(width, 0), 1)
            height = min(max(height, 0), 1)

            yolo_line = f"{class_mapping[category_id]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            yolo_lines.append(yolo_line)

        # 保存YOLO格式的标注文件
        with open(os.path.join(save_path, 'labels', ann_file), 'w') as f:
            f.writelines(yolo_lines)

def main():
    # VisDrone类别映射到YOLO格式
    class_mapping = {
        1: 0,  # pedestrian
        2: 1,  # people
        3: 2,  # bicycle
        4: 3,  # car
        5: 4,  # van
        6: 5,  # truck
        7: 6,  # tricycle
        8: 7,  # awning-tricycle
        9: 8,  # bus
        10: 9  # motor
    }

    # 设置路径
    base_path = r"D:\Data\UAV\SkyToSurface\VisDrone2019\VisDrone2019-DET-train"  # 根据实际路径修改
    img_path = os.path.join(base_path, "images")
    txt_path = os.path.join(base_path, "annotations")
    save_path = os.path.join(base_path, "yolo_labels")
    # save_path = "VisDrone2019-YOLO"

    # 转换数据集
    convert_visdrone_to_yolo(img_path, txt_path, save_path, class_mapping)

    # 创建classes.txt文件
    classes = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    with open(os.path.join(save_path, 'classes.txt'), 'w') as f:
        f.write('\n'.join(classes))

if __name__ == "__main__":
    main()
