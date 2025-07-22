import os
import shutil
import json
import cv2
from tqdm import tqdm   


"""
    功能描述：将YOLO格式转换为自定义数据格式
    输入：
        label_path: 标注路径
        height: 图像高度
        width: 图像宽度
    输出：
        json文件
"""
def transform_labelme_to_custom(label_path,height,width):

    label_list = ["single_plate","double_plate"]
    # label_list = ['fire', 'smoke']
    results = []  # 存储转换后的数据
    # 读取 labelme 文件内容
    with open(label_path, 'r', encoding='utf-8') as file:
        # if file is None:
        #     print("file is None")
        #     return None
        content = file.read()
        if len(content) == 0:
            # print("content is None")
            return None
        file.seek(0)  # 将文件指针移动到文件开头
        for line in file:

            parts = line.split()

            # 如果是 YOLO 格式，假设格式是: class_id center_x center_y width height
            class_id = int(parts[0])  # 类别编号
            x_center = float(parts[1])*width  # 物体中心 x 坐标 (0-1 标准化)
            y_center = float(parts[2])*height  # 物体中心 y 坐标 (0-1 标准化)
            obj_width = float(parts[3])*width  # 物体宽度 (0-1 标准化)
            obj_height = float(parts[4])*height  # 物体高度 (0-1 标准化)
            area = obj_width * obj_height
            # 计算边界框的坐标（通过将 YOLO 格式的中心坐标转换为顶点坐标）
            xmin = x_center - obj_width / 2
            ymin = y_center - obj_height / 2
            
            # xmax = (x_center + obj_width / 2) * width
            # ymax = (y_center + obj_height / 2) * height

            # 将数据格式转换为所需的格式
            custom_data = {
                "name": label_list[class_id],
                "polygon_points": [],  # YOLO 数据没有多边形数据，留空
                "bnd_points": [xmin, ymin, obj_width, obj_height],
                "key_point": []  # YOLO 数据没有关键点，留空
            }
            results.append(custom_data)  # 将每个 label 转换后的数据加入到结果中

    transform_data = {
        "labelList":results,
        "itemType": 0,
        "background": False,
        "markType": 0,
        "checked": False,
        "addTrain": 0,
        "area": area
        }
    return transform_data


def batch_transform(input_dir,output_dir):

    os.makedirs(output_dir,exist_ok=True)
    labels_dir = os.path.join(input_dir,"yolo_labels")
    images_dir = os.path.join(input_dir,"images")
    
    for filename in tqdm(os.listdir(labels_dir),desc="transforming files",unit="files"):
        try:
            label_path = os.path.join(labels_dir,filename)
            image_path = os.path.join(images_dir,filename.replace(".txt",".jpg"))
            im = cv2.imread(image_path)
            h,w = im.shape[:2]
            # 读取json数据


            custom_data = transform_labelme_to_custom(label_path,h,w)
            transform_path = os.path.join(output_dir,filename.replace(".txt",".json"))
            if custom_data is None:
                continue
            with open(transform_path,'w',encoding='utf-8') as file:
                json.dump(custom_data,file)
        except Exception as e:
            # 如果发生异常，打印错误并跳过该文件
            print(f"Failed to process {filename}: {e}")
            continue


if __name__ == "__main__":
    input_dir = r'D:\Data\car_plate\plate_det\detect_plate_datasets\val_detect\val_detect\val_detect'
    output_dir = r'D:\Data\car_plate\plate_det\detect_plate_datasets\val_detect\val_detect\val_detect\labels'
    batch_transform(input_dir,output_dir)