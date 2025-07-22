import os
import cv2
import json
import tqdm
import glob
import shutil
import random
import string
import re

class AnyLabel2YOLO:
    def __init__(self, label_path, image_path, output_path, get_dataset_yaml=False, label_type=1, convert_chinese=False):
        self.label_path = label_path
        self.image_path = image_path
        self.output_path = output_path
        self.get_dataset_yaml = get_dataset_yaml
        self.label_list = {}
        self.label_type = label_type # 1: 矩形框 2: 多边形 3: 点 4: 线
        self.convert_chinese = convert_chinese  # 是否转换中文文件名
        self.filename_mapping = {}  # 用于记录文件名映射关系

        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        
        # 自动生成yolo数据保存路径
        self.yolo_label_path = os.path.join(output_path, "labels")
        self.yolo_image_path = os.path.join(output_path, "images")
        os.makedirs(self.yolo_label_path, exist_ok=True)
        os.makedirs(self.yolo_image_path, exist_ok=True)
        
        # 自动生成dataset.yaml路径
        self.dataset_yaml_path = os.path.join(output_path, "dataset.yaml")

    def _has_chinese(self, text):
        """检查文本是否包含中文字符"""
        pattern = re.compile(r'[\u4e00-\u9fff]')
        return bool(pattern.search(text))

    def _generate_english_filename(self, original_name):
        """生成随机英文文件名"""
        # 获取文件扩展名
        name, ext = os.path.splitext(original_name)
        
        # 如果已经映射过，直接返回
        if original_name in self.filename_mapping:
            return self.filename_mapping[original_name]
        
        # 生成随机英文文件名
        while True:
            # 生成6-12位随机字母数字组合
            length = random.randint(6, 12)
            new_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
            new_filename = new_name + ext
            
            # 确保文件名不重复
            if new_filename not in self.filename_mapping.values():
                self.filename_mapping[original_name] = new_filename
                return new_filename

    def _transform(self, label_file):
        """
            Args: 
                label_file: 标签的路径
            return: 
                None
        """
        norm_bnd_points = []
        
        try:
            with open(label_file, "r", encoding='utf-8') as f:
                label_data = json.load(f)
        except UnicodeDecodeError:
            with open(label_file, "r", encoding='gbk') as f:
                label_data = json.load(f)
        
        label_infos = label_data["shapes"]
        # 获取图像尺寸
        image_height = label_data["imageHeight"]
        image_width = label_data["imageWidth"]
        
        # 获取原始图像文件名
        original_image_name = label_data.get("imagePath", "")
        if not original_image_name:
            # 如果json中没有imagePath，尝试根据label文件名推断
            label_name = os.path.basename(label_file)
            name_without_ext = os.path.splitext(label_name)[0]
            # 尝试常见的图像格式
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                potential_image = name_without_ext + ext
                if os.path.exists(os.path.join(self.image_path, potential_image)):
                    original_image_name = potential_image
                    break

        for label_info in label_infos:
            label_name = label_info["label"]
            label_points = label_info["points"]
            
            if label_name not in self.label_list:
                self.label_list[label_name] = len(self.label_list)
            label_index = self.label_list[label_name]
            
            # 矩形框处理
            if self.label_type == 1:
                if len(label_points) >= 2:
                    left_top = label_points[0] 
                    right_bottom = label_points[1] if len(label_points) == 2 else label_points[2]
                    
                    center_x = (left_top[0] + right_bottom[0]) / 2
                    center_y = (left_top[1] + right_bottom[1]) / 2
                    box_width = abs(right_bottom[0] - left_top[0])
                    box_height = abs(right_bottom[1] - left_top[1])
                    
                    bnd_point = [center_x, center_y, box_width, box_height]
                    norm_bnd_point = self._normalize(bnd_point, image_height, image_width, label_index)
                    norm_bnd_points.append(norm_bnd_point)
            
            # 多边形处理
            elif self.label_type == 2:
                if len(label_points) >= 3:
                    normalized_points = []
                    for point in label_points:
                        norm_x = point[0] / image_width
                        norm_y = point[1] / image_height
                        normalized_points.extend([norm_x, norm_y])
                    
                    norm_bnd_point = [label_index] + normalized_points
                    norm_bnd_points.append(norm_bnd_point)

        # 处理文件名
        label_basename = os.path.basename(label_file)
        label_name_without_ext = os.path.splitext(label_basename)[0]
        
        # 如果启用中文转换且文件名包含中文
        if self.convert_chinese and (self._has_chinese(label_basename) or self._has_chinese(original_image_name)):
            # 重要：确保图像和标签使用相同的基础英文名称
            base_name_key = label_name_without_ext  # 使用标签文件的基础名作为键
            
            # 如果还没有为这个基础名生成英文名，就生成一个
            if base_name_key not in self.filename_mapping:
                # 生成随机英文基础名（不包含扩展名）
                while True:
                    length = random.randint(6, 12)
                    new_base_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
                    # 确保基础名不重复
                    if not any(existing_name.startswith(new_base_name) for existing_name in self.filename_mapping.values()):
                        self.filename_mapping[base_name_key] = new_base_name
                        break
            
            # 使用相同的基础名生成标签和图像文件名
            new_base_name = self.filename_mapping[base_name_key]
            new_label_name = new_base_name + ".json"
            
            if original_image_name:
                original_image_ext = os.path.splitext(original_image_name)[1]
                new_image_name = new_base_name + original_image_ext
                # 记录图像文件的映射关系
                self.filename_mapping[original_image_name] = new_image_name
            else:
                new_image_name = ""
            
            print(f"转换文件名: {label_basename} -> {new_label_name}")
            if original_image_name:
                print(f"转换图像名: {original_image_name} -> {new_image_name}")
                print(f"✓ 确保匹配: {new_base_name}.txt ↔ {new_base_name}{original_image_ext}")
        else:
            new_label_name = label_basename
            new_image_name = original_image_name
            new_base_name = label_name_without_ext

        # 保存YOLO格式标签（使用相同的基础名）
        yolo_label_name = new_base_name + ".txt"
        yolo_label_path = os.path.join(self.yolo_label_path, yolo_label_name)
        self._write_yolo_label(yolo_label_path, norm_bnd_points)

        # 复制并重命名图像文件
        if original_image_name and new_image_name:
            original_image_path = os.path.join(self.image_path, original_image_name)
            new_image_path = os.path.join(self.yolo_image_path, new_image_name)
            
            if os.path.exists(original_image_path):
                shutil.copy2(original_image_path, new_image_path)
                print(f"复制图像: {original_image_path} -> {new_image_path}")
            else:
                print(f"⚠️  警告: 图像文件不存在 - {original_image_path}")
        elif original_image_name:
            print(f"⚠️  警告: 无法确定图像文件的新名称 - {original_image_name}")

    def _write_yolo_label(self, write_path, norm_bnd_points):
        with open(write_path, "w") as f:
            for norm_bnd_point in norm_bnd_points:
                if self.label_type == 1:  # 矩形框
                    norm_data = " ".join([str(x) for x in norm_bnd_point])
                else:  # 多边形
                    norm_data = " ".join([str(x) for x in norm_bnd_point])
                f.write(norm_data + "\n")
    
    def _normalize(self, bnd_point, image_height, image_width, label_idx):
        return [label_idx, bnd_point[0]/image_width, bnd_point[1]/image_height, 
                bnd_point[2]/image_width, bnd_point[3]/image_height]

    def _create_dataset_yaml(self):
        """创建dataset.yaml文件"""
        if self.get_dataset_yaml:
            yaml_content = f"""# YOLO数据集配置文件
path: {os.path.abspath(self.output_path)}
train: images
val: images

# 类别数量
nc: {len(self.label_list)}

# 类别名称
names: {list(self.label_list.keys())}
"""
            with open(self.dataset_yaml_path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            print(f"已创建dataset.yaml文件: {self.dataset_yaml_path}")

    def _save_filename_mapping(self):
        """保存文件名映射关系"""
        if self.convert_chinese and self.filename_mapping:
            mapping_file = os.path.join(self.output_path, "filename_mapping.json")
            
            # 创建详细的映射信息
            detailed_mapping = {
                "说明": "原始文件名到英文文件名的映射关系",
                "重要提示": "图像文件和标签文件使用相同的基础英文名称，只有扩展名不同",
                "映射关系": self.filename_mapping,
                "文件对应示例": {
                    "中文图片.jpg": "abc123.jpg",
                    "中文图片.json": "abc123.json", 
                    "输出": {
                        "标签文件": "abc123.txt",
                        "图像文件": "abc123.jpg"
                    }
                }
            }
            
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_mapping, f, ensure_ascii=False, indent=2)
            print(f"已保存文件名映射: {mapping_file}")
            print("✓ 映射文件包含了详细的对应关系说明")

    def batch_transform(self):
        """批量转换标签文件"""
        label_files = glob.glob(os.path.join(self.label_path, "*.json"))
        
        if not label_files:
            print("未找到JSON标签文件!")
            return
        
        print(f"找到 {len(label_files)} 个标签文件")
        
        for label_file in tqdm.tqdm(label_files, desc="转换进度"):
            try:
                self._transform(label_file)
            except Exception as e:
                print(f"处理文件 {label_file} 时出错: {e}")
        
        # 创建dataset.yaml
        self._create_dataset_yaml()
        
        # 保存文件名映射
        self._save_filename_mapping()
        
        print(f"\n转换完成!")
        print(f"标签文件保存在: {self.yolo_label_path}")
        print(f"图像文件保存在: {self.yolo_image_path}")
        print(f"共处理了 {len(self.label_list)} 个类别: {list(self.label_list.keys())}")


if __name__ == "__main__":
    # 使用示例
    converter = AnyLabel2YOLO(
        label_path=r"D:\Code\Python\wb\xy480\data\Desktop\all",  # labelme标注文件夹
        image_path=r"D:\Code\Python\wb\xy480\data\Desktop\all",  # 图像文件夹
        output_path=r"D:\Code\Python\wb\xy480\data\Desktop\yolo",  # 输出文件夹
        get_dataset_yaml=True,  # 是否生成dataset.yaml
        label_type=1,  # 1: 矩形框, 2: 多边形
        convert_chinese=True  # 是否转换中文文件名为英文
    )
    converter.batch_transform() 

