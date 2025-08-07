#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnyLabel2YOLO训练验证集生成器
将任意标注格式转换为YOLO格式并自动分割为训练集和验证集
"""

import os
import cv2
import json
import tqdm
import glob
import shutil
import random
import string
import re
import argparse
from pathlib import Path
from typing import List, Tuple


class AnyLabel2YOLOTrainVal:
    def __init__(self, label_path, image_path, output_path, 
                 label_type=1, convert_chinese=False, 
                 train_ratio=0.8, random_seed=42):
        """
        初始化标注转换和数据集分割器
        
        Args:
            label_path: 标注文件路径
            image_path: 图片文件路径  
            output_path: 输出路径
            label_type: 标注类型 (1: 矩形框, 2: 多边形)
            convert_chinese: 是否转换中文文件名
            train_ratio: 训练集比例
            random_seed: 随机种子
        """
        self.label_path = label_path
        self.image_path = image_path
        self.output_path = output_path
        self.label_list = {}
        self.label_type = label_type
        self.convert_chinese = convert_chinese
        self.filename_mapping = {}
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        
        # 设置随机种子
        random.seed(random_seed)
        
        # 创建输出目录结构
        self._create_output_directories()
        
        # 临时转换目录
        self.temp_dir = os.path.join(output_path, "temp_conversion")
        self.temp_label_path = os.path.join(self.temp_dir, "labels")
        self.temp_image_path = os.path.join(self.temp_dir, "images")
        os.makedirs(self.temp_label_path, exist_ok=True)
        os.makedirs(self.temp_image_path, exist_ok=True)

    def _create_output_directories(self):
        """创建输出目录结构"""
        dirs = [
            'train/images',
            'train/labels', 
            'val/images',
            'val/labels'
        ]
        
        for dir_path in dirs:
            full_path = Path(self.output_path) / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

    def _has_chinese(self, text):
        """检查文本是否包含中文字符"""
        pattern = re.compile(r'[\u4e00-\u9fff]')
        return bool(pattern.search(text))

    def _generate_english_filename(self, original_name):
        """生成随机英文文件名"""
        name, ext = os.path.splitext(original_name)
        
        if original_name in self.filename_mapping:
            return self.filename_mapping[original_name]
        
        while True:
            length = random.randint(6, 12)
            new_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
            new_filename = new_name + ext
            
            if new_filename not in self.filename_mapping.values():
                self.filename_mapping[original_name] = new_filename
                return new_filename

    def _transform_single_file(self, label_file):
        """转换单个标注文件"""
        norm_bnd_points = []
        
        try:
            with open(label_file, "r", encoding='utf-8') as f:
                label_data = json.load(f)
        except UnicodeDecodeError:
            with open(label_file, "r", encoding='gbk') as f:
                label_data = json.load(f)
        
        label_infos = label_data["shapes"]
        image_height = label_data["imageHeight"]
        image_width = label_data["imageWidth"]
        
        # 获取原始图像文件名
        original_image_name = label_data.get("imagePath", "")
        if not original_image_name:
            label_name = os.path.basename(label_file)
            name_without_ext = os.path.splitext(label_name)[0]
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
            base_name_key = label_name_without_ext
            
            if base_name_key not in self.filename_mapping:
                while True:
                    length = random.randint(6, 12)
                    new_base_name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
                    if not any(existing_name.startswith(new_base_name) for existing_name in self.filename_mapping.values()):
                        self.filename_mapping[base_name_key] = new_base_name
                        break
            
            new_base_name = self.filename_mapping[base_name_key]
            new_label_name = new_base_name + ".json"
            
            if original_image_name:
                original_image_ext = os.path.splitext(original_image_name)[1]
                new_image_name = new_base_name + original_image_ext
                self.filename_mapping[original_image_name] = new_image_name
            else:
                new_image_name = ""
        else:
            new_label_name = label_basename
            new_image_name = original_image_name
            new_base_name = label_name_without_ext

        # 保存到临时目录
        yolo_label_name = new_base_name + ".txt"
        yolo_label_path = os.path.join(self.temp_label_path, yolo_label_name)
        self._write_yolo_label(yolo_label_path, norm_bnd_points)

        # 复制图像到临时目录
        if original_image_name and new_image_name:
            original_image_path = os.path.join(self.image_path, original_image_name)
            new_image_path = os.path.join(self.temp_image_path, new_image_name)
            
            if os.path.exists(original_image_path):
                shutil.copy2(original_image_path, new_image_path)
                return (new_image_path, yolo_label_path)  # 返回文件对
        
        return None

    def _write_yolo_label(self, write_path, norm_bnd_points):
        """写入YOLO格式标签文件"""
        with open(write_path, "w") as f:
            for norm_bnd_point in norm_bnd_points:
                if self.label_type == 1:
                    norm_data = " ".join([str(x) for x in norm_bnd_point])
                else:
                    norm_data = " ".join([str(x) for x in norm_bnd_point])
                f.write(norm_data + "\n")
    
    def _normalize(self, bnd_point, image_height, image_width, label_idx):
        """归一化边界框坐标"""
        return [label_idx, bnd_point[0]/image_width, bnd_point[1]/image_height, 
                bnd_point[2]/image_width, bnd_point[3]/image_height]

    def _split_and_copy_files(self, file_pairs):
        """分割数据集并复制文件"""
        # 随机打乱数据
        random.shuffle(file_pairs)
        
        # 计算分割点
        split_point = int(len(file_pairs) * self.train_ratio)
        
        train_pairs = file_pairs[:split_point]
        val_pairs = file_pairs[split_point:]
        
        print(f"训练集: {len(train_pairs)} 个样本")
        print(f"验证集: {len(val_pairs)} 个样本")
        
        # 复制训练集文件
        for img_path, label_path in train_pairs:
            img_file = Path(img_path)
            label_file = Path(label_path)
            
            dst_img = Path(self.output_path) / 'train' / 'images' / img_file.name
            dst_label = Path(self.output_path) / 'train' / 'labels' / label_file.name
            
            shutil.copy2(img_path, dst_img)
            shutil.copy2(label_path, dst_label)
        
        # 复制验证集文件
        for img_path, label_path in val_pairs:
            img_file = Path(img_path)
            label_file = Path(label_path)
            
            dst_img = Path(self.output_path) / 'val' / 'images' / img_file.name
            dst_label = Path(self.output_path) / 'val' / 'labels' / label_file.name
            
            shutil.copy2(img_path, dst_img)
            shutil.copy2(label_path, dst_label)
        
        return len(train_pairs), len(val_pairs)

    def _create_dataset_yaml(self):
        """创建dataset.yaml文件"""
        yaml_content = f"""# YOLO数据集配置文件
path: {os.path.abspath(self.output_path)}
train: train/images
val: val/images

# 类别数量
nc: {len(self.label_list)}

# 类别名称
names: {list(self.label_list.keys())}
"""
        yaml_path = os.path.join(self.output_path, "dataset.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        print(f"已创建dataset.yaml文件: {yaml_path}")

    def _save_filename_mapping(self):
        """保存文件名映射关系"""
        if self.convert_chinese and self.filename_mapping:
            mapping_file = os.path.join(self.output_path, "filename_mapping.json")
            
            detailed_mapping = {
                "说明": "原始文件名到英文文件名的映射关系",
                "重要提示": "图像文件和标签文件使用相同的基础英文名称，只有扩展名不同",
                "映射关系": self.filename_mapping
            }
            
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_mapping, f, ensure_ascii=False, indent=2)
            print(f"已保存文件名映射: {mapping_file}")

    def _cleanup_temp_dir(self):
        """清理临时目录"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("已清理临时文件")

    def convert_and_split(self):
        """执行完整的转换和分割流程"""
        label_files = glob.glob(os.path.join(self.label_path, "*.json"))
        
        if not label_files:
            print("未找到JSON标签文件!")
            return
        
        print(f"找到 {len(label_files)} 个标签文件")
        
        # 第一步：转换标注格式
        print("\n正在转换标注格式...")
        file_pairs = []
        
        for label_file in tqdm.tqdm(label_files, desc="转换进度"):
            try:
                result = self._transform_single_file(label_file)
                if result:
                    file_pairs.append(result)
            except Exception as e:
                print(f"处理文件 {label_file} 时出错: {e}")
        
        if not file_pairs:
            print("错误: 没有成功转换任何文件")
            return
        
        print(f"成功转换 {len(file_pairs)} 对图片-标签文件")
        
        # 第二步：分割数据集
        print("\n正在分割数据集...")
        train_count, val_count = self._split_and_copy_files(file_pairs)
        
        # 第三步：创建配置文件
        print("\n正在创建配置文件...")
        self._create_dataset_yaml()
        self._save_filename_mapping()
        
        # 第四步：清理临时文件
        self._cleanup_temp_dir()
        
        # 输出结果
        print(f"\n转换和分割完成!")
        print(f"输出目录结构:")
        print(f"  {self.output_path}/")
        print(f"  ├── train/")
        print(f"  │   ├── images/  ({train_count} 张图片)")
        print(f"  │   └── labels/  ({train_count} 个标签)")
        print(f"  ├── val/")
        print(f"  │   ├── images/  ({val_count} 张图片)")
        print(f"  │   └── labels/  ({val_count} 个标签)")
        print(f"  ├── dataset.yaml")
        if self.convert_chinese and self.filename_mapping:
            print(f"  └── filename_mapping.json")
        print(f"\n共处理了 {len(self.label_list)} 个类别: {list(self.label_list.keys())}")


def main():
    parser = argparse.ArgumentParser(description='AnyLabel2YOLO训练验证集生成器')
    parser.add_argument('--label_path', '-l', type=str, default="/expdata/givap/research/test123/dataset/cut1",
                      help='标注文件目录路径（JSON格式）')
    parser.add_argument('--image_path', '-i', type=str, default="/expdata/givap/research/test123/dataset/cut1",
                      help='图片文件目录路径')
    parser.add_argument('--output_path', '-o', type=str, default="/expdata/givap/research/test123/dataset/cut1_yolo",
                      help='输出目录路径')
    parser.add_argument('--label_type', '-t', type=int, default=1, choices=[1, 2],
                      help='标注类型: 1=矩形框, 2=多边形 (默认: 1)')
    parser.add_argument('--convert_chinese', '-c', action='store_true',
                      help='转换中文文件名为英文')
    parser.add_argument('--train_ratio', '-r', type=float, default=0.8,
                      help='训练集比例 (默认: 0.8)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                      help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.label_path):
        print(f"错误: 标注目录 {args.label_path} 不存在")
        return
    
    if not os.path.exists(args.image_path):
        print(f"错误: 图片目录 {args.image_path} 不存在")
        return
    
    print(f"标注目录: {args.label_path}")
    print(f"图片目录: {args.image_path}")
    print(f"输出目录: {args.output_path}")
    print(f"标注类型: {'矩形框' if args.label_type == 1 else '多边形'}")
    print(f"训练集比例: {args.train_ratio}")
    print(f"验证集比例: {1 - args.train_ratio}")
    print(f"转换中文文件名: {'是' if args.convert_chinese else '否'}")
    
    # 创建转换器并执行
    converter = AnyLabel2YOLOTrainVal(
        label_path=args.label_path,
        image_path=args.image_path,
        output_path=args.output_path,
        label_type=args.label_type,
        convert_chinese=args.convert_chinese,
        train_ratio=args.train_ratio,
        random_seed=args.seed
    )
    
    converter.convert_and_split()


if __name__ == "__main__":
    main()
