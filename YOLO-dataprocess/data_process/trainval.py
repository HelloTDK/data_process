#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO数据集训练验证集分割脚本
用于将YOLO格式的数据集按比例分为训练集和验证集
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple


def find_image_label_pairs(data_dir: str, img_extensions: List[str] = None, 
                          label_extension: str = '.txt') -> List[Tuple[str, str]]:
    """
    查找图片和标签文件对
    
    Args:
        data_dir: 数据目录路径（包含images和labels子目录）
        img_extensions: 图片文件扩展名列表
        label_extension: 标签文件扩展名
    
    Returns:
        List[Tuple[str, str]]: (图片路径, 标签路径) 的列表
    """
    if img_extensions is None:
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    data_path = Path(data_dir)
    images_dir = data_path / 'images'
    labels_dir = data_path / 'labels'
    pairs = []
    
    # 检查目录是否存在
    if not images_dir.exists():
        print(f"错误: 图片目录 {images_dir} 不存在")
        return pairs
    
    if not labels_dir.exists():
        print(f"错误: 标签目录 {labels_dir} 不存在")
        return pairs
    
    print(f"正在扫描图片目录: {images_dir}")
    print(f"正在扫描标签目录: {labels_dir}")
    
    # 获取所有图片文件
    for img_ext in img_extensions:
        for img_file in images_dir.glob(f'*{img_ext}'):
            # 查找对应的标签文件
            label_file = labels_dir / f"{img_file.stem}{label_extension}"
            if label_file.exists():
                pairs.append((str(img_file), str(label_file)))
            else:
                print(f"警告: 找不到图片 {img_file.name} 对应的标签文件 {label_file.name}")
    
    return pairs


def create_directories(output_dir: str):
    """
    创建输出目录结构
    
    Args:
        output_dir: 输出根目录
    """
    dirs = [
        'train/images',
        'train/labels', 
        'val/images',
        'val/labels'
    ]
    
    for dir_path in dirs:
        full_path = Path(output_dir) / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {full_path}")


def split_dataset(pairs: List[Tuple[str, str]], train_ratio: float = 0.8) -> Tuple[List, List]:
    """
    分割数据集
    
    Args:
        pairs: 图片标签文件对列表
        train_ratio: 训练集比例
    
    Returns:
        Tuple[List, List]: (训练集, 验证集)
    """
    # 随机打乱数据
    random.shuffle(pairs)
    
    # 计算分割点
    split_point = int(len(pairs) * train_ratio)
    
    train_pairs = pairs[:split_point]
    val_pairs = pairs[split_point:]
    
    return train_pairs, val_pairs


def copy_files(pairs: List[Tuple[str, str]], output_dir: str, subset: str):
    """
    复制文件到目标目录
    
    Args:
        pairs: 文件对列表
        output_dir: 输出目录
        subset: 子集名称 ('train' 或 'val')
    """
    output_path = Path(output_dir)
    
    for img_path, label_path in pairs:
        img_file = Path(img_path)
        label_file = Path(label_path)
        
        # 复制图片文件
        dst_img = output_path / subset / 'images' / img_file.name
        shutil.copy2(img_path, dst_img)
        
        # 复制标签文件
        dst_label = output_path / subset / 'labels' / label_file.name
        shutil.copy2(label_path, dst_label)


def create_yaml_config(output_dir: str, class_names: List[str] = None):
    """
    创建YOLO训练配置文件
    
    Args:
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    yaml_content = f"""# YOLO数据集配置文件
path: {os.path.abspath(output_dir)}  # 数据集根目录
train: train/images  # 训练图片目录 (相对于path)
val: val/images      # 验证图片目录 (相对于path)

# 类别数量
nc: {len(class_names) if class_names else 'YOUR_CLASS_COUNT'}

# 类别名称
names: {class_names if class_names else ['class0', 'class1', 'class2']}  # 请根据实际情况修改
"""
    
    yaml_path = Path(output_dir) / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"创建配置文件: {yaml_path}")


def get_class_names_from_labels(pairs: List[Tuple[str, str]]) -> List[str]:
    """
    从标签文件中提取类别名称
    
    Args:
        pairs: 文件对列表
    
    Returns:
        List[str]: 类别ID列表 (注意：这里返回的是ID，需要手动映射到名称)
    """
    class_ids = set()
    
    for _, label_path in pairs:
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_ids.add(class_id)
        except Exception as e:
            print(f"读取标签文件 {label_path} 时出错: {e}")
    
    return sorted(list(class_ids))


def main():
    parser = argparse.ArgumentParser(description='YOLO数据集训练验证集分割工具')
    parser.add_argument('--input_dir', '-i', type=str, default=r"D:\Code\Python\wb\xy480\data\Desktop\yolo",
                      help='输入数据目录路径（包含images和labels子目录）')
    parser.add_argument('--output_dir', '-o', type=str, default=r"D:\Code\Python\wb\xy480\data\Desktop\trainval",
                      help='输出目录路径')
    parser.add_argument('--train_ratio', '-r', type=float, default=0.8,
                      help='训练集比例 (默认: 0.8)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                      help='随机种子 (默认: 42)')
    parser.add_argument('--copy_mode', '-c', action='store_true',
                      help='复制模式（默认移动文件）')
    parser.add_argument('--create_yaml', '-y', action='store_true',
                      help='创建YOLO配置文件')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录 {args.input_dir} 不存在")
        return
    
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"训练集比例: {args.train_ratio}")
    print(f"验证集比例: {1 - args.train_ratio}")
    
    # 查找图片和标签文件对
    print("\n正在查找图片和标签文件...")
    pairs = find_image_label_pairs(args.input_dir)
    
    if not pairs:
        print("错误: 没有找到有效的图片-标签文件对")
        return
    
    print(f"找到 {len(pairs)} 对图片-标签文件")
    
    # 分割数据集
    print("\n正在分割数据集...")
    train_pairs, val_pairs = split_dataset(pairs, args.train_ratio)
    
    print(f"训练集: {len(train_pairs)} 个样本")
    print(f"验证集: {len(val_pairs)} 个样本")
    
    # 创建输出目录
    print("\n正在创建输出目录...")
    create_directories(args.output_dir)
    
    # 复制文件
    print("\n正在复制文件...")
    print("复制训练集文件...")
    copy_files(train_pairs, args.output_dir, 'train')
    
    print("复制验证集文件...")
    copy_files(val_pairs, args.output_dir, 'val')
    
    # 创建YAML配置文件
    if args.create_yaml:
        print("\n正在创建YAML配置文件...")
        class_ids = get_class_names_from_labels(pairs)
        class_names = [f'class_{i}' for i in class_ids]  # 生成默认类别名称
        create_yaml_config(args.output_dir, class_names)
        print(f"找到类别ID: {class_ids}")
        print("请手动编辑 dataset.yaml 文件中的类别名称")
    
    print("\n数据集分割完成！")
    print(f"输出目录结构:")
    print(f"  {args.output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/  ({len(train_pairs)} 张图片)")
    print(f"  │   └── labels/  ({len(train_pairs)} 个标签)")
    print(f"  ├── val/")
    print(f"  │   ├── images/  ({len(val_pairs)} 张图片)")
    print(f"  │   └── labels/  ({len(val_pairs)} 个标签)")
    if args.create_yaml:
        print(f"  └── dataset.yaml")


if __name__ == '__main__':
    main()
