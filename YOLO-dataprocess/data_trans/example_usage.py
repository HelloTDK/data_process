#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例脚本：演示如何使用AnyLabel2YOLO和训练验证集分割工具
"""

import os
import sys
import argparse
from pathlib import Path

# 添加父目录到路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入转换工具
from data_trans.anylabel2yolo import AnyLabel2YOLO
from data_process.trainval import find_image_label_pairs, create_directories, split_dataset, copy_files, create_yaml_config


def convert_to_yolo(label_path, image_path, output_path, label_type=1, convert_chinese=False):
    """
    将标注数据转换为YOLO格式
    
    Args:
        label_path: 标注文件路径
        image_path: 图像文件路径
        output_path: 输出路径
        label_type: 标注类型 (1: 矩形框, 2: 多边形)
        convert_chinese: 是否转换中文文件名
    """
    print(f"\n{'='*50}")
    print(f"开始转换数据为YOLO格式")
    print(f"{'='*50}")
    print(f"标注文件路径: {label_path}")
    print(f"图像文件路径: {image_path}")
    print(f"输出路径: {output_path}")
    print(f"标注类型: {'矩形框' if label_type == 1 else '多边形'}")
    print(f"转换中文文件名: {'是' if convert_chinese else '否'}")
    
    # 创建转换器
    converter = AnyLabel2YOLO(
        label_path=label_path,
        image_path=image_path,
        output_path=output_path,
        get_dataset_yaml=True,
        label_type=label_type,
        convert_chinese=convert_chinese
    )
    
    # 执行转换
    converter.batch_transform()
    
    print(f"\n转换完成! YOLO格式数据保存在: {output_path}")
    return output_path


def split_yolo_dataset(input_dir, output_dir, train_ratio=0.8, seed=42):
    """
    将YOLO格式数据集分割为训练集和验证集
    
    Args:
        input_dir: 输入数据目录
        output_dir: 输出目录
        train_ratio: 训练集比例
        seed: 随机种子
    """
    import random
    random.seed(seed)
    
    print(f"\n{'='*50}")
    print(f"开始分割数据集为训练集和验证集")
    print(f"{'='*50}")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"训练集比例: {train_ratio}")
    print(f"验证集比例: {1 - train_ratio}")
    
    # 查找图片和标签文件对
    print("\n正在查找图片和标签文件...")
    pairs = find_image_label_pairs(input_dir)
    
    if not pairs:
        print("错误: 没有找到有效的图片-标签文件对")
        return
    
    print(f"找到 {len(pairs)} 对图片-标签文件")
    
    # 分割数据集
    print("\n正在分割数据集...")
    train_pairs, val_pairs = split_dataset(pairs, train_ratio)
    
    print(f"训练集: {len(train_pairs)} 个样本")
    print(f"验证集: {len(val_pairs)} 个样本")
    
    # 创建输出目录
    print("\n正在创建输出目录...")
    create_directories(output_dir)
    
    # 复制文件
    print("\n正在复制文件...")
    print("复制训练集文件...")
    copy_files(train_pairs, output_dir, 'train')
    
    print("复制验证集文件...")
    copy_files(val_pairs, output_dir, 'val')
    
    # 创建YAML配置文件
    print("\n正在创建YAML配置文件...")
    # 从原始数据集的yaml文件中获取类别信息
    original_yaml = os.path.join(input_dir, "dataset.yaml")
    class_names = []
    
    if os.path.exists(original_yaml):
        with open(original_yaml, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith("names:"):
                    try:
                        # 尝试解析类别名称
                        import ast
                        class_names = ast.literal_eval(line.strip()[6:].strip())
                        break
                    except:
                        pass
    
    if not class_names:
        print("无法从原始dataset.yaml获取类别名称，将使用默认名称")
        class_names = [f'class_{i}' for i in range(10)]  # 默认类别名称
    
    create_yaml_config(output_dir, class_names)
    
    print("\n数据集分割完成！")
    print(f"输出目录结构:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── images/  ({len(train_pairs)} 张图片)")
    print(f"  │   └── labels/  ({len(train_pairs)} 个标签)")
    print(f"  ├── val/")
    print(f"  │   ├── images/  ({len(val_pairs)} 张图片)")
    print(f"  │   └── labels/  ({len(val_pairs)} 个标签)")
    print(f"  └── dataset.yaml")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='YOLO数据集准备工具')
    parser.add_argument('--label_path', type=str, required=True,
                      help='标注文件路径')
    parser.add_argument('--image_path', type=str, required=True,
                      help='图像文件路径')
    parser.add_argument('--output_path', type=str, required=True,
                      help='输出路径')
    parser.add_argument('--trainval_path', type=str, default=None,
                      help='训练验证集输出路径 (可选，如不指定则不进行分割)')
    parser.add_argument('--label_type', type=int, default=1,
                      help='标注类型 (1: 矩形框, 2: 多边形)')
    parser.add_argument('--convert_chinese', action='store_true',
                      help='是否转换中文文件名')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                      help='训练集比例 (默认: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 转换为YOLO格式
    yolo_output = convert_to_yolo(
        args.label_path,
        args.image_path,
        args.output_path,
        args.label_type,
        args.convert_chinese
    )
    
    # 如果指定了trainval_path，则进行训练验证集分割
    if args.trainval_path:
        split_yolo_dataset(
            yolo_output,
            args.trainval_path,
            args.train_ratio,
            args.seed
        )
        
        print(f"\n{'='*50}")
        print(f"数据集准备完成!")
        print(f"{'='*50}")
        print(f"YOLO格式数据: {args.output_path}")
        print(f"训练验证集: {args.trainval_path}")
        print(f"\n现在您可以使用以下命令训练YOLOv8模型:")
        print(f"yolo train model=yolov8n.pt data={os.path.join(args.trainval_path, 'dataset.yaml')} epochs=100 imgsz=640")
    else:
        print(f"\n{'='*50}")
        print(f"数据转换完成!")
        print(f"{'='*50}")
        print(f"YOLO格式数据: {args.output_path}")
        print(f"\n如需分割训练验证集，请使用trainval.py脚本或指定--trainval_path参数")


if __name__ == "__main__":
    main()