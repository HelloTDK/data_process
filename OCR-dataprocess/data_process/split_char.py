#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字符分割脚本
从完整的图片中等分切割出每个字符，并生成新的标注文件
"""

import os
import cv2
import numpy as np
from collections import defaultdict
import argparse
import glob


def safe_filename(char):
    """
    将字符转换为安全的文件名
    处理特殊字符，避免文件名冲突
    """
    # 定义不能用作文件名的字符
    invalid_chars = r'<>:"/\|?*'
    
    # 如果是特殊字符，使用其Unicode编码
    if char in invalid_chars or ord(char) < 32:
        return f"unicode_{ord(char)}"
    
    # 处理空格
    if char == ' ':
        return "space"
    
    return char



def split_characters(input_dir, output_dir, images_dir="images"):
    """
    从完整图片中分割出单个字符
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        images_dir: 原始图片目录
    """
    
    # 创建输出目录
    input_txt_path = glob.glob(os.path.join(input_dir, "*.txt"))[0]
    print(f"input_txt_path: {input_txt_path}")
    # imgs_dir = os.path.join(input_dir, "images")
    output_images_dir = os.path.join(output_dir, "images")
    os.makedirs(output_images_dir, exist_ok=True)
    
    # 用于统计字符出现次数，处理重复字符
    char_count = defaultdict(int)
    
    # 存储新的标注信息
    new_annotations = []
    
    # 读取原始标注文件
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"开始处理 {len(lines)} 行数据...")
    
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # 分割图片路径和字符标签
        parts = line.split('\t')
        if len(parts) != 2:
            print(f"跳过格式错误的行 {line_idx + 1}: {line}")
            continue
            
        image_path, text_label = parts
        
        # 构建完整的图片路径
        full_image_path = os.path.join(input_dir, image_path)
        if not os.path.exists(full_image_path):
            print(f"图片不存在: {full_image_path}")
            continue
            
        # 读取图片 - 支持中文路径
        try:
            # 使用numpy和cv2.imdecode来读取中文路径的图片
            image_data = np.fromfile(full_image_path, dtype=np.uint8)
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            if image is None:
                print(f"无法读取图片: {full_image_path}")
                continue
        except Exception as e:
            print(f"读取图片出错 {full_image_path}: {e}")
            continue
            
        height, width = image.shape[:2]
        char_count_in_text = len(text_label)
        
        if char_count_in_text == 0:
            continue
            
        # 等分切割图片
        char_width = width // char_count_in_text
        
        for char_idx, char in enumerate(text_label):
            # 计算字符在图片中的位置
            x_start = char_idx * char_width
            x_end = (char_idx + 1) * char_width if char_idx < char_count_in_text - 1 else width
            
            # 切割字符图片
            char_image = image[:, x_start:x_end]
            
            # 处理重复字符命名
            char_count[char] += 1
            safe_char = safe_filename(char)
            char_filename = f"{safe_char}_{char_count[char]}.jpg"
            
            # 保存字符图片 - 支持中文路径
            char_image_path = os.path.join(output_images_dir, char_filename)
            try:
                # 使用cv2.imencode来保存，支持中文路径
                success, encoded_img = cv2.imencode('.jpg', char_image)
                if success:
                    with open(char_image_path, 'wb') as f:
                        f.write(encoded_img.tobytes())
                else:
                    print(f"编码图片失败: {char_image_path}")
                    continue
            except Exception as e:
                print(f"保存图片出错 {char_image_path}: {e}")
                continue
            
            # 添加到新标注
            relative_path = f"images/{char_filename}"
            new_annotations.append(f"{relative_path}\t{char}")
            
        if (line_idx + 1) % 100 == 0:
            print(f"已处理 {line_idx + 1}/{len(lines)} 行")
    
    # 写入新的标注文件
    output_txt_path = os.path.join(output_dir, "split_chars.txt")
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for annotation in new_annotations:
            f.write(annotation + '\n')
    
    print(f"处理完成!")
    print(f"共分割出 {len(new_annotations)} 个字符")
    print(f"字符图片保存至: {output_images_dir}")
    print(f"新标注文件保存至: {output_txt_path}")
    
    # 统计字符出现次数
    print(f"\n字符统计:")
    sorted_chars = sorted(char_count.items(), key=lambda x: x[1], reverse=True)
    for char, count in sorted_chars[:20]:  # 显示前20个最常见的字符
        print(f"'{char}': {count} 次")
    if len(sorted_chars) > 20:
        print("...")

def main():
    parser = argparse.ArgumentParser(description='字符分割脚本')
    parser.add_argument('--input', '-i', default=r'D:\Data\car_plate\plate_recong\git_plate\val_verify', 
                       help='整个目录路径')
    parser.add_argument('--output', '-o', default='split_output', 
                       help='输出目录 (默认: split_output)')
    parser.add_argument('--images_dir', default='images',
                       help='原始图片目录 (默认: images)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    args.output = os.path.join(args.input, args.output)
    split_characters(args.input, args.output, args.images_dir)

if __name__ == "__main__":
    main()
