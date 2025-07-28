#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双层车牌合并脚本

如：6051挂_0000000000027824_0.jpg 桂P_0000000000027824_0.jpg 合并成一张图像
如果桂P_0000000000027824_0.jpg 的h小于6051挂_0000000000027824_0.jpg，可以先做resize，然后再拼接
拼接在以后得到桂P6051挂_0000000000027824_0.jpg的图像，和并时汉字在前面，不能随意拼接如 这个不能是6051挂桂P，顺序不能弄反了
如防ED5D62_0000000000021452_0.jpg果前面部分是 防ED5D62 这种7位则不用进行合并
保存的结果在文件目录的上一层下output_merge路径下，不覆盖原始图像集
如果有重复的0000000000027824，比如有四个0000000000027824，只需要保留一组，就可以了
如果不需要拼接的，需要直接拷贝到新的目录夹里面

"""
import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import re
def cv2_imread_unicode(file_path):
    """
    解决cv2.imread()无法读取中文路径的问题
    
    Args:
        file_path: 图像文件路径
    
    Returns:
        numpy.ndarray: 读取的图像数组
    """
    # 使用numpy读取二进制数据，然后用cv2解码
    img_buffer = np.fromfile(str(file_path), dtype=np.uint8)
    img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
    return img

def cv2_imwrite_unicode(file_path, img):
    """
    解决cv2.imwrite()无法写入中文路径的问题
    
    Args:
        file_path: 输出文件路径
        img: 要保存的图像数组
    
    Returns:
        bool: 是否保存成功
    """
    # 获取文件扩展名
    ext = Path(file_path).suffix.lower()
    # 编码图像
    success, img_buffer = cv2.imencode(ext, img)
    if success:
        # 写入文件
        img_buffer.tofile(str(file_path))
        return True
    return False

def get_prefix_type(prefix):
    """
    判断前缀类型
    
    Args:
        prefix: 前缀字符串
    
    Returns:
        int: 前缀类型 (0=纯汉字开头, 1=数字字母开头但包含汉字, 2=纯数字字母)
    """
    # 检查是否包含汉字
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', prefix))
    
    if not has_chinese:
        return 2  # 纯数字字母
    
    # 检查是否以汉字开头
    if re.match(r'^[\u4e00-\u9fff]', prefix):
        return 0  # 纯汉字开头（如：桂N）
    else:
        return 1  # 数字字母开头但包含汉字（如：4C66挂）

def sort_prefixes_for_merge(prefixes, files):
    """
    按照汉字在前的规则对前缀和对应文件进行排序
    规则：纯汉字开头 > 数字字母开头但包含汉字 > 纯数字字母
    
    Args:
        prefixes: 前缀列表
        files: 对应的文件列表
    
    Returns:
        tuple: 排序后的(prefixes, files)
    """
    # 创建前缀和文件的配对
    paired = list(zip(prefixes, files))
    
    # 按照前缀类型排序
    def sort_key(item):
        prefix = item[0]
        prefix_type = get_prefix_type(prefix)
        return (prefix_type, prefix)
    
    paired.sort(key=sort_key)
    
    # 分离排序后的前缀和文件
    sorted_prefixes, sorted_files = zip(*paired)
    return list(sorted_prefixes), list(sorted_files)

def merge_license_plates(input_dir):
    """
    合并双层车牌图像
    
    Args:
        input_dir: 输入图像目录路径
    """
    input_path = Path(input_dir)
    output_path = input_path.parent / "output_merge"
    output_path.mkdir(exist_ok=True)
    
    # 获取所有图像文件
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    # 按照文件名中的ID分组
    image_groups = {}
    single_files = []  # 存储不需要合并的单个文件
    
    for img_file in image_files:
        filename = img_file.stem
        parts = filename.split('_')
        
        if len(parts) >= 3:
            # 提取前缀和ID
            prefix = parts[0]
            image_id = parts[1]
            
            # 如果前缀是7位，直接添加到单个文件列表
            if len(prefix) == 7:
                single_files.append(img_file)
                continue
                
            if image_id not in image_groups:
                image_groups[image_id] = []
            image_groups[image_id].append(img_file)
    
    # 处理单个文件（7位前缀或不需要合并的文件）
    for single_file in single_files:
        output_file = output_path / single_file.name
        shutil.copy2(single_file, output_file)
        print(f"拷贝完成: {single_file.name}")
    
    # 处理需要合并的分组
    processed_ids = set()  # 记录已处理的ID，避免重复处理
    
    for image_id, files in image_groups.items():
        # 如果ID已经处理过，跳过
        if image_id in processed_ids:
            continue
            
        # 如果只有一个文件，直接拷贝
        if len(files) == 1:
            output_file = output_path / files[0].name
            shutil.copy2(files[0], output_file)
            print(f"拷贝完成: {files[0].name}")
            processed_ids.add(image_id)
            continue
        
        # 如果有多个文件，只取前两个进行合并（避免重复ID问题）
        files_to_merge = files[:2]
        
        images = []
        prefixes = []
        valid_files = []
        
        # 读取图像
        for file in files_to_merge:
            img = cv2_imread_unicode(file)
            if img is not None:
                images.append(img)
                prefix = file.stem.split('_')[0]
                prefixes.append(prefix)
                valid_files.append(file)
        
        if len(images) < 2:
            # 如果有效图像少于2个，拷贝第一个
            if len(images) == 1:
                output_file = output_path / valid_files[0].name
                shutil.copy2(valid_files[0], output_file)
                print(f"拷贝完成: {valid_files[0].name}")
            processed_ids.add(image_id)
            continue
        
        # 按照汉字在前的规则排序
        sorted_prefixes, sorted_files = sort_prefixes_for_merge(prefixes, valid_files)
        
        # 重新读取排序后的图像
        sorted_images = []
        for file in sorted_files:
            img = cv2_imread_unicode(file)
            if img is not None:
                sorted_images.append(img)
        
        # 找到最大高度
        max_height = max(img.shape[0] for img in sorted_images)
        
        # 调整图像高度
        resized_images = []
        for img in sorted_images:
            if img.shape[0] < max_height:
                # 计算缩放比例
                scale = max_height / img.shape[0]
                new_width = int(img.shape[1] * scale)
                resized_img = cv2.resize(img, (new_width, max_height))
                resized_images.append(resized_img)
            else:
                resized_images.append(img)
        
        # 水平拼接图像
        merged_img = np.hstack(resized_images)
        
        # 生成输出文件名（使用排序后的前缀）
        merged_prefix = ''.join(sorted_prefixes)
        output_filename = f"{merged_prefix}_{image_id}_{sorted_files[0].stem.split('_')[-1]}.jpg"
        output_file = output_path / output_filename
        
        # 保存合并后的图像
        success = cv2_imwrite_unicode(output_file, merged_img)
        if success:
            print(f"合并完成: {output_filename}")
        else:
            print(f"保存失败: {output_filename}")
        
        # 标记该ID已处理
        processed_ids.add(image_id)

def main():
    """主函数"""
    import sys
    
    input_directory = r"D:\Data\car_plate\plate_recong\live\ID1928278804434837504\ID1928278804434837504\all"

    
    merge_license_plates(input_directory)
    print("车牌合并完成!")

if __name__ == "__main__":
    main()
