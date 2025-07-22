#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文文件名转英文文件名工具
支持图片文件和对应的JSON文件同步重命名
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

# 常见图片文件扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}

# 中文到英文的映射字典（可根据需要扩展）
CHINESE_TO_ENGLISH = {
    # 数字
    '零': 'zero', '一': 'one', '二': 'two', '三': 'three', '四': 'four',
    '五': 'five', '六': 'six', '七': 'seven', '八': 'eight', '九': 'nine',
    '十': 'ten', '百': 'hundred', '千': 'thousand', '万': 'ten_thousand',
    
    # 常用词汇
    '图片': 'image', '照片': 'photo', '文件': 'file', '数据': 'data',
    '标签': 'label', '检测': 'detect', '识别': 'recognize', '分类': 'classify',
    '训练': 'train', '测试': 'test', '验证': 'val', '样本': 'sample',
    '目标': 'target', '对象': 'object', '类别': 'class', '种类': 'type',
    '人': 'person', '车': 'car', '动物': 'animal', '建筑': 'building',
    '道路': 'road', '天空': 'sky', '树': 'tree', '花': 'flower',
    '猫': 'cat', '狗': 'dog', '鸟': 'bird', '鱼': 'fish',
    '红': 'red', '绿': 'green', '蓝': 'blue', '黄': 'yellow',
    '黑': 'black', '白': 'white', '灰': 'gray', '紫': 'purple',
    '大': 'big', '小': 'small', '高': 'high', '低': 'low',
    '新': 'new', '旧': 'old', '好': 'good', '坏': 'bad',
    '左': 'left', '右': 'right', '上': 'up', '下': 'down',
    '前': 'front', '后': 'back', '中': 'center', '边': 'edge',
}

def contains_chinese(text: str) -> bool:
    """检查字符串是否包含中文字符"""
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(chinese_pattern.search(text))

def chinese_to_english(text: str) -> str:
    """将中文字符转换为英文"""
    result = text
    
    # 先尝试直接映射
    for chinese, english in CHINESE_TO_ENGLISH.items():
        result = result.replace(chinese, english)
    
    # 如果还有中文字符，使用拼音或者生成hash
    if contains_chinese(result):
        # 移除所有中文字符，用下划线替代
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        chinese_parts = chinese_pattern.findall(result)
        
        # 为每个中文部分生成简短的hash
        for chinese_part in chinese_parts:
            hash_value = hashlib.md5(chinese_part.encode('utf-8')).hexdigest()[:6]
            result = result.replace(chinese_part, f'ch_{hash_value}')
    
    # 清理文件名：替换特殊字符为下划线
    result = re.sub(r'[^\w\-_.]', '_', result)
    # 移除多余的下划线
    result = re.sub(r'_+', '_', result)
    # 移除开头和结尾的下划线
    result = result.strip('_')
    
    return result

def get_file_pairs(directory: Path) -> List[Tuple[Path, Path]]:
    """获取需要重命名的文件对（原文件，新文件名）"""
    file_pairs = []
    
    for file_path in directory.iterdir():
        if file_path.is_file():
            filename = file_path.name
            
            # 检查是否包含中文
            if contains_chinese(filename):
                # 分离文件名和扩展名
                stem = file_path.stem
                suffix = file_path.suffix
                
                # 转换文件名
                new_stem = chinese_to_english(stem)
                new_filename = f"{new_stem}{suffix}"
                new_file_path = file_path.parent / new_filename
                
                # 确保新文件名不冲突
                counter = 1
                while new_file_path.exists():
                    new_filename = f"{new_stem}_{counter}{suffix}"
                    new_file_path = file_path.parent / new_filename
                    counter += 1
                
                file_pairs.append((file_path, new_file_path))
    
    return file_pairs

def find_related_files(file_path: Path, directory: Path) -> List[Path]:
    """查找与给定文件相关的文件（如同名的JSON文件）"""
    related_files = []
    stem = file_path.stem
    
    # 查找同名但不同扩展名的文件
    for related_file in directory.iterdir():
        if (related_file.is_file() and 
            related_file.stem == stem and 
            related_file != file_path):
            related_files.append(related_file)
    
    return related_files

def rename_files(directory: str, dry_run: bool = True) -> None:
    """重命名目录中的中文文件名"""
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"错误：目录 {directory} 不存在")
        return
    
    if not directory_path.is_dir():
        print(f"错误：{directory} 不是一个目录")
        return
    
    print(f"处理目录：{directory_path.absolute()}")
    print(f"模式：{'预览模式' if dry_run else '执行模式'}")
    print("-" * 50)
    
    # 获取所有需要重命名的文件
    file_pairs = get_file_pairs(directory_path)
    
    if not file_pairs:
        print("没有找到包含中文字符的文件")
        return
    
    # 按文件组织重命名操作
    rename_groups = {}  # {original_stem: [(old_path, new_path), ...]}
    
    for old_path, new_path in file_pairs:
        old_stem = old_path.stem
        if old_stem not in rename_groups:
            rename_groups[old_stem] = []
        rename_groups[old_stem].append((old_path, new_path))
        
        # 查找相关文件（如JSON文件）
        related_files = find_related_files(old_path, directory_path)
        for related_file in related_files:
            if contains_chinese(related_file.name):
                # 为相关文件生成新名称
                new_stem = chinese_to_english(old_stem)
                new_related_name = f"{new_stem}{related_file.suffix}"
                new_related_path = related_file.parent / new_related_name
                
                # 确保新文件名不冲突
                counter = 1
                while new_related_path.exists():
                    new_related_name = f"{new_stem}_{counter}{related_file.suffix}"
                    new_related_path = related_file.parent / new_related_name
                    counter += 1
                
                rename_groups[old_stem].append((related_file, new_related_path))
    
    # 显示重命名计划
    total_files = 0
    for stem, file_list in rename_groups.items():
        print(f"\n文件组：{stem}")
        for old_path, new_path in file_list:
            print(f"  {old_path.name} -> {new_path.name}")
            total_files += 1
    
    print(f"\n总共 {total_files} 个文件需要重命名")
    
    if dry_run:
        print("\n这是预览模式，没有实际重命名文件。")
        print("如要执行重命名，请使用 --execute 参数")
        return
    
    # 执行重命名
    success_count = 0
    error_count = 0
    
    for stem, file_list in rename_groups.items():
        print(f"\n处理文件组：{stem}")
        for old_path, new_path in file_list:
            try:
                old_path.rename(new_path)
                print(f"  ✓ {old_path.name} -> {new_path.name}")
                success_count += 1
            except Exception as e:
                print(f"  ✗ 重命名失败：{old_path.name} -> {new_path.name}")
                print(f"    错误：{e}")
                error_count += 1
    
    print(f"\n重命名完成：")
    print(f"  成功：{success_count} 个文件")
    print(f"  失败：{error_count} 个文件")

def main():
    parser = argparse.ArgumentParser(
        description="将目录中的中文文件名转换为英文文件名",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 预览模式（不实际重命名）
  python Chinese2Eng.py "D:\\Code\\Python\\wb\\xy480\\data\\Desktop\\all"
  
  # 执行重命名
  python Chinese2Eng.py "D:\\Code\\Python\\wb\\xy480\\data\\Desktop\\all" --execute
        """
    )
    
    parser.add_argument(
        'directory',
        help='要处理的目录路径'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='执行重命名操作（默认为预览模式）'
    )
    
    args = parser.parse_args()
    
    rename_files(args.directory, dry_run=not args.execute)

if __name__ == "__main__":
    main()
