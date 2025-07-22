#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版文件名特殊字符提取脚本
自动扫描当前目录及子目录，提取所有文件名中的字符并保存到txt文件
"""

import os

def extract_chars_from_filenames(directory_path=".", output_file="文件名字符.txt"):
    """
    从指定目录下所有文件名中提取字符
    """
    
    # 存储所有唯一字符的集合
    all_chars = set()
    
    print(f"正在扫描目录: {os.path.abspath(directory_path)}")
    
    # 遍历目录及其子目录
    file_count = 0
    for root, dirs, files in os.walk(directory_path):
        # 处理文件名
        for filename in files:
            file_count += 1
            # 提取文件名中的每个字符
            for char in filename:
                all_chars.add(char)
        
        # 处理目录名
        for dirname in dirs:
            for char in dirname:
                all_chars.add(char)
    
    print(f"共扫描了 {file_count} 个文件")
    
    # 将字符集合转换为排序后的列表
    sorted_chars = sorted(list(all_chars))
    
    # 写入文件，每行一个字符
    with open(output_file, 'w', encoding='utf-8') as f:
        for char in sorted_chars:
            f.write(char + '\n')
    
    print(f"共提取到 {len(sorted_chars)} 个不同的字符")
    print(f"已保存到文件: {output_file}")
    
    # 显示前30个字符作为预览
    print(f"\n前30个字符预览:")
    for i, char in enumerate(sorted_chars[:30]):
        if char == '\n':
            print(f"{i+1:2d}: \\n (换行符)")
        elif char == '\t':
            print(f"{i+1:2d}: \\t (制表符)")
        elif char == ' ':
            print(f"{i+1:2d}: (空格)")
        elif char == '.':
            print(f"{i+1:2d}: . (点)")
        elif char == '-':
            print(f"{i+1:2d}: - (连字符)")
        elif char == '_':
            print(f"{i+1:2d}: _ (下划线)")
        else:
            print(f"{i+1:2d}: {char}")
    
    if len(sorted_chars) > 30:
        print(f"... 还有 {len(sorted_chars) - 30} 个字符")

if __name__ == "__main__":
    print("文件名字符提取工具")
    print("=" * 50)
    
    # 直接扫描当前目录
    extract_chars_from_filenames(".", "文件名字符.txt") 