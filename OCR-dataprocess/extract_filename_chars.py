#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件名特殊字符提取脚本
解析指定路径下所有文件名，提取其中的特殊字符并去重保存到txt文件
"""

import os
import re
from pathlib import Path

def extract_all_chars_from_filenames(directory_path=".", output_file="filename_chars.txt"):
    """
    从指定目录下所有文件名中提取字符
    
    Args:
        directory_path (str): 要扫描的目录路径，默认为当前目录
        output_file (str): 输出文件名
    """
    
    # 存储所有唯一字符的集合
    all_chars = set()
    
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory_path):
        # 处理文件名
        for filename in files:
            # 提取文件名中的每个字符
            for char in filename:
                all_chars.add(char)
        
        # 处理目录名
        for dirname in dirs:
            for char in dirname:
                all_chars.add(char)
    
    # 将字符集合转换为排序后的列表
    sorted_chars = sorted(list(all_chars))
    
    # 写入文件，每行一个字符
    with open(output_file, 'w', encoding='utf-8') as f:
        for char in sorted_chars:
            f.write(char + '\n')
    
    print(f"共提取到 {len(sorted_chars)} 个不同的字符")
    print(f"已保存到文件: {output_file}")
    
    # 显示前20个字符作为预览
    print("\n前20个字符预览:")
    for i, char in enumerate(sorted_chars[:20]):
        if char == '\n':
            print(f"{i+1:2d}: \\n (换行符)")
        elif char == '\t':
            print(f"{i+1:2d}: \\t (制表符)")
        elif char == ' ':
            print(f"{i+1:2d}: (空格)")
        else:
            print(f"{i+1:2d}: {char}")

def extract_special_chars_only(directory_path=".", output_file="special_chars.txt"):
    """
    只提取特殊字符（非字母数字）
    
    Args:
        directory_path (str): 要扫描的目录路径
        output_file (str): 输出文件名
    """
    
    # 存储所有唯一特殊字符的集合
    special_chars = set()
    
    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory_path):
        # 处理文件名
        for filename in files:
            for char in filename:
                # 如果不是字母或数字，则认为是特殊字符
                if not char.isalnum():
                    special_chars.add(char)
        
        # 处理目录名
        for dirname in dirs:
            for char in dirname:
                if not char.isalnum():
                    special_chars.add(char)
    
    # 将字符集合转换为排序后的列表
    sorted_chars = sorted(list(special_chars))
    
    # 写入文件，每行一个字符
    with open(output_file, 'w', encoding='utf-8') as f:
        for char in sorted_chars:
            f.write(char + '\n')
    
    print(f"共提取到 {len(sorted_chars)} 个特殊字符")
    print(f"已保存到文件: {output_file}")
    
    # 显示所有特殊字符
    print("\n提取到的特殊字符:")
    for i, char in enumerate(sorted_chars):
        if char == '\n':
            print(f"{i+1:2d}: \\n (换行符)")
        elif char == '\t':
            print(f"{i+1:2d}: \\t (制表符)")
        elif char == ' ':
            print(f"{i+1:2d}: (空格)")
        else:
            print(f"{i+1:2d}: {char}")

if __name__ == "__main__":
    print("文件名字符提取工具")
    print("=" * 40)
    
    # 获取用户输入的目录路径
    directory = input("请输入要扫描的目录路径（回车使用当前目录）: ").strip()
    if not directory:
        directory = "."
    
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"错误: 目录 '{directory}' 不存在!")
        exit(1)
    
    # 选择提取模式
    print("\n请选择提取模式:")
    print("1. 提取所有字符")
    print("2. 只提取特殊字符（非字母数字）")
    
    choice = input("请输入选择 (1/2): ").strip()
    
    if choice == "1":
        extract_all_chars_from_filenames(directory, "所有字符.txt")
    elif choice == "2":
        extract_special_chars_only(directory, "特殊字符.txt")
    else:
        print("无效选择，默认提取所有字符")
        extract_all_chars_from_filenames(directory, "所有字符.txt") 