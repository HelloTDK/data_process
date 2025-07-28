import os
import re

def process_filename(filename):
    # 分离文件名和扩展名
    name, ext = os.path.splitext(filename)
    
    # 使用正则表达式匹配文件名格式 - 允许字母和数字在第一部分
    match = re.match(r'^([a-zA-Z0-9\.]+)_(\d+)$', name)
    if match:
        first_part, second_part = match.groups()
        
        # 处理包含小数点的情况，无论是否以T开头
        if '.' in first_part:
            # 如果已经以T开头，只移除小数点
            if first_part.startswith('T'):
                new_first_part = 'T' + first_part[1:].replace(".", "")
            else:
                new_first_part = 'T' + first_part.replace(".", "")
        # 检查第一部分是否以T开头
        elif not first_part.startswith('T'):
            # 如果以字母开头，替换第一个字母为T
            if first_part[0].isalpha():
                new_first_part = 'T' + first_part[1:]
            # 如果长度小于4，则添加T
            elif len(first_part) < 4:
                new_first_part = 'T' + first_part
            # 如果长度等于4或5，则替换第一位为T
            elif len(first_part) in [4, 5]:
                new_first_part = 'T' + first_part[1:]
            else:
                new_first_part = first_part
        else:
            # 已经以T开头且没有小数点，保持不变
            new_first_part = first_part
                
        # 构建新文件名
        new_name = f"{new_first_part}_{second_part}{ext}"
        return new_name
    
    # 如果不符合条件，返回原文件名
    return filename

def batch_rename_files(directory):
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            new_filename = process_filename(filename)
            if new_filename != filename:
                os.rename(
                    os.path.join(directory, filename),
                    os.path.join(directory, new_filename)
                )
                print(f"Renamed: {filename} -> {new_filename}")

# 使用当前目录作为处理目录
if __name__ == "__main__":
    batch_rename_files(r"D:\Desktop\split\folder_2")