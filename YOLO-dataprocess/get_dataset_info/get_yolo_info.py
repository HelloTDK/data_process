import os
from pathlib import Path
from collections import Counter

def get_yolo_classes_info(labels_dir):
    """
    统计YOLO数据集中的类别信息
    
    Args:
        labels_dir (str): 标签文件目录路径
    
    Returns:
        tuple: (类别总数, 类别计数字典, 标签文件总数)
    """
    labels_path = Path(labels_dir)
    class_counts = Counter()
    total_files = 0
    
    # 遍历所有txt文件
    for txt_file in labels_path.glob('*.txt'):
        total_files += 1
        try:
            # 读取标签文件
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # YOLO格式中，每行的第一个数字是类别ID
                    class_id = int(line.strip().split()[0])
                    class_counts[class_id] += 1
        except Exception as e:
            print(f'处理文件 {txt_file.name} 时出错: {str(e)}')
    
    return len(class_counts), dict(class_counts), total_files

def print_classes_info(class_counts_dict, total_files):
    """
    打印类别统计信息
    
    Args:
        class_counts_dict (dict): 类别计数字典
        total_files (int): 标签文件总数
    """
    print(f"\n找到标签文件总数: {total_files}")
    print(f"数据集中共有 {len(class_counts_dict)} 个类别")
    print("\n各类别统计信息:")
    print("-" * 30)
    print("类别ID\t\t标注框数量")
    print("-" * 30)
    
    # 按类别ID排序输出
    for class_id in sorted(class_counts_dict.keys()):
        count = class_counts_dict[class_id]
        print(f"{class_id}\t\t{count}")
    print("-" * 30)

if __name__ == '__main__':
    # 设置标签文件目录
    labels_directory = r'D:\Data\car_plate\plate_det\detect_plate_datasets\train_data1\train_data1\train_data\CCPD_cut'  # 替换为您的标签文件目录
    
    # 获取类别信息
    num_classes, class_counts, total_files = get_yolo_classes_info(labels_directory)
    
    # 打印统计信息
    print_classes_info(class_counts, total_files)
