import os
import shutil
from pathlib import Path

def cut_yolo_labels(input_dir, output_dir):
    """
    处理指定目录下的所有YOLO标签文件，截取每行的前5个数值
    
    Args:
        input_dir (str): 输入标签文件目录路径
        output_dir (str): 输出标签文件目录路径
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入目录中的所有.txt文件
    input_path = Path(input_dir)
    for txt_file in input_path.glob('*.txt'):
        try:
            # 读取原始文件，使用UTF-8编码
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 处理每一行，只保留前5个数值
            processed_lines = []
            for line in lines:
                values = line.strip().split()
                if len(values) >= 5:
                    processed_line = ' '.join(values[:5])
                    processed_lines.append(processed_line)
            
            # 创建输出文件路径
            output_file = Path(output_dir) / txt_file.name
            
            # 写入处理后的内容，使用UTF-8编码
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(processed_lines))
            
            print(f'已处理: {txt_file.name}')
        except Exception as e:
            print(f'处理文件 {txt_file.name} 时出错: {str(e)}')

if __name__ == '__main__':
    # 设置输入和输出目录
    input_directory = r'D:\Data\car_plate\plate_det\detect_plate_datasets\val_detect\val_detect\val_detect\all_labels'  # 原始标签文件目录
    output_directory = r'D:\Data\car_plate\plate_det\detect_plate_datasets\val_detect\val_detect\val_detect\all_labels_cut'  # 处理后的标签文件目录
    
    # 执行处理
    cut_yolo_labels(input_directory, output_directory)
    print('所有文件处理完成！')
