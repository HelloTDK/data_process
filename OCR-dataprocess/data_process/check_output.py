import os
import sys

def check_output_files(output_dir):
    """检查生成的train.txt和val.txt文件"""
    train_txt = os.path.join(output_dir, 'train.txt')
    val_txt = os.path.join(output_dir, 'val.txt')
    
    for txt_file in [train_txt, val_txt]:
        if not os.path.exists(txt_file):
            print(f"文件不存在: {txt_file}")
            continue
            
        print(f"\n检查文件: {txt_file}")
        print("-" * 50)
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            print(f"总行数: {len(lines)}")
            print("前5行内容:")
            
            for i, line in enumerate(lines[:5]):
                line = line.strip()
                if '\t' in line:
                    img_path, label = line.split('\t')
                    print(f"  行{i+1}: 图片路径='{img_path}', 标签='{label}'")
                    print(f"        标签长度: {len(label)} 字符")
                    print(f"        标签编码: {label.encode('utf-8')}")
                else:
                    print(f"  行{i+1}: {line} (格式异常)")
                    
        except Exception as e:
            print(f"读取文件时出错: {e}")
            
        # 检查是否有中文字符
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                chinese_count = sum(1 for char in content if '\u4e00' <= char <= '\u9fff')
                print(f"文件中包含中文字符数: {chinese_count}")
        except Exception as e:
            print(f"检查中文字符时出错: {e}")

if __name__ == '__main__':
    # 使用默认输出目录
    output_dir = r'D:\Data\car_plate\plate_recong\git_plate\git_plate'
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    
    print(f"检查目录: {output_dir}")
    check_output_files(output_dir) 