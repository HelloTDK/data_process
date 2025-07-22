import os
import re
from pathlib import Path

def extract_label_from_filename(filename):
    """
    从文件名中提取标签文本
    例如: '黑RE2881_5452.jpg' -> '黑RE2881'
    例如: '171001使_111.jpg' -> '171001使'
    """
    # 去除文件扩展名
    name_without_ext = os.path.splitext(filename)[0]

    name_without_ext = name_without_ext.split('_')[0]
    
    # # 尝试匹配不同的格式
    # # 格式1: 汉字+字母+数字_数字 (如: 黑RE2881_5452)
    # match1 = re.match(r'^([\u4e00-\u9fff]+[A-Z]+\d+)_', name_without_ext)
    # if match1:
    #     return match1.group(1)
    
    # # 格式2: 数字+汉字_数字 (如: 171001使_111)  
    # match2 = re.match(r'^(\d+[\u4e00-\u9fff]+)_', name_without_ext)
    # if match2:
    #     return match2.group(1)
    
    # # 格式3: 纯数字开头 (如: 171001_111)
    # match3 = re.match(r'^(\d+)_', name_without_ext)
    # if match3:
    #     return match3.group(1)
        
    # # 格式4: 开头的数字部分 (兼容原有格式)
    # match4 = re.match(r'^(\d+)', name_without_ext)
    # if match4:
    #     return match4.group(1)
    
    # 如果都不匹配，返回完整的文件名（不含扩展名）
    return name_without_ext

def generate_label_txt(image_dir, output_txt_path, image_prefix="images/"):
    """
    生成训练用的标注文件
    
    Args:
        image_dir: 图片目录路径
        output_txt_path: 输出txt文件路径
        image_prefix: 图片路径前缀（如 "images/" 或 "train_images/"）
    """
    image_path = Path(image_dir)
    
    if not image_path.exists():
        print(f"图片目录不存在: {image_dir}")
        return
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 收集所有图片文件和对应标签
    image_label_pairs = []
    
    for file_path in image_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            filename = file_path.name
            label = extract_label_from_filename(filename)
            
            # 构建图片相对路径
            relative_path = image_prefix + filename
            
            image_label_pairs.append((relative_path, label))
            print(f"处理: {filename} -> 标签: {label}")
    
    # 按文件名排序
    image_label_pairs.sort(key=lambda x: x[0])
    
    # 写入txt文件
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for image_path_str, label in image_label_pairs:
                f.write(f"{image_path_str}\t{label}\n")
        
        print(f"\n成功生成标注文件: {output_txt_path}")
        print(f"共处理 {len(image_label_pairs)} 个图片文件")
        
        # 显示前几行作为预览
        print("\n文件内容预览:")
        with open(output_txt_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < 10:  # 显示前10行
                    print(f"  {line.strip()}")
                else:
                    print(f"  ... (共{len(image_label_pairs)}行)")
                    break
                    
    except Exception as e:
        print(f"写入文件时出错: {e}")

def generate_custom_labels(image_dir, output_txt_path, custom_labels_dict=None):
    """
    使用自定义标签字典生成标注文件
    
    Args:
        image_dir: 图片目录路径
        output_txt_path: 输出txt文件路径
        custom_labels_dict: 自定义标签字典 {前缀: 标签}
    """
    if custom_labels_dict is None:
        # 示例自定义标签
        custom_labels_dict = {
            "黑PE": "黑色车牌PE",
            "黑PW": "黑色车牌PW", 
            "黑RE": "黑色车牌RE",
            "171001": "车牌171001",
            # 可以根据需要添加更多映射
        }
    
    image_path = Path(image_dir)
    
    if not image_path.exists():
        print(f"图片目录不存在: {image_dir}")
        return
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # 收集所有图片文件和对应标签
    image_label_pairs = []
    
    for file_path in image_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            filename = file_path.name
            extracted_prefix = extract_label_from_filename(filename)
            
            # 尝试匹配自定义标签
            custom_label = None
            for prefix, label in custom_labels_dict.items():
                if extracted_prefix and extracted_prefix.startswith(prefix):
                    custom_label = label
                    break
            
            # 如果没有找到自定义标签，使用提取的前缀
            final_label = custom_label if custom_label else extracted_prefix
            
            # 构建图片相对路径
            relative_path = "images/" + filename
            
            image_label_pairs.append((relative_path, final_label))
            print(f"处理: {filename} -> 前缀: {extracted_prefix} -> 标签: {final_label}")
    
    # 按文件名排序
    image_label_pairs.sort(key=lambda x: x[0])
    
    # 写入txt文件
    try:
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for image_path_str, label in image_label_pairs:
                f.write(f"{image_path_str}\t{label}\n")
        
        print(f"\n成功生成自定义标注文件: {output_txt_path}")
        print(f"共处理 {len(image_label_pairs)} 个图片文件")
        
    except Exception as e:
        print(f"写入文件时出错: {e}")

def main():
    """
    主函数
    """
    print("=== 生成训练标注文件工具 ===\n")
    
    # 获取用户输入
    image_directory = input("请输入图片目录路径: ").strip()
    if not image_directory:
        print("使用默认路径: ./images")
        image_directory = "./images"
    
    output_file = input("请输入输出txt文件路径 (默认: ./labels.txt): ").strip()
    if not output_file:
        output_file = "./labels.txt"
    
    image_prefix = input("请输入图片路径前缀 (默认: images/): ").strip()
    if not image_prefix:
        image_prefix = "images/"
    
    print("\n选择标注模式:")
    print("1. 自动提取 - 直接使用文件名前缀作为标签")
    print("2. 自定义映射 - 使用预定义的标签映射")
    
    choice = input("请选择 (1/2，默认为1): ").strip()
    
    print(f"\n开始处理...")
    print(f"图片目录: {image_directory}")
    print(f"输出文件: {output_file}")
    print(f"图片前缀: {image_prefix}")
    print("-" * 50)
    
    if choice == "2":
        generate_custom_labels(image_directory, output_file)
    else:
        generate_label_txt(image_directory, output_file, image_prefix)

if __name__ == "__main__":
    main() 