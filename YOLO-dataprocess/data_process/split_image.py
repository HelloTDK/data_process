import os
import shutil
from math import ceil
from PIL import Image

def split_images_into_folders(source_dir, n_folders,output):
    """
    读取目录下所有图片，并等分图片为N份（N个文件夹）
    
    参数:
    source_dir: 源图片目录
    n_folders: 要分成的文件夹数量
    """
    os.makedirs(output,exist_ok=True)
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
    
    # 获取所有图片文件
    image_files = []
    for file in os.listdir(source_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            image_files.append(file)
    
    # 计算每个文件夹应该包含的图片数量
    total_images = len(image_files)
    if total_images == 0:
        print("没有找到图片文件")
        return
    
    images_per_folder = ceil(total_images / n_folders)
    
    # 创建目标文件夹并分配图片
    for i in range(n_folders):
        folder_name = f"folder_{i+1}"
        folder_path = os.path.join(output, folder_name)
        
        # 如果文件夹已存在，先删除
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        
        # 创建新文件夹
        os.makedirs(folder_path)
        
        # 计算当前文件夹的图片范围
        start_idx = i * images_per_folder
        end_idx = min((i + 1) * images_per_folder, total_images)
        
        # 复制图片到目标文件夹
        for j in range(start_idx, end_idx):
            src_path = os.path.join(source_dir, image_files[j])
            dst_path = os.path.join(folder_path, image_files[j])
            shutil.copy2(src_path, dst_path)
        
        print(f"已创建 {folder_name}，包含 {end_idx - start_idx} 张图片")

# 使用示例
if __name__ == "__main__":
    # 修改为你的图片目录
    image_directory = r"D:\Desktop\output_ppocr1\output_ppocr1"
    output_dir = r"D:\Desktop\split"
    # 修改为你想要的文件夹数量
    number_of_folders = 4
    
    split_images_into_folders(image_directory, number_of_folders,output_dir)
