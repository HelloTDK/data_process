import json
import os
import random
import shutil
from pathlib import Path

def load_coco_annotations(annotation_file):
    """加载COCO标注文件"""
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_small_coco_dataset(
    source_image_dir,
    source_annotation_file,
    output_dir,
    total_images=1000,
    train_ratio=0.8
):
    """
    从COCO数据集创建小型数据集
    
    Args:
        source_image_dir: 原始图片目录
        source_annotation_file: 原始标注文件
        output_dir: 输出目录
        total_images: 总图片数量
        train_ratio: 训练集比例
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    train_img_dir = output_path / 'images' / 'train2017'
    val_img_dir = output_path / 'images' / 'val2017'
    annotations_dir = output_path / 'annotations'
    
    # 创建目录
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"正在加载标注文件: {source_annotation_file}")
    # 加载原始标注数据
    coco_data = load_coco_annotations(source_annotation_file)
    
    # 获取所有图片信息
    all_images = coco_data['images']
    print(f"原始数据集共有 {len(all_images)} 张图片")
    
    # 随机抽取指定数量的图片
    if len(all_images) < total_images:
        selected_images = all_images
        print(f"原始数据集图片数量不足，将使用全部 {len(all_images)} 张图片")
    else:
        selected_images = random.sample(all_images, total_images)
        print(f"随机抽取了 {total_images} 张图片")
    
    # 计算训练集和验证集数量
    train_count = int(len(selected_images) * train_ratio)
    val_count = len(selected_images) - train_count
    
    # 分割图片
    train_images = selected_images[:train_count]
    val_images = selected_images[train_count:]
    
    print(f"训练集: {len(train_images)} 张图片")
    print(f"验证集: {len(val_images)} 张图片")
    
    # 获取选中图片的ID集合
    train_img_ids = {img['id'] for img in train_images}
    val_img_ids = {img['id'] for img in val_images}
    
    # 复制图片文件
    print("正在复制训练集图片...")
    for img_info in train_images:
        src_path = Path(source_image_dir) / img_info['file_name']
        dst_path = train_img_dir / img_info['file_name']
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            print(f"警告: 图片文件不存在 {src_path}")
    
    print("正在复制验证集图片...")
    for img_info in val_images:
        src_path = Path(source_image_dir) / img_info['file_name']
        dst_path = val_img_dir / img_info['file_name']
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            print(f"警告: 图片文件不存在 {src_path}")
    
    # 过滤标注数据
    print("正在生成标注文件...")
    
    # 创建训练集标注
    train_annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] in train_img_ids:
            train_annotations.append(ann)
    
    train_coco = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'categories': coco_data['categories'],
        'images': train_images,
        'annotations': train_annotations
    }
    
    # 创建验证集标注
    val_annotations = []
    for ann in coco_data['annotations']:
        if ann['image_id'] in val_img_ids:
            val_annotations.append(ann)
    
    val_coco = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'categories': coco_data['categories'],
        'images': val_images,
        'annotations': val_annotations
    }
    
    # 保存标注文件
    train_ann_file = annotations_dir / 'instances_train2017.json'
    val_ann_file = annotations_dir / 'instances_val2017.json'
    
    with open(train_ann_file, 'w', encoding='utf-8') as f:
        json.dump(train_coco, f, ensure_ascii=False, indent=2)
    
    with open(val_ann_file, 'w', encoding='utf-8') as f:
        json.dump(val_coco, f, ensure_ascii=False, indent=2)
    
    print(f"训练集标注已保存: {train_ann_file}")
    print(f"验证集标注已保存: {val_ann_file}")
    print(f"训练集标注数量: {len(train_annotations)}")
    print(f"验证集标注数量: {len(val_annotations)}")
    
    # 创建数据集信息文件
    info_file = output_path / 'dataset_info.txt'
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Small COCO Dataset Information\n")
        f.write(f"================================\n")
        f.write(f"Total images: {len(selected_images)}\n")
        f.write(f"Train images: {len(train_images)}\n")
        f.write(f"Val images: {len(val_images)}\n")
        f.write(f"Train annotations: {len(train_annotations)}\n")
        f.write(f"Val annotations: {len(val_annotations)}\n")
        f.write(f"Categories: {len(coco_data['categories'])}\n")
    
    print(f"数据集信息已保存: {info_file}")
    print("Small COCO数据集创建完成！")

def main():
    """主函数"""
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 配置路径
    source_image_dir = r"D:\Data\公开数据集\detection\coco2017\val2017"
    source_annotation_file = r"D:\Data\公开数据集\detection\coco2017\annotations_trainval2017\annotations\instances_val2017.json"
    output_dir = r"D:\Data\公开数据集\detection\coco2017\small_coco"
    
    # 检查输入路径是否存在
    if not os.path.exists(source_image_dir):
        print(f"错误: 图片目录不存在 {source_image_dir}")
        return
    
    if not os.path.exists(source_annotation_file):
        print(f"错误: 标注文件不存在 {source_annotation_file}")
        return
    
    print("开始创建Small COCO数据集...")
    print(f"源图片目录: {source_image_dir}")
    print(f"源标注文件: {source_annotation_file}")
    print(f"输出目录: {output_dir}")
    
    # 创建小型数据集
    create_small_coco_dataset(
        source_image_dir=source_image_dir,
        source_annotation_file=source_annotation_file,
        output_dir=output_dir,
        total_images=5000,
        train_ratio=0.8
    )

if __name__ == "__main__":
    main()
