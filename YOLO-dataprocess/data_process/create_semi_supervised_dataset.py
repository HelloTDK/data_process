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

def create_semi_supervised_dataset(
    source_train_image_dir,
    source_val_image_dir,
    source_train_annotation_file,
    source_val_annotation_file,
    output_dir,
    unlabeled_count=5000,
    supervised_train_ratio=0.8
):
    """
    从COCO数据集创建半监督学习数据集
    
    Args:
        source_train_image_dir: 原始训练图片目录 (train2017) - 用于抽取无监督数据
        source_val_image_dir: 原始验证图片目录 (val2017) - 用于有监督训练和验证
        source_train_annotation_file: 原始训练标注文件 (instances_train2017.json)
        source_val_annotation_file: 原始验证标注文件 (instances_val2017.json)
        output_dir: 输出目录
        unlabeled_count: 无监督数据数量
        supervised_train_ratio: 有监督数据中训练集的比例
    """
    
    # 创建输出目录结构
    output_path = Path(output_dir)
    
    # 创建目录结构
    train_img_dir = output_path / 'train2017'
    val_img_dir = output_path / 'val2017'
    unlabeled_img_dir = output_path / 'unlabeled2017'
    annotations_dir = output_path / 'annotations'
    
    # 创建所有必要的目录
    train_img_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_img_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载训练集标注数据（用于无监督数据）
    print(f"正在加载训练集标注文件: {source_train_annotation_file}")
    train_coco_data = load_coco_annotations(source_train_annotation_file)
    
    # 加载验证集标注数据（用于有监督数据）
    print(f"正在加载验证集标注文件: {source_val_annotation_file}")
    val_coco_data = load_coco_annotations(source_val_annotation_file)
    
    # 1. 从train2017中抽取无监督数据
    train_images = train_coco_data['images']
    print(f"train2017共有 {len(train_images)} 张图片")
    
    # 随机抽取无监督数据
    random.shuffle(train_images)
    if len(train_images) < unlabeled_count:
        unlabeled_images = train_images[:len(train_images)//2]
        print(f"警告: 训练集图片数量不足，无监督数据使用 {len(unlabeled_images)} 张图片")
    else:
        unlabeled_images = train_images[:unlabeled_count]
    
    # 2. 从val2017中分割有监督的训练集和验证集
    val_images = val_coco_data['images']
    print(f"val2017共有 {len(val_images)} 张图片")
    
    # 随机打乱val2017图片
    random.shuffle(val_images)
    
    # 分割有监督数据
    supervised_train_count = int(len(val_images) * supervised_train_ratio)
    supervised_train_images = val_images[:supervised_train_count]
    supervised_val_images = val_images[supervised_train_count:]
    
    print(f"无监督数据: {len(unlabeled_images)} 张图片 (来自train2017)")
    print(f"有监督训练集: {len(supervised_train_images)} 张图片 (来自val2017)")
    print(f"有监督验证集: {len(supervised_val_images)} 张图片 (来自val2017)")
    
    # 获取图片ID集合
    unlabeled_img_ids = {img['id'] for img in unlabeled_images}
    supervised_train_img_ids = {img['id'] for img in supervised_train_images}
    supervised_val_img_ids = {img['id'] for img in supervised_val_images}
    
    # 复制图片文件
    print("正在复制无监督数据图片...")
    for img_info in unlabeled_images:
        src_path = Path(source_train_image_dir) / img_info['file_name']
        dst_path = unlabeled_img_dir / img_info['file_name']
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            print(f"警告: 图片文件不存在 {src_path}")
    
    print("正在复制有监督训练集图片...")
    for img_info in supervised_train_images:
        src_path = Path(source_val_image_dir) / img_info['file_name']
        dst_path = train_img_dir / img_info['file_name']
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            print(f"警告: 图片文件不存在 {src_path}")
    
    print("正在复制有监督验证集图片...")
    for img_info in supervised_val_images:
        src_path = Path(source_val_image_dir) / img_info['file_name']
        dst_path = val_img_dir / img_info['file_name']
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            print(f"警告: 图片文件不存在 {src_path}")
    
    # 创建标注文件
    print("正在生成标注文件...")
    
    # 1. 创建无监督数据的图片信息文件 (image_info_unlabeled2017.json)
    unlabeled_info = {
        'info': train_coco_data['info'],
        'licenses': train_coco_data['licenses'],
        'images': unlabeled_images
    }
    
    # 2. 创建有监督训练集标注
    supervised_train_annotations = []
    for ann in val_coco_data['annotations']:
        if ann['image_id'] in supervised_train_img_ids:
            supervised_train_annotations.append(ann)
    
    supervised_train_coco = {
        'info': val_coco_data['info'],
        'licenses': val_coco_data['licenses'],
        'categories': val_coco_data['categories'],
        'images': supervised_train_images,
        'annotations': supervised_train_annotations
    }
    
    # 3. 创建有监督验证集标注
    supervised_val_annotations = []
    for ann in val_coco_data['annotations']:
        if ann['image_id'] in supervised_val_img_ids:
            supervised_val_annotations.append(ann)
    
    supervised_val_coco = {
        'info': val_coco_data['info'],
        'licenses': val_coco_data['licenses'],
        'categories': val_coco_data['categories'],
        'images': supervised_val_images,
        'annotations': supervised_val_annotations
    }
    
    # 4. 创建unlabeled数据的标注文件（用于某些需要标注信息的半监督算法）
    unlabeled_annotations = []
    for ann in train_coco_data['annotations']:
        if ann['image_id'] in unlabeled_img_ids:
            unlabeled_annotations.append(ann)
    
    unlabeled_coco = {
        'info': train_coco_data['info'],
        'licenses': train_coco_data['licenses'],
        'categories': train_coco_data['categories'],
        'images': unlabeled_images,
        'annotations': unlabeled_annotations
    }
    
    # 保存标注文件
    train_ann_file = annotations_dir / 'instances_train2017.json'
    val_ann_file = annotations_dir / 'instances_val2017.json'
    unlabeled_info_file = annotations_dir / 'image_info_unlabeled2017.json'
    unlabeled_ann_file = annotations_dir / 'instances_unlabeled2017.json'
    
    with open(train_ann_file, 'w', encoding='utf-8') as f:
        json.dump(supervised_train_coco, f, ensure_ascii=False, indent=2)
    
    with open(val_ann_file, 'w', encoding='utf-8') as f:
        json.dump(supervised_val_coco, f, ensure_ascii=False, indent=2)
    
    with open(unlabeled_info_file, 'w', encoding='utf-8') as f:
        json.dump(unlabeled_info, f, ensure_ascii=False, indent=2)
    
    with open(unlabeled_ann_file, 'w', encoding='utf-8') as f:
        json.dump(unlabeled_coco, f, ensure_ascii=False, indent=2)
    
    print(f"有监督训练集标注已保存: {train_ann_file}")
    print(f"有监督验证集标注已保存: {val_ann_file}")
    print(f"无监督数据信息已保存: {unlabeled_info_file}")
    print(f"无监督数据标注已保存: {unlabeled_ann_file}")
    
    print(f"有监督训练集标注数量: {len(supervised_train_annotations)}")
    print(f"有监督验证集标注数量: {len(supervised_val_annotations)}")
    print(f"无监督数据标注数量: {len(unlabeled_annotations)}")
    
    # 创建数据集信息文件
    info_file = output_path / 'dataset_info.txt'
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Semi-Supervised COCO Dataset Information\n")
        f.write(f"=========================================\n")
        f.write(f"Original train2017 images: {len(train_images)}\n")
        f.write(f"Original val2017 images: {len(val_images)}\n")
        f.write(f"Unlabeled images: {len(unlabeled_images)} (from train2017)\n")
        f.write(f"Supervised train images: {len(supervised_train_images)} (from val2017)\n")
        f.write(f"Supervised val images: {len(supervised_val_images)} (from val2017)\n")
        f.write(f"Supervised train annotations: {len(supervised_train_annotations)}\n")
        f.write(f"Supervised val annotations: {len(supervised_val_annotations)}\n")
        f.write(f"Unlabeled annotations: {len(unlabeled_annotations)}\n")
        f.write(f"Categories: {len(train_coco_data['categories'])}\n")
        f.write(f"\nDataset structure:\n")
        f.write(f"├── train2017/ (有监督训练集，来自val2017)\n")
        f.write(f"├── val2017/ (有监督验证集，来自val2017)\n")
        f.write(f"├── unlabeled2017/ (无监督数据，来自train2017)\n")
        f.write(f"└── annotations/\n")
        f.write(f"    ├── instances_train2017.json\n")
        f.write(f"    ├── instances_val2017.json\n")
        f.write(f"    ├── image_info_unlabeled2017.json\n")
        f.write(f"    └── instances_unlabeled2017.json\n")
    
    print(f"数据集信息已保存: {info_file}")
    print("半监督学习数据集创建完成！")

def main():
    """主函数"""
    # 设置随机种子以确保可重复性
    random.seed(42)
    
    # 配置路径
    source_train_image_dir = r"D:\Data\公开数据集\detection\coco2017\train2017"
    source_val_image_dir = r"D:\Data\公开数据集\detection\coco2017\val2017"
    source_train_annotation_file = r"D:\Data\公开数据集\detection\coco2017\annotations_trainval2017\annotations\instances_train2017.json"
    source_val_annotation_file = r"D:\Data\公开数据集\detection\coco2017\annotations_trainval2017\annotations\instances_val2017.json"
    output_dir = r"D:\Data\公开数据集\detection\coco2017\semi_supervised_coco"
    
    # 检查输入路径是否存在
    if not os.path.exists(source_train_image_dir):
        print(f"错误: 训练图片目录不存在 {source_train_image_dir}")
        return
    
    if not os.path.exists(source_val_image_dir):
        print(f"错误: 验证图片目录不存在 {source_val_image_dir}")
        return
    
    if not os.path.exists(source_train_annotation_file):
        print(f"错误: 训练标注文件不存在 {source_train_annotation_file}")
        return
    
    if not os.path.exists(source_val_annotation_file):
        print(f"错误: 验证标注文件不存在 {source_val_annotation_file}")
        return
    
    print("开始创建半监督学习COCO数据集...")
    print(f"源训练图片目录: {source_train_image_dir} (用于无监督数据)")
    print(f"源验证图片目录: {source_val_image_dir} (用于有监督训练和验证)")
    print(f"源训练标注文件: {source_train_annotation_file}")
    print(f"源验证标注文件: {source_val_annotation_file}")
    print(f"输出目录: {output_dir}")
    
    # 创建半监督学习数据集
    create_semi_supervised_dataset(
        source_train_image_dir=source_train_image_dir,
        source_val_image_dir=source_val_image_dir,
        source_train_annotation_file=source_train_annotation_file,
        source_val_annotation_file=source_val_annotation_file,
        output_dir=output_dir,
        unlabeled_count=5000
    )

if __name__ == "__main__":
    main() 