import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def check_coco_dataset(dataset_dir):
    """检查COCO数据集的质量和完整性"""
    
    dataset_path = Path(dataset_dir)
    
    # 检查目录结构
    train_img_dir = dataset_path / 'images' / 'train2017'
    val_img_dir = dataset_path / 'images' / 'val2017'
    annotations_dir = dataset_path / 'annotations'
    
    train_ann_file = annotations_dir / 'instances_train2017.json'
    val_ann_file = annotations_dir / 'instances_val2017.json'
    
    print("=== 数据集目录结构检查 ===")
    print(f"训练图片目录: {train_img_dir} - {'存在' if train_img_dir.exists() else '不存在'}")
    print(f"验证图片目录: {val_img_dir} - {'存在' if val_img_dir.exists() else '不存在'}")
    print(f"标注目录: {annotations_dir} - {'存在' if annotations_dir.exists() else '不存在'}")
    print(f"训练标注文件: {train_ann_file} - {'存在' if train_ann_file.exists() else '不存在'}")
    print(f"验证标注文件: {val_ann_file} - {'存在' if val_ann_file.exists() else '不存在'}")
    
    if not all([train_img_dir.exists(), val_img_dir.exists(), 
                train_ann_file.exists(), val_ann_file.exists()]):
        print("❌ 数据集目录结构不完整！")
        return False
    
    # 统计图片数量
    train_imgs = list(train_img_dir.glob('*.jpg'))
    val_imgs = list(val_img_dir.glob('*.jpg'))
    
    print(f"\n=== 图片数量统计 ===")
    print(f"训练集图片数量: {len(train_imgs)}")
    print(f"验证集图片数量: {len(val_imgs)}")
    
    # 检查标注文件
    print(f"\n=== 标注文件检查 ===")
    
    # 检查训练集标注
    with open(train_ann_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    with open(val_ann_file, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    print(f"训练集标注 - 图片: {len(train_data['images'])}, 标注: {len(train_data['annotations'])}")
    print(f"验证集标注 - 图片: {len(val_data['images'])}, 标注: {len(val_data['annotations'])}")
    
    # 检查类别信息
    print(f"\n=== 类别信息检查 ===")
    categories = train_data['categories']
    print(f"类别数量: {len(categories)}")
    
    # 检查类别ID是否连续
    cat_ids = [cat['id'] for cat in categories]
    print(f"类别ID范围: {min(cat_ids)} - {max(cat_ids)}")
    
    # 检查是否有重复的类别ID
    if len(cat_ids) != len(set(cat_ids)):
        print("❌ 发现重复的类别ID！")
        return False
    
    # 检查类别ID是否从0或1开始
    if min(cat_ids) not in [0, 1]:
        print(f"⚠️  类别ID不是从0或1开始，而是从{min(cat_ids)}开始")
        print("这可能导致YOLO训练问题，建议重新映射类别ID")
    
    # 统计每个类别的标注数量
    train_cat_counts = Counter(ann['category_id'] for ann in train_data['annotations'])
    val_cat_counts = Counter(ann['category_id'] for ann in val_data['annotations'])
    
    print(f"\n=== 类别分布统计 ===")
    print("训练集前10个类别的标注数量:")
    for cat_id, count in train_cat_counts.most_common(10):
        cat_name = next(cat['name'] for cat in categories if cat['id'] == cat_id)
        print(f"  {cat_name} (ID:{cat_id}): {count}")
    
    print("\n验证集前10个类别的标注数量:")
    for cat_id, count in val_cat_counts.most_common(10):
        cat_name = next(cat['name'] for cat in categories if cat['id'] == cat_id)
        print(f"  {cat_name} (ID:{cat_id}): {count}")
    
    # 检查是否有空的标注
    empty_train_imgs = 0
    empty_val_imgs = 0
    
    train_img_ids = {img['id'] for img in train_data['images']}
    val_img_ids = {img['id'] for img in val_data['images']}
    
    train_ann_img_ids = {ann['image_id'] for ann in train_data['annotations']}
    val_ann_img_ids = {ann['image_id'] for ann in val_data['annotations']}
    
    empty_train_imgs = len(train_img_ids - train_ann_img_ids)
    empty_val_imgs = len(val_img_ids - val_ann_img_ids)
    
    print(f"\n=== 空标注检查 ===")
    print(f"训练集无标注图片数量: {empty_train_imgs}")
    print(f"验证集无标注图片数量: {empty_val_imgs}")
    
    if empty_train_imgs > len(train_imgs) * 0.1:
        print("⚠️  训练集中空标注图片过多，可能影响训练效果")
    
    # 检查标注框的大小分布
    print(f"\n=== 标注框大小分布 ===")
    
    train_areas = []
    for ann in train_data['annotations']:
        if 'area' in ann:
            train_areas.append(ann['area'])
        elif 'bbox' in ann:
            # bbox格式: [x, y, width, height]
            train_areas.append(ann['bbox'][2] * ann['bbox'][3])
    
    if train_areas:
        train_areas = np.array(train_areas)
        print(f"训练集标注框面积统计:")
        print(f"  平均面积: {np.mean(train_areas):.2f}")
        print(f"  中位数面积: {np.median(train_areas):.2f}")
        print(f"  最小面积: {np.min(train_areas):.2f}")
        print(f"  最大面积: {np.max(train_areas):.2f}")
        
        # 检查是否有异常小的标注框
        very_small = np.sum(train_areas < 100)
        print(f"  面积小于100的标注框数量: {very_small}")
    
    # 生成修复建议
    print(f"\n=== 修复建议 ===")
    
    issues = []
    
    if min(cat_ids) not in [0, 1]:
        issues.append("类别ID需要重新映射到从0或1开始")
    
    if empty_train_imgs > 0:
        issues.append(f"有{empty_train_imgs}张训练图片没有标注")
    
    if len(train_data['annotations']) < 100:
        issues.append("训练集标注数量太少，建议增加到至少每类100个标注")
    
    if not issues:
        print("✅ 数据集质量检查通过！")
        return True
    else:
        print("❌ 发现以下问题需要修复:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        return False

def main():
    """主函数"""
    dataset_dir = r"D:\Data\公开数据集\detection\coco2017\small_coco"
    
    if not os.path.exists(dataset_dir):
        print(f"错误: 数据集目录不存在 {dataset_dir}")
        return
    
    print("开始检查COCO数据集质量...")
    print(f"数据集目录: {dataset_dir}")
    
    is_valid = check_coco_dataset(dataset_dir)
    
    if not is_valid:
        print("\n建议运行修复脚本来解决这些问题。")

if __name__ == "__main__":
    main() 