#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试中文文件名转换功能
确保图像文件和标签文件使用相同的英文基础名
"""

import os
import json
from anylabel2yolo import AnyLabel2YOLO

def create_test_data():
    """创建测试数据"""
    print("=== 创建测试数据 ===")
    
    # 创建测试目录
    test_dir = "test_chinese_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # 模拟中文文件
    test_files = [
        "中文图片1.jpg",
        "中文图片1.json", 
        "狗狗照片.png",
        "狗狗照片.json",
        "汽车检测.jpeg",
        "汽车检测.json"
    ]
    
    # 创建模拟的JSON标注文件
    for filename in test_files:
        if filename.endswith('.json'):
            image_name = filename.replace('.json', '.jpg').replace('.json', '.png').replace('.json', '.jpeg')
            if '狗狗' in filename:
                image_name = filename.replace('.json', '.png')
            elif '汽车' in filename:
                image_name = filename.replace('.json', '.jpeg')
            
            sample_json = {
                "version": "5.0.1",
                "flags": {},
                "shapes": [
                    {
                        "label": "目标",
                        "points": [[100, 100], [200, 200]],
                        "group_id": None,
                        "shape_type": "rectangle",
                        "flags": {}
                    }
                ],
                "imagePath": image_name,
                "imageData": None,
                "imageHeight": 480,
                "imageWidth": 640
            }
            
            file_path = os.path.join(test_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sample_json, f, ensure_ascii=False, indent=2)
            print(f"创建标注文件: {filename}")
        
        elif filename.endswith(('.jpg', '.png', '.jpeg')):
            # 创建空的图像文件（模拟）
            file_path = os.path.join(test_dir, filename)
            with open(file_path, 'w') as f:
                f.write("# 这是模拟的图像文件")
            print(f"创建图像文件: {filename}")
    
    print(f"测试数据已创建在: {test_dir}")
    return test_dir

def test_conversion():
    """测试转换功能"""
    print("\n=== 测试中文文件名转换 ===")
    
    # 创建测试数据
    test_dir = create_test_data()
    output_dir = "test_output_yolo"
    
    # 执行转换
    converter = AnyLabel2YOLO(
        label_path=test_dir,
        image_path=test_dir,
        output_path=output_dir,
        get_dataset_yaml=True,
        label_type=1,
        convert_chinese=True  # 启用中文转换
    )
    
    print("\n开始转换...")
    converter.batch_transform()
    
    # 检查结果
    print("\n=== 检查转换结果 ===")
    
    # 检查输出目录结构
    if os.path.exists(output_dir):
        print(f"✓ 输出目录已创建: {output_dir}")
        
        # 检查子目录
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        
        if os.path.exists(images_dir):
            image_files = os.listdir(images_dir)
            print(f"✓ 图像文件 ({len(image_files)}): {image_files}")
        
        if os.path.exists(labels_dir):
            label_files = os.listdir(labels_dir)
            print(f"✓ 标签文件 ({len(label_files)}): {label_files}")
        
        # 检查文件名匹配
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            image_bases = [os.path.splitext(f)[0] for f in os.listdir(images_dir)]
            label_bases = [os.path.splitext(f)[0] for f in os.listdir(labels_dir)]
            
            print(f"\n=== 文件名匹配检查 ===")
            print(f"图像基础名: {image_bases}")
            print(f"标签基础名: {label_bases}")
            
            # 检查是否匹配
            if set(image_bases) == set(label_bases):
                print("✅ 完美！图像和标签文件名完全匹配")
            else:
                print("❌ 警告：图像和标签文件名不匹配")
                print(f"仅在图像中: {set(image_bases) - set(label_bases)}")
                print(f"仅在标签中: {set(label_bases) - set(image_bases)}")
        
        # 检查映射文件
        mapping_file = os.path.join(output_dir, "filename_mapping.json")
        if os.path.exists(mapping_file):
            print(f"\n=== 文件名映射 ===")
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            print("映射关系:")
            for key, value in mapping_data.get("映射关系", {}).items():
                print(f"  {key} -> {value}")
    
    print(f"\n测试完成！请检查输出目录: {output_dir}")

def demonstrate_matching():
    """演示文件名匹配逻辑"""
    print("=== 文件名匹配演示 ===")
    print("原始文件:")
    print("  中文图片1.jpg")
    print("  中文图片1.json")
    print("")
    print("转换后:")
    print("  abc123def.jpg   (图像文件)")
    print("  abc123def.txt   (YOLO标签文件)")
    print("")
    print("关键点:")
    print("✓ 相同基础名: abc123def")
    print("✓ 不同扩展名: .jpg 和 .txt")
    print("✓ YOLO可以正确匹配它们")

if __name__ == "__main__":
    demonstrate_matching()
    
    # 取消注释以运行实际测试
    # test_conversion() 