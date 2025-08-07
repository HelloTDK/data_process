#!/usr/bin/env python3
"""
测试多格式数据增强工具的格式检测和转换功能
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from yolo_augmenter import YOLOAugmenter

def create_test_data():
    """创建测试数据"""
    # 创建测试目录
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "yolo_labels"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "xml_annotations"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "json_annotations"), exist_ok=True)
    
    # 创建测试YOLO标注
    yolo_content = """0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.1 0.2"""
    with open(os.path.join(test_dir, "yolo_labels", "test.txt"), 'w') as f:
        f.write(yolo_content)
    
    # 创建测试XML标注
    xml_content = """<?xml version='1.0' encoding='utf-8'?>
<annotation>
    <folder>images</folder>
    <filename>test.jpg</filename>
    <path>test.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>640</width>
        <height>480</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>person</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>224</xmin>
            <ymin>144</ymin>
            <xmax>416</xmax>
            <ymax>336</ymax>
        </bndbox>
    </object>
    <object>
        <name>car</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>64</xmin>
            <ymin>96</ymin>
            <xmax>128</xmax>
            <ymax>192</ymax>
        </bndbox>
    </object>
</annotation>"""
    with open(os.path.join(test_dir, "xml_annotations", "test.xml"), 'w', encoding='utf-8') as f:
        f.write(xml_content)
    
    # 创建测试JSON标注
    json_content = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [
            {
                "label": "person",
                "points": [[224, 144], [416, 336]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            },
            {
                "label": "car", 
                "points": [[64, 96], [128, 192]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
        ],
        "imagePath": "test.jpg",
        "imageData": None,
        "imageHeight": 480,
        "imageWidth": 640
    }
    with open(os.path.join(test_dir, "json_annotations", "test.json"), 'w', encoding='utf-8') as f:
        json.dump(json_content, f, ensure_ascii=False, indent=2)
    
    print(f"测试数据已创建在 {test_dir} 目录中")

def test_format_detection():
    """测试格式检测功能"""
    augmenter = YOLOAugmenter()
    
    # 测试YOLO格式检测
    yolo_format = augmenter.detect_annotation_format("test_data/yolo_labels")
    print(f"YOLO格式检测结果: {yolo_format}")
    
    # 测试XML格式检测
    xml_format = augmenter.detect_annotation_format("test_data/xml_annotations")
    print(f"XML格式检测结果: {xml_format}")
    
    # 测试JSON格式检测
    json_format = augmenter.detect_annotation_format("test_data/json_annotations")
    print(f"JSON格式检测结果: {json_format}")

def test_annotation_loading():
    """测试标注加载功能"""
    augmenter = YOLOAugmenter()
    
    # 测试YOLO标注加载
    yolo_bboxes = augmenter.load_yolo_annotation("test_data/yolo_labels/test.txt")
    print(f"YOLO标注加载结果: {yolo_bboxes}")
    
    # 测试XML标注加载
    xml_bboxes = augmenter.load_labelimg_annotation("test_data/xml_annotations/test.xml", 640, 480)
    print(f"XML标注加载结果: {xml_bboxes}")
    
    # 测试JSON标注加载
    json_bboxes = augmenter.load_labelme_annotation("test_data/json_annotations/test.json", 640, 480)
    print(f"JSON标注加载结果: {json_bboxes}")

def main():
    """主函数"""
    print("=" * 50)
    print("多格式数据增强工具测试")
    print("=" * 50)
    
    # 创建测试数据
    create_test_data()
    print()
    
    # 测试格式检测
    print("测试格式检测功能:")
    test_format_detection()
    print()
    
    # 测试标注加载
    print("测试标注加载功能:")
    test_annotation_loading()
    print()
    
    print("测试完成！")

if __name__ == "__main__":
    main()