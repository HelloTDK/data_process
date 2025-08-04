import sys
import os
import json
import shutil
import base64
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                             QLineEdit, QFileDialog, QProgressBar, QCheckBox,
                             QSpinBox, QGroupBox, QScrollArea, QTextEdit,
                             QSlider, QComboBox, QMessageBox, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QFont

class DataAugmenter:
    """数据增强核心类，支持YOLO、LabelMe和XML格式"""
    
    def __init__(self):
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
    def load_yolo_annotation(self, annotation_path):
        """加载YOLO格式标注文件"""
        if not os.path.exists(annotation_path):
            return []
        
        bboxes = []
        try:
            # 使用UTF-8编码读取文件，支持中文路径
            with open(annotation_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            bboxes.append([x_center, y_center, width, height, class_id])
        except Exception as e:
            print(f"读取标签文件失败 {annotation_path}: {e}")
        return bboxes
    
    def load_labelme_annotation(self, annotation_path, img_width, img_height):
        """加载LabelMe格式标注文件"""
        if not os.path.exists(annotation_path):
            return []
        
        bboxes = []
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for shape in data.get('shapes', []):
                if shape['shape_type'] == 'rectangle':
                    points = shape['points']
                    x1, y1 = points[0]
                    x2, y2 = points[1]
                    
                    # 确保坐标顺序正确
                    x_min = min(x1, x2)
                    x_max = max(x1, x2)
                    y_min = min(y1, y2)
                    y_max = max(y1, y2)
                    
                    # 转换为YOLO格式 (归一化的中心点坐标和宽高)
                    x_center = (x_min + x_max) / 2.0 / img_width
                    y_center = (y_min + y_max) / 2.0 / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height
                    
                    # 使用标签名作为类别
                    label = shape['label']
                    bboxes.append([x_center, y_center, width, height, label])
                    
        except Exception as e:
            print(f"读取LabelMe标签文件失败 {annotation_path}: {e}")
        return bboxes
    
    def load_xml_annotation(self, annotation_path, img_width, img_height):
        """加载LabelImg XML格式标注文件"""
        if not os.path.exists(annotation_path):
            return []
        
        bboxes = []
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                name = obj.find('name').text
                bbox = obj.find('bndbox')
                
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # 转换为YOLO格式 (归一化的中心点坐标和宽高)
                x_center = (xmin + xmax) / 2.0 / img_width
                y_center = (ymin + ymax) / 2.0 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                
                bboxes.append([x_center, y_center, width, height, name])
                
        except Exception as e:
            print(f"读取XML标签文件失败 {annotation_path}: {e}")
        return bboxes
    
    def save_yolo_annotation(self, bboxes, annotation_path):
        """保存YOLO格式标注文件"""
        try:
            # 使用UTF-8编码保存文件，支持中文路径
            with open(annotation_path, 'w', encoding='utf-8') as f:
                for bbox in bboxes:
                    x_center, y_center, width, height, class_id = bbox
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        except Exception as e:
            print(f"保存标签文件失败 {annotation_path}: {e}")
    
    def save_labelme_annotation(self, bboxes, annotation_path, img_width, img_height, image_path):
        """保存LabelMe格式标注文件"""
        try:
            # 读取图片的base64编码
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # 创建LabelMe格式数据
            labelme_data = {
                "version": "5.4.1",
                "flags": {},
                "shapes": [],
                "imagePath": os.path.basename(image_path),
                "imageData": image_base64,
                "imageHeight": img_height,
                "imageWidth": img_width
            }
            
            # 转换边界框
            for bbox in bboxes:
                x_center, y_center, width, height, label = bbox
                
                # 从YOLO格式转回像素坐标
                x_min = (x_center - width / 2) * img_width
                x_max = (x_center + width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                y_max = (y_center + height / 2) * img_height
                
                shape = {
                    "label": str(label),
                    "points": [[x_min, y_min], [x_max, y_max]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {}
                }
                labelme_data["shapes"].append(shape)
            
            # 保存JSON文件
            with open(annotation_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"保存LabelMe标签文件失败 {annotation_path}: {e}")
    
    def save_xml_annotation(self, bboxes, annotation_path, img_width, img_height, image_path):
        """保存LabelImg XML格式标注文件"""
        try:
            # 创建XML根元素
            annotation = ET.Element('annotation')
            
            # 添加文件夹信息
            folder = ET.SubElement(annotation, 'folder')
            folder.text = os.path.dirname(image_path)
            
            # 添加文件名信息
            filename = ET.SubElement(annotation, 'filename')
            filename.text = os.path.basename(image_path)
            
            # 添加路径信息
            path = ET.SubElement(annotation, 'path')
            path.text = image_path
            
            # 添加数据源信息
            source = ET.SubElement(annotation, 'source')
            database = ET.SubElement(source, 'database')
            database.text = 'Unknown'
            
            # 添加图片尺寸信息
            size = ET.SubElement(annotation, 'size')
            width_elem = ET.SubElement(size, 'width')
            width_elem.text = str(img_width)
            height_elem = ET.SubElement(size, 'height')
            height_elem.text = str(img_height)
            depth_elem = ET.SubElement(size, 'depth')
            depth_elem.text = '3'
            
            # 添加分割信息
            segmented = ET.SubElement(annotation, 'segmented')
            segmented.text = '0'
            
            # 添加对象信息
            for bbox in bboxes:
                x_center, y_center, width, height, label = bbox
                
                # 从YOLO格式转回像素坐标
                x_min = int((x_center - width / 2) * img_width)
                x_max = int((x_center + width / 2) * img_width)
                y_min = int((y_center - height / 2) * img_height)
                y_max = int((y_center + height / 2) * img_height)
                
                # 确保坐标在图片范围内
                x_min = max(0, min(x_min, img_width - 1))
                x_max = max(0, min(x_max, img_width - 1))
                y_min = max(0, min(y_min, img_height - 1))
                y_max = max(0, min(y_max, img_height - 1))
                
                obj = ET.SubElement(annotation, 'object')
                
                name = ET.SubElement(obj, 'name')
                name.text = str(label)
                
                pose = ET.SubElement(obj, 'pose')
                pose.text = 'Unspecified'
                
                truncated = ET.SubElement(obj, 'truncated')
                truncated.text = '0'
                
                difficult = ET.SubElement(obj, 'difficult')
                difficult.text = '0'
                
                bndbox = ET.SubElement(obj, 'bndbox')
                xmin_elem = ET.SubElement(bndbox, 'xmin')
                xmin_elem.text = str(x_min)
                ymin_elem = ET.SubElement(bndbox, 'ymin')
                ymin_elem.text = str(y_min)
                xmax_elem = ET.SubElement(bndbox, 'xmax')
                xmax_elem.text = str(x_max)
                ymax_elem = ET.SubElement(bndbox, 'ymax')
                ymax_elem.text = str(y_max)
            
            # 格式化XML并保存
            self._prettify_xml(annotation)
            tree = ET.ElementTree(annotation)
            tree.write(annotation_path, encoding='utf-8', xml_declaration=True)
            
        except Exception as e:
            print(f"保存XML标签文件失败 {annotation_path}: {e}")
    
    def _prettify_xml(self, elem, level=0):
        """格式化XML，使其更易读"""
        indent = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._prettify_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent
    
    def create_augmentation_pipeline(self, aug_params):
        """创建数据增强管道"""
        transforms = []
        
        # 水平翻转
        if aug_params.get('horizontal_flip', False):
            transforms.append(A.HorizontalFlip(p=aug_params.get('horizontal_flip_prob', 0.5)))
        
        # 垂直翻转
        if aug_params.get('vertical_flip', False):
            transforms.append(A.VerticalFlip(p=aug_params.get('vertical_flip_prob', 0.5)))
        
        # 旋转
        if aug_params.get('rotation', False):
            transforms.append(A.Rotate(
                limit=aug_params.get('rotation_limit', 15),
                p=aug_params.get('rotation_prob', 0.5)
            ))
        
        # 缩放 (使用Affine变换实现缩放)
        if aug_params.get('scale', False):
            scale_limit = aug_params.get('scale_limit', 0.2)
            transforms.append(A.Affine(
                scale=(1.0 - scale_limit, 1.0 + scale_limit),
                p=aug_params.get('scale_prob', 0.5)
            ))
        
        # 亮度对比度
        if aug_params.get('brightness_contrast', False):
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=aug_params.get('brightness_limit', 0.2),
                contrast_limit=aug_params.get('contrast_limit', 0.2),
                p=aug_params.get('brightness_contrast_prob', 0.5)
            ))
        
        # 色相饱和度
        if aug_params.get('hue_saturation', False):
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=aug_params.get('hue_limit', 20),
                sat_shift_limit=aug_params.get('saturation_limit', 30),
                val_shift_limit=aug_params.get('value_limit', 20),
                p=aug_params.get('hue_saturation_prob', 0.5)
            ))
        
        # 高斯噪声
        if aug_params.get('gaussian_noise', False):
            transforms.append(A.GaussNoise(
                var_limit=(10, 50),
                p=aug_params.get('gaussian_noise_prob', 0.3)
            ))
        
        # 高斯模糊
        if aug_params.get('gaussian_blur', False):
            transforms.append(A.GaussianBlur(
                blur_limit=(3, 7),
                p=aug_params.get('gaussian_blur_prob', 0.3)
            ))
        
        # 弹性变换
        if aug_params.get('elastic_transform', False):
            transforms.append(A.ElasticTransform(
                alpha=1, sigma=50, alpha_affine=50,
                p=aug_params.get('elastic_transform_prob', 0.3)
            ))
        
        # 网格扭曲
        if aug_params.get('grid_distortion', False):
            transforms.append(A.GridDistortion(
                num_steps=5, distort_limit=0.3,
                p=aug_params.get('grid_distortion_prob', 0.3)
            ))
        
        # 透视变换
        if aug_params.get('perspective', False):
            transforms.append(A.Perspective(
                scale=(0.05, 0.1),
                p=aug_params.get('perspective_prob', 0.3)
            ))
        
        # CLAHE (对比度受限的自适应直方图均衡化)
        if aug_params.get('clahe', False):
            transforms.append(A.CLAHE(
                clip_limit=4.0, tile_grid_size=(8, 8),
                p=aug_params.get('clahe_prob', 0.3)
            ))
        
        return A.Compose(transforms, bbox_params=A.BboxParams(
            format='yolo', 
            label_fields=['class_labels']
        ))
    
    def cv2_imread_unicode(self, file_path):
        """支持中文路径的图片读取"""
        try:
            # 使用numpy读取文件，然后用cv2解码
            img_buffer = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"读取图片失败: {e}")
            return None
    
    def cv2_imwrite_unicode(self, file_path, img):
        """支持中文路径的图片保存"""
        try:
            # 获取文件扩展名
            ext = os.path.splitext(file_path)[1]
            # 编码图片
            is_success, img_buffer = cv2.imencode(ext, img)
            if is_success:
                # 写入文件
                img_buffer.tofile(file_path)
                return True
            return False
        except Exception as e:
            print(f"保存图片失败: {e}")
            return False

    def augment_image_and_labels(self, image_path, annotation_path, aug_pipeline, output_dir, multiplier, format_type='yolo'):
        """对单张图片和标签进行增强"""
        # 加载图片 - 支持中文路径
        image = self.cv2_imread_unicode(image_path)
        if image is None:
            return False, f"无法加载图片: {image_path}"
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        # 根据格式加载标注
        if format_type == 'labelme':
            bboxes = self.load_labelme_annotation(annotation_path, img_width, img_height)
        elif format_type == 'xml':
            bboxes = self.load_xml_annotation(annotation_path, img_width, img_height)
        else:
            bboxes = self.load_yolo_annotation(annotation_path)
        
        # 获取文件名
        image_name = Path(image_path).stem
        image_ext = Path(image_path).suffix
        
        results = []
        
        for i in range(multiplier):
            try:
                if bboxes:
                    # 准备边界框数据
                    bbox_coords = [[bbox[0], bbox[1], bbox[2], bbox[3]] for bbox in bboxes]
                    class_labels = [bbox[4] for bbox in bboxes]
                    
                    # 应用增强
                    augmented = aug_pipeline(
                        image=image, 
                        bboxes=bbox_coords, 
                        class_labels=class_labels
                    )
                    
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_class_labels = augmented['class_labels']
                    
                    # 重新组合边界框数据
                    new_bboxes = []
                    for j, bbox in enumerate(aug_bboxes):
                        new_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3], aug_class_labels[j]])
                else:
                    # 没有标注的情况
                    augmented = aug_pipeline(image=image)
                    aug_image = augmented['image']
                    new_bboxes = []
                
                # 创建输出目录结构
                if format_type in ['labelme', 'xml']:
                    output_images_dir = output_dir
                    output_labels_dir = output_dir
                else:
                    output_images_dir = os.path.join(output_dir, "images")
                    output_labels_dir = os.path.join(output_dir, "labels")
                    os.makedirs(output_images_dir, exist_ok=True)
                    os.makedirs(output_labels_dir, exist_ok=True)
                
                # 保存增强后的图片
                output_image_name = f"{image_name}_aug_{i+1}{image_ext}"
                output_image_path = os.path.join(output_images_dir, output_image_name)
                
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                # 使用支持中文路径的保存方法
                if not self.cv2_imwrite_unicode(output_image_path, aug_image_bgr):
                    return False, f"保存图片失败: {output_image_path}"
                
                # 保存增强后的标注
                if new_bboxes:
                    aug_height, aug_width = aug_image.shape[:2]
                    if format_type == 'labelme':
                        output_annotation_name = f"{image_name}_aug_{i+1}.json"
                        output_annotation_path = os.path.join(output_labels_dir, output_annotation_name)
                        self.save_labelme_annotation(new_bboxes, output_annotation_path, aug_width, aug_height, output_image_path)
                    elif format_type == 'xml':
                        output_annotation_name = f"{image_name}_aug_{i+1}.xml"
                        output_annotation_path = os.path.join(output_labels_dir, output_annotation_name)
                        self.save_xml_annotation(new_bboxes, output_annotation_path, aug_width, aug_height, output_image_path)
                    else:
                        output_annotation_name = f"{image_name}_aug_{i+1}.txt"
                        output_annotation_path = os.path.join(output_labels_dir, output_annotation_name)
                        self.save_yolo_annotation(new_bboxes, output_annotation_path)
                
                results.append((output_image_path, len(new_bboxes)))
                
            except Exception as e:
                return False, f"增强处理失败: {str(e)}"
        
        return True, results


class AugmentationWorker(QThread):
    """数据增强工作线程"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    stats_updated = pyqtSignal(str)  # 新增统计信号
    
    def __init__(self, input_dir, output_dir, aug_params, multiplier, format_type='yolo'):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.aug_params = aug_params
        self.multiplier = multiplier
        self.format_type = format_type
        self.augmenter = DataAugmenter()
        
    def run(self):
        try:
            # 创建输出目录
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 根据格式类型检查输入目录结构
            if self.format_type == 'labelme':
                # LabelMe格式：图片和JSON文件在同一目录
                self.status_updated.emit("LabelMe格式：从根目录读取图片和JSON文件...")
                image_files = []
                for ext in self.augmenter.image_extensions:
                    image_files.extend(Path(self.input_dir).glob(f"*{ext}"))
                    image_files.extend(Path(self.input_dir).glob(f"*{ext.upper()}"))
            elif self.format_type == 'xml':
                # XML格式：图片和XML文件在同一目录
                self.status_updated.emit("XML格式：从根目录读取图片和XML文件...")
                image_files = []
                for ext in self.augmenter.image_extensions:
                    image_files.extend(Path(self.input_dir).glob(f"*{ext}"))
                    image_files.extend(Path(self.input_dir).glob(f"*{ext.upper()}"))
            else:
                # YOLO格式：检查images和labels目录
                images_dir = os.path.join(self.input_dir, "images")
                labels_dir = os.path.join(self.input_dir, "labels")
                
                image_files = []
                
                if os.path.exists(images_dir):
                    # 如果存在images目录，从该目录读取图片
                    self.status_updated.emit("检测到images目录，从中读取图片文件...")
                    for ext in self.augmenter.image_extensions:
                        image_files.extend(Path(images_dir).glob(f"*{ext}"))
                        image_files.extend(Path(images_dir).glob(f"*{ext.upper()}"))
                else:
                    # 否则从根目录读取图片
                    self.status_updated.emit("从根目录读取图片文件...")
                    for ext in self.augmenter.image_extensions:
                        image_files.extend(Path(self.input_dir).glob(f"*{ext}"))
                        image_files.extend(Path(self.input_dir).glob(f"*{ext.upper()}"))
            
            if not image_files:
                self.error_occurred.emit("输入目录中没有找到图片文件")
                return
            
            total_files = len(image_files)
            processed_files = 0
            total_augmented = 0
            files_with_labels = 0
            files_without_labels = 0
            
            self.status_updated.emit(f"找到 {total_files} 张图片，增强倍数: {self.multiplier}，格式: {self.format_type.upper()}")
            
            # 创建增强管道
            aug_pipeline = self.augmenter.create_augmentation_pipeline(self.aug_params)
            
            for image_file in image_files:
                image_path = str(image_file)
                
                # 根据格式类型确定标签文件路径
                if self.format_type == 'labelme':
                    # LabelMe格式：JSON文件在同一目录
                    annotation_path = str(image_file.with_suffix('.json'))
                elif self.format_type == 'xml':
                    # XML格式：XML文件在同一目录
                    annotation_path = str(image_file.with_suffix('.xml'))
                else:
                    # YOLO格式：根据输入目录结构确定标签文件路径
                    images_dir = os.path.join(self.input_dir, "images")
                    labels_dir = os.path.join(self.input_dir, "labels")
                    if os.path.exists(labels_dir):
                        # 如果存在labels目录，从该目录寻找标签文件
                        annotation_filename = image_file.stem + '.txt'
                        annotation_path = os.path.join(labels_dir, annotation_filename)
                    else:
                        # 否则在同目录下寻找标签文件
                        annotation_path = str(image_file.with_suffix('.txt'))
                
                # 检查是否有对应的标签文件
                has_labels = os.path.exists(annotation_path)
                if has_labels:
                    files_with_labels += 1
                else:
                    files_without_labels += 1
                
                self.status_updated.emit(f"正在处理 ({processed_files+1}/{total_files}): {image_file.name}")
                
                success, result = self.augmenter.augment_image_and_labels(
                    image_path, annotation_path, aug_pipeline, 
                    self.output_dir, self.multiplier, self.format_type
                )
                
                if success:
                    # 验证实际生成的图片数量
                    actual_generated = len(result)
                    total_augmented += actual_generated
                    processed_files += 1
                    
                    # 如果生成数量不等于预期，记录详细信息
                    if actual_generated != self.multiplier:
                        self.status_updated.emit(f"注意: {image_file.name} 预期生成 {self.multiplier} 张，实际生成 {actual_generated} 张")
                else:
                    self.error_occurred.emit(f"处理失败: {result}")
                    return
                
                # 更新进度
                progress = int((processed_files / total_files) * 100)
                self.progress_updated.emit(progress)
                
                # 更新实时统计
                current_expected = processed_files * self.multiplier
                stats_text = f"已处理: {processed_files}/{total_files} | 已生成: {total_augmented} 张 (预期: {current_expected} 张)"
                self.stats_updated.emit(stats_text)
            
            # 计算预期数量
            expected_augmented = processed_files * self.multiplier
            
            # 详细的完成信息
            result_message = f"""处理完成！
格式类型: {self.format_type.upper()}
原始图片: {total_files} 张
成功处理: {processed_files} 张
有标签文件: {files_with_labels} 张
无标签文件: {files_without_labels} 张
增强倍数: {self.multiplier}x
预期生成: {expected_augmented} 张
实际生成: {total_augmented} 张"""
            
            if total_augmented != expected_augmented:
                result_message += f"\n⚠️ 注意: 实际生成数量与预期不符！"
            
            self.finished.emit(result_message)
            
        except Exception as e:
            self.error_occurred.emit(f"处理过程中出现错误: {str(e)}")


class DataAugmenterGUI(QMainWindow):
    """数据增强GUI主界面，支持YOLO、LabelMe和XML格式"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker = None
        
    def init_ui(self):
        self.setWindowTitle("数据增强工具 - 支持YOLO/LabelMe/XML格式")
        self.setGeometry(100, 100, 900, 700)
        
        # 创建中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 创建选项卡
        tab_widget = QTabWidget()
        main_layout.addWidget(tab_widget)
        
        # 基本设置选项卡
        basic_tab = QWidget()
        tab_widget.addTab(basic_tab, "基本设置")
        self.setup_basic_tab(basic_tab)
        
        # 增强参数选项卡
        aug_tab = QWidget()
        tab_widget.addTab(aug_tab, "增强参数")
        self.setup_augmentation_tab(aug_tab)
        
        # 进度和日志选项卡
        progress_tab = QWidget()
        tab_widget.addTab(progress_tab, "处理进度")
        self.setup_progress_tab(progress_tab)
        
    def setup_basic_tab(self, tab):
        """设置基本选项卡"""
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # 输入目录选择
        input_group = QGroupBox("输入设置")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)
        
        input_dir_layout = QHBoxLayout()
        input_dir_layout.addWidget(QLabel("输入目录:"))
        self.input_dir_edit = QLineEdit()
        input_dir_layout.addWidget(self.input_dir_edit)
        self.input_browse_btn = QPushButton("浏览")
        self.input_browse_btn.clicked.connect(self.browse_input_dir)
        input_dir_layout.addWidget(self.input_browse_btn)
        input_layout.addLayout(input_dir_layout)
        
        # 输出目录选择
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("输出目录:"))
        self.output_dir_edit = QLineEdit()
        output_dir_layout.addWidget(self.output_dir_edit)
        self.output_browse_btn = QPushButton("浏览")
        self.output_browse_btn.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_browse_btn)
        input_layout.addLayout(output_dir_layout)
        
        layout.addWidget(input_group)
        
        # 格式和增强设置
        settings_group = QGroupBox("增强设置")
        settings_layout = QGridLayout()
        settings_group.setLayout(settings_layout)
        
        # 格式选择
        settings_layout.addWidget(QLabel("数据格式:"), 0, 0)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["YOLO", "LabelMe", "XML"])
        self.format_combo.currentTextChanged.connect(self.on_format_changed)
        settings_layout.addWidget(self.format_combo, 0, 1)
        
        # 增强倍数
        settings_layout.addWidget(QLabel("增强倍数:"), 1, 0)
        self.multiplier_spinbox = QSpinBox()
        self.multiplier_spinbox.setMinimum(1)
        self.multiplier_spinbox.setMaximum(20)
        self.multiplier_spinbox.setValue(3)
        settings_layout.addWidget(self.multiplier_spinbox, 1, 1)
        
        # 格式说明
        self.format_info_label = QLabel("YOLO格式：支持images/labels目录结构或同目录下的图片和txt文件")
        self.format_info_label.setStyleSheet("QLabel { color: #666; font-size: 11px; }")
        self.format_info_label.setWordWrap(True)
        settings_layout.addWidget(self.format_info_label, 2, 0, 1, 2)
        
        layout.addWidget(settings_group)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始增强")
        self.start_btn.clicked.connect(self.start_augmentation)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }")
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止处理")
        self.stop_btn.clicked.connect(self.stop_augmentation)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }")
        button_layout.addWidget(self.stop_btn)
        
        layout.addLayout(button_layout)
        layout.addStretch()
    
    def on_format_changed(self, format_text):
        """格式选择变化时的处理"""
        if format_text == "LabelMe":
            self.format_info_label.setText("LabelMe格式：图片和JSON标注文件在同一目录下")
        elif format_text == "XML":
            self.format_info_label.setText("XML格式：图片和XML标注文件在同一目录下（LabelImg格式）")
        else:
            self.format_info_label.setText("YOLO格式：支持images/labels目录结构或同目录下的图片和txt文件")
        
    def setup_augmentation_tab(self, tab):
        """设置增强参数选项卡"""
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        # 几何变换
        geo_group = QGroupBox("几何变换")
        geo_layout = QGridLayout()
        geo_group.setLayout(geo_layout)
        
        # 水平翻转
        self.horizontal_flip_cb = QCheckBox("水平翻转")
        geo_layout.addWidget(self.horizontal_flip_cb, 0, 0)
        self.horizontal_flip_prob_slider = QSlider(Qt.Horizontal)
        self.horizontal_flip_prob_slider.setRange(0, 100)
        self.horizontal_flip_prob_slider.setValue(50)
        geo_layout.addWidget(self.horizontal_flip_prob_slider, 0, 1)
        self.horizontal_flip_prob_label = QLabel("50%")
        geo_layout.addWidget(self.horizontal_flip_prob_label, 0, 2)
        self.horizontal_flip_prob_slider.valueChanged.connect(
            lambda v: self.horizontal_flip_prob_label.setText(f"{v}%")
        )
        
        # 垂直翻转
        self.vertical_flip_cb = QCheckBox("垂直翻转")
        geo_layout.addWidget(self.vertical_flip_cb, 1, 0)
        self.vertical_flip_prob_slider = QSlider(Qt.Horizontal)
        self.vertical_flip_prob_slider.setRange(0, 100)
        self.vertical_flip_prob_slider.setValue(30)
        geo_layout.addWidget(self.vertical_flip_prob_slider, 1, 1)
        self.vertical_flip_prob_label = QLabel("30%")
        geo_layout.addWidget(self.vertical_flip_prob_label, 1, 2)
        self.vertical_flip_prob_slider.valueChanged.connect(
            lambda v: self.vertical_flip_prob_label.setText(f"{v}%")
        )
        
        # 旋转
        self.rotation_cb = QCheckBox("旋转")
        geo_layout.addWidget(self.rotation_cb, 2, 0)
        self.rotation_limit_slider = QSlider(Qt.Horizontal)
        self.rotation_limit_slider.setRange(0, 45)
        self.rotation_limit_slider.setValue(15)
        geo_layout.addWidget(self.rotation_limit_slider, 2, 1)
        self.rotation_limit_label = QLabel("15°")
        geo_layout.addWidget(self.rotation_limit_label, 2, 2)
        self.rotation_limit_slider.valueChanged.connect(
            lambda v: self.rotation_limit_label.setText(f"{v}°")
        )
        
        # 缩放
        self.scale_cb = QCheckBox("随机缩放")
        geo_layout.addWidget(self.scale_cb, 3, 0)
        self.scale_limit_slider = QSlider(Qt.Horizontal)
        self.scale_limit_slider.setRange(0, 50)
        self.scale_limit_slider.setValue(20)
        geo_layout.addWidget(self.scale_limit_slider, 3, 1)
        self.scale_limit_label = QLabel("20%")
        geo_layout.addWidget(self.scale_limit_label, 3, 2)
        self.scale_limit_slider.valueChanged.connect(
            lambda v: self.scale_limit_label.setText(f"{v}%")
        )
        
        scroll_layout.addWidget(geo_group)
        
        # 颜色变换
        color_group = QGroupBox("颜色变换")
        color_layout = QGridLayout()
        color_group.setLayout(color_layout)
        
        # 亮度对比度
        self.brightness_contrast_cb = QCheckBox("亮度对比度")
        color_layout.addWidget(self.brightness_contrast_cb, 0, 0)
        self.brightness_limit_slider = QSlider(Qt.Horizontal)
        self.brightness_limit_slider.setRange(0, 50)
        self.brightness_limit_slider.setValue(20)
        color_layout.addWidget(self.brightness_limit_slider, 0, 1)
        self.brightness_limit_label = QLabel("20%")
        color_layout.addWidget(self.brightness_limit_label, 0, 2)
        self.brightness_limit_slider.valueChanged.connect(
            lambda v: self.brightness_limit_label.setText(f"{v}%")
        )
        
        # 色相饱和度
        self.hue_saturation_cb = QCheckBox("色相饱和度")
        color_layout.addWidget(self.hue_saturation_cb, 1, 0)
        self.hue_limit_slider = QSlider(Qt.Horizontal)
        self.hue_limit_slider.setRange(0, 50)
        self.hue_limit_slider.setValue(20)
        color_layout.addWidget(self.hue_limit_slider, 1, 1)
        self.hue_limit_label = QLabel("20")
        color_layout.addWidget(self.hue_limit_label, 1, 2)
        self.hue_limit_slider.valueChanged.connect(
            lambda v: self.hue_limit_label.setText(str(v))
        )
        
        scroll_layout.addWidget(color_group)
        
        # 噪声和模糊
        noise_group = QGroupBox("噪声和模糊")
        noise_layout = QGridLayout()
        noise_group.setLayout(noise_layout)
        
        # 高斯噪声
        self.gaussian_noise_cb = QCheckBox("高斯噪声")
        noise_layout.addWidget(self.gaussian_noise_cb, 0, 0, 1, 3)
        
        # 高斯模糊
        self.gaussian_blur_cb = QCheckBox("高斯模糊")
        noise_layout.addWidget(self.gaussian_blur_cb, 1, 0, 1, 3)
        
        scroll_layout.addWidget(noise_group)
        
        # 高级变换
        advanced_group = QGroupBox("高级变换")
        advanced_layout = QGridLayout()
        advanced_group.setLayout(advanced_layout)
        
        self.elastic_transform_cb = QCheckBox("弹性变换")
        advanced_layout.addWidget(self.elastic_transform_cb, 0, 0)
        
        self.grid_distortion_cb = QCheckBox("网格扭曲")
        advanced_layout.addWidget(self.grid_distortion_cb, 1, 0)
        
        self.perspective_cb = QCheckBox("透视变换")
        advanced_layout.addWidget(self.perspective_cb, 2, 0)
        
        self.clahe_cb = QCheckBox("CLAHE增强")
        advanced_layout.addWidget(self.clahe_cb, 3, 0)
        
        scroll_layout.addWidget(advanced_group)
        
        # 预设按钮
        preset_layout = QHBoxLayout()
        preset_light_btn = QPushButton("轻度增强")
        preset_light_btn.clicked.connect(self.apply_light_preset)
        preset_layout.addWidget(preset_light_btn)
        
        preset_medium_btn = QPushButton("中度增强")
        preset_medium_btn.clicked.connect(self.apply_medium_preset)
        preset_layout.addWidget(preset_medium_btn)
        
        preset_heavy_btn = QPushButton("重度增强")
        preset_heavy_btn.clicked.connect(self.apply_heavy_preset)
        preset_layout.addWidget(preset_heavy_btn)
        
        scroll_layout.addLayout(preset_layout)
        
    def setup_progress_tab(self, tab):
        """设置进度选项卡"""
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # 进度条
        progress_group = QGroupBox("处理进度")
        progress_layout = QVBoxLayout()
        progress_group.setLayout(progress_layout)
        
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("就绪")
        progress_layout.addWidget(self.status_label)
        
        # 添加统计信息标签
        self.stats_label = QLabel("统计信息将在处理时显示")
        self.stats_label.setStyleSheet("QLabel { color: #666; font-size: 12px; }")
        progress_layout.addWidget(self.stats_label)
        
        layout.addWidget(progress_group)
        
        # 日志区域
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
    def browse_input_dir(self):
        """浏览输入目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if dir_path:
            self.input_dir_edit.setText(dir_path)
            
    def browse_output_dir(self):
        """浏览输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def apply_light_preset(self):
        """应用轻度增强预设"""
        self.horizontal_flip_cb.setChecked(True)
        self.brightness_contrast_cb.setChecked(True)
        self.rotation_cb.setChecked(True)
        self.rotation_limit_slider.setValue(10)
        
    def apply_medium_preset(self):
        """应用中度增强预设"""
        self.horizontal_flip_cb.setChecked(True)
        self.vertical_flip_cb.setChecked(True)
        self.brightness_contrast_cb.setChecked(True)
        self.hue_saturation_cb.setChecked(True)
        self.rotation_cb.setChecked(True)
        self.scale_cb.setChecked(True)
        self.gaussian_noise_cb.setChecked(True)
        
    def apply_heavy_preset(self):
        """应用重度增强预设"""
        self.horizontal_flip_cb.setChecked(True)
        self.vertical_flip_cb.setChecked(True)
        self.brightness_contrast_cb.setChecked(True)
        self.hue_saturation_cb.setChecked(True)
        self.rotation_cb.setChecked(True)
        self.scale_cb.setChecked(True)
        self.gaussian_noise_cb.setChecked(True)
        self.gaussian_blur_cb.setChecked(True)
        self.elastic_transform_cb.setChecked(True)
        self.grid_distortion_cb.setChecked(True)
        self.perspective_cb.setChecked(True)
        self.clahe_cb.setChecked(True)
    
    def get_augmentation_params(self):
        """获取增强参数"""
        params = {}
        
        # 几何变换
        params['horizontal_flip'] = self.horizontal_flip_cb.isChecked()
        params['horizontal_flip_prob'] = self.horizontal_flip_prob_slider.value() / 100.0
        
        params['vertical_flip'] = self.vertical_flip_cb.isChecked()
        params['vertical_flip_prob'] = self.vertical_flip_prob_slider.value() / 100.0
        
        params['rotation'] = self.rotation_cb.isChecked()
        params['rotation_limit'] = self.rotation_limit_slider.value()
        params['rotation_prob'] = 0.5
        
        params['scale'] = self.scale_cb.isChecked()
        params['scale_limit'] = self.scale_limit_slider.value() / 100.0
        params['scale_prob'] = 0.5
        
        # 颜色变换
        params['brightness_contrast'] = self.brightness_contrast_cb.isChecked()
        params['brightness_limit'] = self.brightness_limit_slider.value() / 100.0
        params['contrast_limit'] = self.brightness_limit_slider.value() / 100.0
        params['brightness_contrast_prob'] = 0.5
        
        params['hue_saturation'] = self.hue_saturation_cb.isChecked()
        params['hue_limit'] = self.hue_limit_slider.value()
        params['saturation_limit'] = 30
        params['value_limit'] = 20
        params['hue_saturation_prob'] = 0.5
        
        # 噪声和模糊
        params['gaussian_noise'] = self.gaussian_noise_cb.isChecked()
        params['gaussian_noise_prob'] = 0.3
        
        params['gaussian_blur'] = self.gaussian_blur_cb.isChecked()
        params['gaussian_blur_prob'] = 0.3
        
        # 高级变换
        params['elastic_transform'] = self.elastic_transform_cb.isChecked()
        params['elastic_transform_prob'] = 0.3
        
        params['grid_distortion'] = self.grid_distortion_cb.isChecked()
        params['grid_distortion_prob'] = 0.3
        
        params['perspective'] = self.perspective_cb.isChecked()
        params['perspective_prob'] = 0.3
        
        params['clahe'] = self.clahe_cb.isChecked()
        params['clahe_prob'] = 0.3
        
        return params
    
    def start_augmentation(self):
        """开始数据增强"""
        # 验证输入
        input_dir = self.input_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        
        if not input_dir or not os.path.exists(input_dir):
            QMessageBox.warning(self, "警告", "请选择有效的输入目录")
            return
        
        if not output_dir:
            QMessageBox.warning(self, "警告", "请选择输出目录")
            return
        
        # 获取参数
        aug_params = self.get_augmentation_params()
        multiplier = self.multiplier_spinbox.value()
        format_type = self.format_combo.currentText().lower()
        
        # 检查是否选择了至少一种增强方法
        if not any(aug_params.values()):
            QMessageBox.warning(self, "警告", "请至少选择一种数据增强方法")
            return
        
        # 禁用开始按钮，启用停止按钮
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # 清空日志
        self.log_text.clear()
        self.progress_bar.setValue(0)
        
        # 创建并启动工作线程
        self.worker = AugmentationWorker(input_dir, output_dir, aug_params, multiplier, format_type)
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.status_updated.connect(self.add_log)
        self.worker.stats_updated.connect(self.stats_label.setText)  # 连接统计更新
        self.worker.finished.connect(self.on_augmentation_finished)
        self.worker.error_occurred.connect(self.on_augmentation_error)
        self.worker.start()
    
    def stop_augmentation(self):
        """停止数据增强"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("已停止")
        self.stats_label.setText("处理已停止")
        self.add_log("处理已被用户停止")
    
    def on_augmentation_finished(self, message):
        """增强完成处理"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("完成")
        self.stats_label.setText("处理完成！查看详细信息请点击确定")
        self.add_log(message)
        
        # 创建详细的完成对话框
        msg_box = QMessageBox()
        msg_box.setWindowTitle("处理完成")
        msg_box.setText("数据增强处理已完成！")
        msg_box.setDetailedText(message)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()
    
    def on_augmentation_error(self, error_message):
        """增强错误处理"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("错误")
        self.add_log(f"错误: {error_message}")
        QMessageBox.critical(self, "错误", error_message)
    
    def add_log(self, message):
        """添加日志"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("数据增强工具")
    
    # 设置应用程序图标和样式
    app.setStyle('Fusion')
    
    window = DataAugmenterGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main() 