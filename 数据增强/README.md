# 多格式数据增强工具

支持YOLO、LabelImg XML、LabelMe JSON三种标注格式的数据增强工具。

## 功能特点

- **多格式支持**: 支持YOLO (.txt)、LabelImg XML (.xml)、LabelMe JSON (.json) 三种标注格式
- **自动格式检测**: 可以自动检测输入数据的标注格式
- **丰富的增强方法**: 包括几何变换、颜色变换、噪声模糊、高级变换等
- **批量处理**: 支持批量处理大量图片和标注文件
- **中文路径支持**: 完全支持中文文件名和路径
- **实时进度显示**: 提供详细的处理进度和统计信息

## 支持的标注格式

### 1. YOLO格式 (.txt)
```
class_id x_center y_center width height
```
- 坐标为归一化坐标 (0-1)
- 每行一个对象

### 2. LabelImg XML格式 (.xml)
```xml
<annotation>
    <object>
        <name>class_name</name>
        <bndbox>
            <xmin>x1</xmin>
            <ymin>y1</ymin>
            <xmax>x2</xmax>
            <ymax>y2</ymax>
        </bndbox>
    </object>
</annotation>
```
- Pascal VOC格式
- 坐标为像素坐标

### 3. LabelMe JSON格式 (.json)
```json
{
    "shapes": [
        {
            "label": "class_name",
            "points": [[x1, y1], [x2, y2]],
            "shape_type": "rectangle"
        }
    ]
}
```
- LabelMe工具生成的格式
- 坐标为像素坐标

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 启动GUI界面
```bash
python run.py
```

### 2. 目录结构

支持以下两种目录结构：

#### 结构1: 分离式目录
```
input_dir/
├── images/          # 图片文件
│   ├── img1.jpg
│   └── img2.jpg
└── labels/          # YOLO标注文件
    ├── img1.txt
    └── img2.txt
```

或者

```
input_dir/
├── images/          # 图片文件
│   ├── img1.jpg
│   └── img2.jpg
└── annotations/     # XML/JSON标注文件
    ├── img1.xml
    └── img2.json
```

#### 结构2: 混合式目录
```
input_dir/
├── img1.jpg
├── img1.txt        # 标注文件与图片在同一目录
├── img2.jpg
└── img2.xml
```

### 3. 操作步骤

1. **选择输入目录**: 包含图片和标注文件的目录
2. **选择输出目录**: 增强后文件的保存位置
3. **选择标注格式**: 
   - 自动检测（推荐）
   - 手动指定格式
4. **设置增强参数**: 根据需要选择增强方法和强度
5. **设置增强倍数**: 每张原图生成多少张增强图片
6. **开始处理**: 点击"开始增强"按钮

### 4. 输出结果

增强后的文件将保存在输出目录中：

```
output_dir/
├── images/              # 增强后的图片
│   ├── img1_aug_1.jpg
│   ├── img1_aug_2.jpg
│   └── ...
└── labels/              # YOLO格式标注
    ├── img1_aug_1.txt
    ├── img1_aug_2.txt
    └── ...
```

或者

```
output_dir/
├── images/              # 增强后的图片
│   ├── img1_aug_1.jpg
│   ├── img1_aug_2.jpg
│   └── ...
└── annotations/         # XML/JSON格式标注
    ├── img1_aug_1.xml
    ├── img1_aug_2.json
    └── ...
```

## 增强方法

### 几何变换
- 水平翻转
- 垂直翻转
- 旋转
- 随机缩放

### 颜色变换
- 亮度对比度调整
- 色相饱和度调整

### 噪声和模糊
- 高斯噪声
- 高斯模糊

### 高级变换
- 弹性变换
- 网格扭曲
- 透视变换
- CLAHE增强

## 预设配置

- **轻度增强**: 适合对数据质量要求较高的场景
- **中度增强**: 平衡增强效果和数据质量
- **重度增强**: 最大化数据多样性

## 测试功能

运行测试脚本验证格式支持：

```bash
python test_formats.py
```

## 注意事项

1. **备份原始数据**: 建议在处理前备份原始数据
2. **检查输出结果**: 处理完成后检查输出文件的正确性
3. **格式一致性**: 确保同一批数据使用相同的标注格式
4. **内存使用**: 处理大量图片时注意内存使用情况
5. **路径长度**: 避免使用过长的文件路径

## 常见问题

### Q: 如何处理混合格式的数据？
A: 建议将不同格式的数据分别放在不同目录中处理。

### Q: 增强后的标注坐标是否准确？
A: 工具会自动调整标注坐标以匹配增强后的图片，但建议抽查验证。

### Q: 支持哪些图片格式？
A: 支持 .jpg, .jpeg, .png, .bmp, .tiff 格式。

### Q: 如何处理没有标注的图片？
A: 工具会自动处理没有对应标注文件的图片，只进行图片增强。

## 更新日志

### v2.0
- 新增LabelImg XML格式支持
- 新增LabelMe JSON格式支持
- 新增自动格式检测功能
- 优化用户界面
- 改进错误处理机制

### v1.0
- 基础YOLO格式支持
- 基本数据增强功能