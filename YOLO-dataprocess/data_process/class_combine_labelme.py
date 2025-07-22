import os
import json
from pathlib import Path
from tqdm import tqdm

class LabelmeClassCombiner:
    def __init__(self, json_dir, output_dir, class_mapping):
        """
        初始化Labelme类别合并器
        
        Args:
            json_dir (str): Labelme JSON文件所在目录
            output_dir (str): 合并后的JSON文件输出目录
            class_mapping (dict): 类别映射字典，例如 {"car": "vehicle", "bus": "vehicle"} 
                                表示将car和bus类别都合并为vehicle
        """
        self.json_dir = Path(json_dir)
        self.output_dir = Path(output_dir)
        self.class_mapping = class_mapping
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_single_file(self, file_path):
        """处理单个JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理shapes中的标注
        processed_shapes = []
        for shape in data['shapes']:
            label = shape['label']
            
            # 如果类别在映射字典中，则替换为新的类别
            if label in self.class_mapping:
                shape['label'] = self.class_mapping[label]
                processed_shapes.append(shape)
            else:
                continue  # 跳过不在映射中的类别
        
        # 更新shapes
        data['shapes'] = processed_shapes
        
        return data
    
    def process_all_files(self):
        """处理目录下的所有JSON文件"""
        json_files = list(self.json_dir.glob('*.json'))
        print(f"找到 {len(json_files)} 个JSON文件")
        
        for file_path in tqdm(json_files, desc="处理标注文件"):
            try:
                # 处理文件
                processed_data = self.process_single_file(file_path)
                
                # 保存处理后的文件
                output_path = self.output_dir / file_path.name
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
                continue

def main():
    # 配置参数
    class_map = {
        "car": "vehicle",
        "bus": "vehicle",
        "truck": "vehicle",
        "person": "person",
        "bicycle": "bicycle"
    }
    
    json_dir = r"path/to/your/labelme/jsons"  # 请修改为您的JSON文件目录
    output_dir = os.path.join(os.path.dirname(json_dir), "labelme_combine")
    print(f"输出目录: {output_dir}")
    
    # 创建并运行处理器
    processor = LabelmeClassCombiner(json_dir, output_dir, class_map)
    processor.process_all_files()
    
    print("处理完成！")

if __name__ == '__main__':
    main()
