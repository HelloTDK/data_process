import os
import argparse
from pathlib import Path
from tqdm import tqdm

class YOLOClassCombiner:
    def __init__(self, label_dir, output_dir, class_mapping):
        """
        初始化YOLO类别合并器
        
        Args:
            label_dir (str): YOLO标签文件所在目录
            output_dir (str): 合并后的标签文件输出目录
            class_mapping (dict): 类别映射字典，例如 {1: 0, 2: 0} 表示将类别1和2都合并为类别0
        """
        self.label_dir = Path(label_dir)
        self.output_dir = Path(output_dir)
        self.class_mapping = class_mapping
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
    
    def process_single_file(self, file_path):
        """处理单个标签文件"""
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        processed_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:  # 跳过空行
                continue
                
            class_id = int(parts[0])
            # 如果类别在映射字典中，则替换为新的类别
            if class_id in self.class_mapping:
                parts[0] = str(self.class_mapping[class_id])
            
            else:
                continue

            
            processed_lines.append(' '.join(parts) + '\n')
        
        return processed_lines
    
    def process_all_files(self):
        """处理目录下的所有.txt文件"""
        label_files = list(self.label_dir.glob('*.txt'))
        print(f"找到 {len(label_files)} 个标签文件")
        
        for file_path in tqdm(label_files, desc="处理标签文件"):
            # 处理文件
            processed_lines = self.process_single_file(file_path)
            
            # 保存处理后的文件
            output_path = self.output_dir / file_path.name
            with open(output_path, 'w') as f:
                f.writelines(processed_lines)

def main():

    class_map = {0:0,1:0,3:1,4:1,5:1,8:1}
    # args = parser.parse_args()
    label_dir = r"D:\Data\UAV\SkyToSurface\VisDrone2019\VisDrone2019-DET-val\yolo_labels\labels"
    output_dir = os.path.join(os.path.dirname(label_dir),"yolo_combine")
    print(f"output_dir: {output_dir}")
    

    
    # 创建并运行处理器
    processor = YOLOClassCombiner(label_dir, output_dir, class_map)
    processor.process_all_files()
    
    print("处理完成！")

if __name__ == '__main__':
    main()
