import os
import cv2
import random


def split_trainval(img_dir, output_dir, ratio=0.8):
    train_set = []
    val_set = []
    all_img_names = os.listdir(img_dir)
    random.shuffle(all_img_names)
    train_num = int(len(all_img_names) * ratio)
    train_set = all_img_names[:train_num]
    val_set = all_img_names[train_num:]
    train_txt_path = os.path.join(output_dir, 'train.txt')
    val_txt_path = os.path.join(output_dir, 'val.txt')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    def write_set(img_set, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for img_name in img_set:
                try:
                    suffix = img_name.split('.')[-1].lower()
                    if suffix in ['jpg', 'png', 'jpeg']:
                        img_path = os.path.join(img_dir, img_name)
                        label = img_name.split('_')[0]
                        
                        # 调试信息：打印处理的文件名和标签
                        print(f"处理文件: {img_name}")
                        print(f"提取标签: {label}")
                        print(f"标签长度: {len(label)} 字符")
                        
                        # 写入文件
                        line = f"images/{img_name}\t{label}\n"
                        f.write(line)
                        
                except Exception as e:
                    print(f"处理文件 {img_name} 时出错: {e}")
                    continue
    
    print(f"开始处理训练集，共 {len(train_set)} 个文件")
    write_set(train_set, train_txt_path)
    print(f"训练集文件已保存到: {train_txt_path}")
    
    print(f"开始处理验证集，共 {len(val_set)} 个文件")
    write_set(val_set, val_txt_path)
    print(f"验证集文件已保存到: {val_txt_path}")


if __name__ == '__main__':
    img_dir = r'/expdata/givap/dataset/train_data/plate_ocr5/images'
    output_dir = os.path.dirname(img_dir)
    print(f"输出目录: {output_dir}")
    split_trainval(img_dir, output_dir)







