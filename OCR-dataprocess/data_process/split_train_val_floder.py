import os
import shutil
import random
import glob

def split_train_val_floder(img_dir, output_dir, ratio=0.9):
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
    image_files = glob.glob(os.path.join(img_dir, '**', '*.*'), recursive=True)
    image_files = [file for file in image_files if file.split('.')[-1] in image_extensions]

    random.shuffle(image_files)

    train_files = image_files[:int(len(image_files) * ratio)]
    val_files = image_files[int(len(image_files) * ratio):]

    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for file in train_files:
        shutil.copy(file, os.path.join(train_dir, os.path.basename(file)))

    for file in val_files:
        shutil.copy(file, os.path.join(val_dir, os.path.basename(file)))

    print(f"训练集和验证集已保存到: {output_dir}")  

if __name__ == '__main__':
    img_dir = r'D:\Data\car_plate\plate_recong\git_plate\merge7.11'
    output_dir = os.path.dirname(img_dir)
    output_dir = os.path.join(output_dir, 'train_val_split')
    split_train_val_floder(img_dir, output_dir)

    
