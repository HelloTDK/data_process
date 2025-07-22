"""
切分双层车牌为两张图片，并把文件名切分
如桂AF797挂_0000000000025871_0.jpg -> 桂A_0000000000025871_0.jpg 和 F797挂_0000000000025871_1.jpg

"""
import cv2
import numpy as np
import os


def cv_imread(file_path):
    """
    解决cv2.imread()不能读取中文路径的问题
    """
    # 使用np.fromfile()读取文件为bytes，然后用cv2.imdecode()解码
    img_array = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


def cv_imwrite(file_path, img):
    """
    解决cv2.imwrite()不能写入中文路径的问题
    """
    # 获取文件扩展名
    ext = os.path.splitext(file_path)[1]
    # 使用cv2.imencode()编码图像，然后用tofile()保存到文件
    result, img_encode = cv2.imencode(ext, img)
    if result:
        img_encode.tofile(file_path)
        return True
    return False


def get_split_merge(img):
    h,w,c = img.shape
    img_upper = img[0:int(5/12*h),:]
    img_lower = img[int(1/3*h):,:]
    img_upper = cv2.resize(img_upper,(img_lower.shape[1],img_lower.shape[0]))
    # new_img = np.hstack((img_upper,img_lower))
    return img_upper,img_lower

def split_double_plate(img_path,output_dir):
    img_name = os.path.basename(img_path)
    plate_name = img_name.split('_')[0]
    plate_suffix = img_name.split('_')[1:]
    img = cv_imread(img_path)  # 使用自定义的读取函数
    if img is None:
        print(f"无法读取图像: {img_path}")
        return
    
    img_upper,img_lower = get_split_merge(img)
    # 使用自定义的写入函数
    upper_path = os.path.join(output_dir,plate_name[:2]+f'_{"_".join(plate_suffix)}')
    lower_path = os.path.join(output_dir,plate_name[2:]+f'_{"_".join(plate_suffix)}')
    
    cv_imwrite(upper_path, img_upper)
    cv_imwrite(lower_path, img_lower)

def batch_split_double_plate(img_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for img_name in os.listdir(img_dir):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(img_dir,img_name)
            split_double_plate(img_path,output_dir)

if __name__ == '__main__':
    img_dir = r'D:\Data\car_plate\plate_recong\live\ID1928278804434837504\ID1928278804434837504\outputplate_rec_color\roi (2)\roi\double'
    output_dir = r'D:\Data\car_plate\plate_recong\live\ID1928278804434837504\ID1928278804434837504\outputplate_rec_color\roi (2)\roi\double_output'
    batch_split_double_plate(img_dir,output_dir)




