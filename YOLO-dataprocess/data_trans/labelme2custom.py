import os
import shutil
import json

"""
    功能描述：将labelme格式转换为自定义数据格式
    输入：
        labelme_data: labelme数据
    输出：
        custom_data: 自定义数据
"""

def transform_labelme_to_custom(labelme_data):

    results = []
    for shape in labelme_data.get("shapes",[]):
        custom_data = {
            "name": shape["label"],
            "polygon_points":shape["points"],
            "bnd_points":[],
            "key_point":[]
        }
        results.append(custom_data)
    transform_data = {
        "labelList":results,
        "itemType": 0,
        "background": False,
        "markType": 0,
        "checked": False,
        "addTrain": 0
        }
    return transform_data


def batch_transform(input_dir,output_dir):

    os.makedirs(output_dir,exist_ok=True)
    for filename in os.listdir(input_dir):
        try:
            label_path = os.path.join(input_dir,filename)
            # 读取json数据
            with open(label_path ,'r',encoding='utf-8') as file:
                labelme_data = json.load(file)
            custom_data = transform_labelme_to_custom(labelme_data)
            transform_path = os.path.join(output_dir,filename)
            with open(transform_path,'w',encoding='utf-8') as file:
                json.dump(custom_data,file)
        except Exception as e:
            # 如果发生异常，打印错误并跳过该文件
            print(f"Failed to process {filename}: {e}")
            continue


if __name__ == "__main__":
    input_dir = r'D:\Data\fire_and_smoke\dataset5.4-seg\label'
    output_dir = r'D:\Data\fire_and_smoke\dataset5.4-seg\labels'
    batch_transform(input_dir,output_dir)