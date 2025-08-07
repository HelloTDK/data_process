#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检出数据集标签脚本
比如_1375.jpg，_1375前边是空的应该直接删除_1375小于2的都直接删除
"""

import os 

def delete_problem_label(input_dir):
    img_names = os.listdir(input_dir)
    for img_name in img_names:
        label = img_name.split('_')[0]
        if len(label) <= 2:
            print(f"{img_name} removed")
            rm_path = os.path.join(input_dir,img_name)
            os.remove(rm_path)
if __name__ == "__main__":
    input_dir = "/expdata/givap/data/plate_recong/merge/b1"
    delete_problem_label(input_dir)

