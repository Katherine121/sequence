import PIL
import math
import random
import shutil

import numpy as np
import os

from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True


# 1. 删除所有的None图像
# 2. 手动删除开头和结尾多余的图像
# 3. 对于序列预测任务，删了多余的几张照片75986：15，83832：3，97128：全部
# 到这步U盘里有
# 4. 聚类
# 5. 打不开的图像用同类图像的增强版本进行替换
def alter_pics(k):
    res = []

    path1 = "processOrder/" + str(k) + "/all_class"
    a = 0

    for i in range(0, k):
        path2 = os.path.join(path1, str(i))

        files_for_one_class = os.listdir(path2)
        files_for_one_class.sort()

        for j in range(0, len(files_for_one_class)):
            full_file_path = os.path.join(path2, files_for_one_class[j])

            try:
                pic = Image.open(full_file_path)
                pic = pic.convert('RGB')
            except(OSError, PIL.UnidentifiedImageError):
                a += 1
                # 使用同类图像去代替，+5，-5肯定同在一个训练集/测试集
                if j - 5 >= 0:
                    alter_full_file_path = os.path.join(path2, files_for_one_class[j - 5])
                    alter_pic = Image.open(alter_full_file_path)
                    # # 放大
                    # alter_pic = alter_pic.resize((360, 220))
                    # # 随机裁剪
                    # rand_left = random.randint(0, 40)
                    # rand_top = random.randint(0, 40)
                    # alter_pic = alter_pic.crop((rand_left, rand_top, rand_left+320, rand_top+180))
                    # 保存到原来的位置
                    alter_pic.save(full_file_path)
                elif j + 5 < len(files_for_one_class):
                    alter_full_file_path = os.path.join(path2, files_for_one_class[j + 5])
                    alter_pic = Image.open(alter_full_file_path)
                    # # 放大
                    # alter_pic = alter_pic.resize((360, 220))
                    # # 随机裁剪
                    # rand_left = random.randint(0, 40)
                    # rand_top = random.randint(0, 40)
                    # alter_pic = alter_pic.crop((rand_left, rand_top, rand_left + 320, rand_top + 180))
                    # 保存到原来的位置
                    alter_pic.save(full_file_path)

            res.append((full_file_path, i))
    print(a)
    print(len(res))


# 6. 原来路径中打不开的图像用替换后的图像进行覆盖
def copy_alter_pics(k):
    res = []

    path1 = "processOrder/order"
    path2 = "processOrder/" + str(k) + "/all_class"

    dir1 = os.listdir(path1)
    dir1.sort()

    a = 0

    for dir in dir1:
        full_dir_path = os.path.join(path1, dir)

        files = os.listdir(full_dir_path)
        files.sort()

        for file in files:
            full_file_path = os.path.join(full_dir_path, file)

            try:
                pic = Image.open(full_file_path)
                pic = pic.convert('RGB')
            except(OSError):
                a += 1
                # 如果某条路径有打不开的图像，那么使用刚才生成的增强版本图像替换
                for i in range(0, k):
                    full_dir_path2 = os.path.join(path2, str(i))
                    full_file_path2 = os.path.join(full_dir_path2, file)

                    if os.path.exists(full_file_path2):
                        alter_pic = Image.open(full_file_path2)
                        alter_pic.save(full_file_path)
                        break
            res.append(full_file_path)
    print(a)
    print(len(res))


# if __name__ == '__main__':
#     alter_pics(k=100)
#     copy_alter_pics(k=100)