import os

import math
import torch
from PIL import Image
from torch.utils.data import Dataset


def write_img_angle(k):
    path1 = "processOrder/order"
    path2 = "processOrder/" + str(k) + "/all_class"
    res = []

    dir_path1 = os.listdir(path1)
    dir_path1.sort()

    for i in range(0, len(dir_path1)):
        dir1 = dir_path1[i]
        full_dir_path1 = os.path.join(path1, dir1)

        file_path1 = os.listdir(full_dir_path1)
        file_path1.sort()

        path = []

        for file1 in file_path1:
            full_file_path1 = os.path.join(full_dir_path1, file1)

            # # **************************************************
            # # 只需要跑一次
            # if i % 2 != 0:
            #     # 需要顺时针180°旋转图像
            #     pic = Image.open(full_file_path1)
            #     pic = pic.rotate(angle=180)
            #     pic.save(full_file_path1)
            # # **************************************************

            # 纬度
            lat_index = file1.find("lat")
            # 高度
            alt_index = file1.find("alt")
            # 经度
            lon_index = file1.find("lon")

            start = file1[0: lat_index - 1]
            lat_pos = file1[lat_index + 4: alt_index - 1]
            alt_pos = file1[alt_index + 4: lon_index - 1]
            lon_pos = file1[lon_index + 4: -4]

            for j in range(0, k):
                full_dir_path2 = os.path.join(path2, str(j))
                full_file_path2 = os.path.join(full_dir_path2, file1)
                if os.path.exists(full_file_path2):
                    # 当前图像，当前标签，当前纬度，当前经度
                    path.append((full_file_path1, j, eval(lat_pos), eval(lon_pos)))
                    break

        # 正反两种路径
        if i % 2 != 0:
            path.reverse()
        res.append(path)

    res_delta = []
    for path in res:
        path_delta = []

        for i in range(0, len(path)):
            flag = False
            for j in range(i + 1, len(path)):
                # 如果找到了下一个里程碑
                if path[j][1] != path[i][1]:
                    # 计算里程碑位置距离当前位置的距离
                    lat_delta = (path[j][2] - path[i][2]) * 111000
                    lon_delta = (path[j][3] - path[i][3]) * 111000 * math.cos(path[i][2] / 180 * math.pi)

                    sum = math.sqrt(lat_delta * lat_delta + lon_delta * lon_delta)
                    sin = lat_delta / sum
                    cos = lon_delta / sum
                    # 里程碑编号，飞到里程碑时纬度偏转多少，经度偏转多少
                    stone_part = (path[j][1], sin, cos)

                    # 还要计算下一时刻的偏转方向
                    # 计算下一时刻位置距离当前位置的角度
                    lat_delta = (path[i + 1][2] - path[i][2]) * 111000
                    lon_delta = (path[i + 1][3] - path[i][3]) * 111000 * math.cos(path[i][2] / 180 * math.pi)

                    sum = math.sqrt(lat_delta * lat_delta + lon_delta * lon_delta)
                    sin = lat_delta / sum
                    cos = lon_delta / sum
                    # 飞到下一个时刻时纬度偏转多少，经度偏转多少
                    next_part = (sin, cos)

                    # 找到了下一个里程碑就退出
                    flag = True
                    break

            # 如果是终点
            if flag is False:
                break

            # 终点图像
            dest_part = path[-1][0]

            path_delta.append((path[i][0], next_part, dest_part, path[i][1], stone_part))

        res_delta.append(path_delta)

    if os.path.exists("datasets") is False:
        os.mkdir("datasets")
    for path in res_delta:
        for pic in path:
            name, next_part, dest_part, label_part, stone_part = pic[0], pic[1], pic[2], pic[3], pic[4]
            new_txt_name = name[len("processOrder/order/"):len("processOrder/order/") + 5]
            with open("datasets/" + new_txt_name + ".txt", "a") as file1:
                file1.write(name + " " + str(next_part[0]) + " " + str(next_part[1]) + " " +
                            dest_part + " " +
                            str(label_part) + " " +
                            str(stone_part[0]) + " " + str(stone_part[1]) + " " + str(stone_part[2]) + "\n"
                            )
            file1.close()


if __name__ == '__main__':
    write_img_angle(k=100)


class OrderTrainDataset(Dataset):
    def __init__(self, transform, input_len):
        self.transform = transform
        self.input_len = input_len

        res_seq = []
        files = os.listdir("datasets")
        files.sort()
        for i in range(0, len(files)):
            if i % 5 != 0:
                path_seq = []
                file = files[i]
                full_file_path = os.path.join("datasets", file)
                f = open(full_file_path, 'rt')
                for line in f:
                    line = line.strip('\n')
                    line = line.split(' ')

                    line = [line[0], [float(line[1]), float(line[2])],
                            line[3],
                            int(line[4]),
                            int(line[5]), [float(line[6]), float(line[7])]
                            ]
                    path_seq.append(line)
                f.close()

                for j in range(1, len(path_seq) + 1):
                    if j - self.input_len < 0:
                        res_seq.append(path_seq[0: j])
                    else:
                        res_seq.append(path_seq[j - self.input_len: j])

        print(len(res_seq))
        self.imgs = res_seq

    # 返回数据集大小
    def __len__(self):
        return len(self.imgs)

    # 打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        # len, 5
        item = self.imgs[index]

        next_imgs = None
        next_angles = []

        # 本来就有的图像和角度
        for i in range(0, len(item)):
            img = item[i][0]
            img = Image.open(img)
            img = img.convert('RGB')
            img = self.transform(img).unsqueeze(dim=0)
            if next_imgs is None:
                next_imgs = img
            else:
                next_imgs = torch.cat((next_imgs, img), dim=0)

            if i == len(item) - 1:
                next_angles.append([0, 0])
            else:
                next_angles.append(item[i][1])

        # 第五张输入图像对应的终点图像
        dest_img = Image.open(item[-1][2])
        dest_img = dest_img.convert('RGB')
        dest_img = self.transform(dest_img).unsqueeze(dim=0)
        next_imgs = torch.cat((next_imgs, dest_img), dim=0)
        # 第五张输入图像对应的终点偏转方向
        dest_angle = [0, 0]
        next_angles.append(dest_angle)
        next_angles = torch.tensor(next_angles, dtype=torch.float)
        # 当前里程碑编号、预测的里程碑编号和角度
        label1 = item[-1][3]
        label2 = item[-1][4]
        label3 = torch.tensor(item[-1][5], dtype=torch.float)

        # 补上缺的图像和角度
        for i in range(0, self.input_len - len(item)):
            next_imgs = torch.cat((next_imgs, torch.zeros((1, 3, 224, 224))), dim=0)
            next_angles = torch.cat((next_angles, torch.zeros((1, 2))), dim=0)

        # 输入六张图像，六对偏转方向，输出两个里程碑编号，一对里程碑偏转方向
        return next_imgs, next_angles, label1, label2, label3


class OrderTestDataset(Dataset):
    def __init__(self, transform, input_len):
        self.transform = transform
        self.input_len = input_len

        res_seq = []
        files = os.listdir("datasets")
        files.sort()
        for i in range(0, len(files)):
            if i % 5 == 0:
                path_seq = []
                file = files[i]
                full_file_path = os.path.join("datasets", file)
                f = open(full_file_path, 'rt')
                for line in f:
                    line = line.strip('\n')
                    line = line.split(' ')

                    line = [line[0], [float(line[1]), float(line[2])],
                            line[3],
                            int(line[4]),
                            int(line[5]), [float(line[6]), float(line[7])]
                            ]
                    path_seq.append(line)
                f.close()

                for j in range(1, len(path_seq) + 1):
                    if j - self.input_len < 0:
                        res_seq.append(path_seq[0: j])
                    else:
                        res_seq.append(path_seq[j - self.input_len: j])

        print(len(res_seq))
        self.imgs = res_seq

    # 返回数据集大小
    def __len__(self):
        return len(self.imgs)

    # 打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        # len, 5
        item = self.imgs[index]

        next_imgs = None
        next_angles = []

        # 本来就有的图像和角度
        for i in range(0, len(item)):
            img = item[i][0]
            img = Image.open(img)
            img = img.convert('RGB')
            img = self.transform(img).unsqueeze(dim=0)
            if next_imgs is None:
                next_imgs = img
            else:
                next_imgs = torch.cat((next_imgs, img), dim=0)

            if i == len(item) - 1:
                next_angles.append([0, 0])
            else:
                next_angles.append(item[i][1])

        # 第五张输入图像对应的终点图像
        dest_img = Image.open(item[-1][2])
        dest_img = dest_img.convert('RGB')
        dest_img = self.transform(dest_img).unsqueeze(dim=0)
        next_imgs = torch.cat((next_imgs, dest_img), dim=0)
        # 第五张输入图像对应的终点偏转方向
        dest_angle = [0, 0]
        next_angles.append(dest_angle)
        next_angles = torch.tensor(next_angles, dtype=torch.float)
        # 当前里程碑编号、预测的里程碑编号和角度
        label1 = item[-1][3]
        label2 = item[-1][4]
        label3 = torch.tensor(item[-1][5], dtype=torch.float)

        # 补上缺的图像和角度
        for i in range(0, self.input_len - len(item)):
            next_imgs = torch.cat((next_imgs, torch.zeros((1, 3, 224, 224))), dim=0)
            next_angles = torch.cat((next_angles, torch.zeros((1, 2))), dim=0)

        # 输入六张图像，六对偏转方向，输出两个里程碑编号，一对里程碑偏转方向
        return next_imgs, next_angles, label1, label2, label3
