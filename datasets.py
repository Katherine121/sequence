import os

import math
import torch
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F


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
                    path.append((full_file_path1, j, eval(lat_pos), eval(lon_pos)))
                    break

        res.append(path)

    # 需要选择下一个里程碑作为标签
    a = 10000
    b = -10000
    c = 10000
    d = -10000
    res_delta = []
    for path in res:
        path_delta = []
        for i in range(0, len(path)):
            flag = False
            for j in range(i + 1, len(path)):
                # 如果找到了下一个里程碑
                if path[j][1] != path[i][1]:
                    # 如果不是起点
                    if i > 0:
                        # 计算里程碑位置距离当前位置的角度
                        lat_delta2 = path[j][2] - path[i][2]
                        lon_delta2 = path[j][3] - path[i][3]
                        tan2 = float(lat_delta2 / (lon_delta2 + 1e-10))
                        # -90~90°
                        angel2 = math.atan(tan2) / math.pi * 180

                        # x正轴：-180~180°
                        if lat_delta2 >= 0 and lon_delta2 >= 0:
                            angel2 = angel2
                        elif lat_delta2 >= 0 and lon_delta2 <= 0:
                            angel2 += 180
                        elif lat_delta2 <= 0 and lon_delta2 >= 0:
                            angel2 = angel2
                        elif lat_delta2 <= 0 and lon_delta2 <= 0:
                            angel2 -= 180

                        # 计算当前位置距离上一时刻位置的角度
                        lat_delta1 = path[i][2] - path[i - 1][2]
                        lon_delta1 = path[i][3] - path[i - 1][3]
                        tan1 = float(lat_delta1 / (lon_delta1 + 1e-10))
                        # -90~90°
                        angel1 = math.atan(tan1) / math.pi * 180

                        # x正轴：-180~180°
                        if lat_delta1 >= 0 and lon_delta1 >= 0:
                            angel1 = angel1
                        elif lat_delta1 >= 0 and lon_delta1 <= 0:
                            angel1 += 180
                        elif lat_delta1 <= 0 and lon_delta1 >= 0:
                            angel1 = angel1
                        elif lat_delta1 <= 0 and lon_delta1 <= 0:
                            angel1 -= 180

                        # 计算角度差
                        angle_delta = angel1 - angel2
                    # 如果是起点
                    else:
                        angle_delta = 0

                    path_delta.append((path[i][0], path[j][1], angle_delta))

                    # 找到了下一个里程碑就退出
                    flag = True
                    break

            # 还要计算下一时刻图像的偏转方向
            # 如果不是终点
            if flag is True:
                # 如果不是起点
                if i > 0:
                    # 计算下一时刻位置距离当前位置的角度
                    lat_delta2 = path[i + 1][2] - path[i][2]
                    lon_delta2 = path[i + 1][3] - path[i][3]
                    tan2 = float(lat_delta2 / (lon_delta2 + 1e-10))
                    # -90~90°
                    angel2 = math.atan(tan2) / math.pi * 180

                    # x正轴：-180~180°
                    if lat_delta2 >= 0 and lon_delta2 >= 0:
                        angel2 = angel2
                    elif lat_delta2 >= 0 and lon_delta2 <= 0:
                        angel2 += 180
                    elif lat_delta2 <= 0 and lon_delta2 >= 0:
                        angel2 = angel2
                    elif lat_delta2 <= 0 and lon_delta2 <= 0:
                        angel2 -= 180

                    # 计算当前位置距离上一时刻位置的角度
                    lat_delta1 = path[i][2] - path[i - 1][2]
                    lon_delta1 = path[i][3] - path[i - 1][3]
                    tan1 = float(lat_delta1 / (lon_delta1 + 1e-10))
                    # -90~90°
                    angel1 = math.atan(tan1) / math.pi * 180

                    # x正轴：-180~180°
                    if lat_delta1 >= 0 and lon_delta1 >= 0:
                        angel1 = angel1
                    elif lat_delta1 >= 0 and lon_delta1 <= 0:
                        angel1 += 180
                    elif lat_delta1 <= 0 and lon_delta1 >= 0:
                        angel1 = angel1
                    elif lat_delta1 <= 0 and lon_delta1 <= 0:
                        angel1 -= 180

                    # 计算角度差
                    angle_delta = angel1 - angel2
                # 如果是起点
                else:
                    angle_delta = 0

                path_delta[-1] = (path_delta[-1][0], angle_delta, path_delta[-1][1], path_delta[-1][2])

                if path_delta[-1][1] < a:
                    a = path_delta[-1][1]
                if path_delta[-1][1] > b:
                    b = path_delta[-1][1]
                if path_delta[-1][3] < c:
                    c = path_delta[-1][3]
                if path_delta[-1][3] > d:
                    d = path_delta[-1][3]

        res_delta.append(path_delta)

    print(a)
    print(b)
    print(c)
    print(d)

    if os.path.exists("datasets") is False:
        os.mkdir("datasets")
    for path in res_delta:
        for pic in path:
            name, angle, stone, stone_angle = pic[0], pic[1], pic[2], pic[3]
            new_txt_name = name[len("processOrder/order/"):len("processOrder/order/") + 5]
            with open("datasets/" + new_txt_name + ".txt", "a") as file1:
                file1.write(name + " " + str(angle) + " " + str(stone) + " " + str(stone_angle) + "\n")
            file1.close()


class OrderTrainDataset(Dataset):
    def __init__(self, transform, num_classes2, input_len):
        self.transform = transform
        self.num_classes2 = num_classes2
        self.input_len = input_len

        res_seq = []
        files = os.listdir("datasets")
        files.sort()
        for i in range(0, int(0.8 * len(files))):
            path_seq = []
            file = files[i]
            full_file_path = os.path.join("datasets", file)
            f = open(full_file_path, 'rt')
            for line in f:
                line = line.strip('\n')
                line = line.split(' ')
                line = [line[0], float(line[1]) + 90, float(line[2]), float(line[3]) + 90]
                path_seq.append(line)

            for i in range(0, len(path_seq) - self.input_len):
                res_seq.append(path_seq[i: i + self.input_len])

        print(len(res_seq))
        self.imgs = res_seq

    # 返回数据集大小
    def __len__(self):
        return len(self.imgs)

    # 打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        # len, 4
        item = self.imgs[index]
        imgs = None
        angles = []
        for i in range(0, self.input_len):
            img = item[i][0]
            img = Image.open(img)
            img = img.convert('RGB')
            img = self.transform(img).unsqueeze(dim=0)
            if imgs is None:
                imgs = img
            else:
                imgs = torch.cat((imgs, img), dim=0)

            angles.append(item[i][1])

        angles = torch.tensor(angles, dtype=torch.int)
        return imgs, angles, item[-1][2], item[-1][3]


class OrderTestDataset(Dataset):
    def __init__(self, transform, num_classes2, input_len):
        self.transform = transform
        self.num_classes2 = num_classes2
        self.input_len = input_len

        res_seq = []
        files = os.listdir("datasets")
        files.sort()
        for i in range(int(0.8 * len(files)), len(files)):
            path_seq = []
            file = files[i]
            full_file_path = os.path.join("datasets", file)
            f = open(full_file_path, 'rt')
            for line in f:
                line = line.strip('\n')
                line = line.split(' ')
                line = [line[0], float(line[1]) + 90, float(line[2]), float(line[3]) + 90]
                path_seq.append(line)

            for i in range(0, len(path_seq) - self.input_len):
                res_seq.append(path_seq[i: i + self.input_len])

        print(len(res_seq))
        self.imgs = res_seq

    # 返回数据集大小
    def __len__(self):
        return len(self.imgs)

    # 打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        # 100, 4
        item = self.imgs[index]
        imgs = None
        angles = []

        for i in range(0, self.input_len):
            img = item[i][0]
            img = Image.open(img)
            img = img.convert('RGB')
            img = self.transform(img).unsqueeze(dim=0)
            if imgs is None:
                imgs = img
            else:
                imgs = torch.cat((imgs, img), dim=0)

            angles.append(item[i][1])

        angles = torch.tensor(angles, dtype=torch.int)
        return imgs, angles, item[-1][2], item[-1][3]


if __name__ == '__main__':
    write_img_angle(k=100)
    # train_dataset = OrderTrainDataset(transform=None)
    # test_dataset = OrderTestDataset(transform=None)
