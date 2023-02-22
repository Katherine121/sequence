import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class DronetTrainDataset(Dataset):
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

        img = item[-1][0]
        img = Image.open(img)
        img = img.convert('RGB')
        img = self.transform(img)

        # 当前里程碑编号、预测的里程碑编号和角度
        label = item[-1][3]
        stone_img = item[-1][4]
        stone_angle = torch.tensor(item[-1][5], dtype=torch.float)

        # 输入图像，输出一个里程碑编号，一对里程碑偏转方向
        return img, label, stone_img, stone_angle


class DronetTestDataset(Dataset):
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

        img = item[-1][0]
        img = Image.open(img)
        img = img.convert('RGB')
        img = self.transform(img)

        # 当前里程碑编号、预测的里程碑编号和角度
        label = item[-1][3]
        stone_img = item[-1][4]
        stone_angle = torch.tensor(item[-1][5], dtype=torch.float)

        # 输入图像，输出一个里程碑编号，一对里程碑偏转方向
        return img, label, stone_img, stone_angle