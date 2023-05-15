import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class DronetTrainDataset(Dataset):
    def __init__(self, dataset_path, transform, input_len):
        """
        baseline train dataset.
        :param transform: torchvision.transforms.
        :param input_len: input sequence length of the model.
        """
        self.transform = transform
        self.input_len = input_len

        res_seq = []
        files = os.listdir(dataset_path)
        files.sort()
        for i in range(0, len(files)):
            # 0.8 as train dataset
            if i % 5 != 0:
                path_seq = []
                file = files[i]
                full_file_path = os.path.join(dataset_path, file)
                f = open(full_file_path, 'rt')
                for line in f:
                    line = line.strip('\n')
                    line = line.split(' ')

                    # read frame, angle,
                    # end point frame,
                    # the current position label,
                    # the next position label, the direction angle
                    line = [line[0], [float(line[1]), float(line[2])],
                            line[3],
                            int(line[4]),
                            int(line[5]), [float(line[6]), float(line[7])]
                            ]
                    path_seq.append(line)

                # if there are not enough input frames
                for j in range(1, len(path_seq) + 1):
                    if j - self.input_len < 0:
                        res_seq.append(path_seq[0: j])
                    else:
                        res_seq.append(path_seq[j - self.input_len: j])

        print(len(res_seq))
        self.imgs = res_seq

    def __len__(self):
        """
        return the length of the dataset.
        :return:
        """
        return len(self.imgs)

    def __getitem__(self, index):
        """
        read the image and label corresponding to the index in the dataset.
        :param index: index of self.imgs.
        :return: image, the current position, the next position, the direction angle.
        """
        item = self.imgs[index]

        img = item[-1][0]
        img = Image.open(img)
        img = img.convert('RGB')
        img = self.transform(img)

        label = item[-1][3]
        stone_img = item[-1][4]
        stone_angle = torch.tensor(item[-1][5], dtype=torch.float)

        # image, the current position, the next position, the direction angle
        return img, label, stone_img, stone_angle


class DronetTestDataset(Dataset):
    def __init__(self, dataset_path, transform, input_len):
        """
        baseline test dataset.
        :param transform: torchvision.transforms.
        :param input_len: input sequence length of the model.
        """
        self.transform = transform
        self.input_len = input_len

        res_seq = []
        files = os.listdir(dataset_path)
        files.sort()
        for i in range(0, len(files)):
            # 0.2 as test dataset
            if i % 5 == 0:
                path_seq = []
                file = files[i]
                full_file_path = os.path.join(dataset_path, file)
                f = open(full_file_path, 'rt')
                for line in f:
                    line = line.strip('\n')
                    line = line.split(' ')

                    # read frame, angle,
                    # end point frame,
                    # the current position label,
                    # the next position label, the direction angle
                    line = [line[0], [float(line[1]), float(line[2])],
                            line[3],
                            int(line[4]),
                            int(line[5]), [float(line[6]), float(line[7])]
                            ]
                    path_seq.append(line)

                # if there are not enough input frames
                for j in range(1, len(path_seq) + 1):
                    if j - self.input_len < 0:
                        res_seq.append(path_seq[0: j])
                    else:
                        res_seq.append(path_seq[j - self.input_len: j])

        print(len(res_seq))
        self.imgs = res_seq

    def __len__(self):
        """
        return the length of the dataset.
        :return:
        """
        return len(self.imgs)

    def __getitem__(self, index):
        """
        read the image and label corresponding to the index in the dataset.
        :param index: index of self.imgs.
        :return: image, the current position, the next position, the direction angle.
        """
        item = self.imgs[index]

        img = item[-1][0]
        img = Image.open(img)
        img = img.convert('RGB')
        img = self.transform(img)

        label = item[-1][3]
        stone_img = item[-1][4]
        stone_angle = torch.tensor(item[-1][5], dtype=torch.float)

        # image, the current position, the next position, the direction angle
        return img, label, stone_img, stone_angle
