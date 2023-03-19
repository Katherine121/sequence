import argparse
import shutil

import numpy as np
import os

import timm.models
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from thop import profile
from torch import nn
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision import transforms


class KmeansDataset(Dataset):
    def __init__(self, transform):
        self.transform = transform

        images = []
        path1 = "../processOrder/order"

        dir_path1 = os.listdir(path1)
        dir_path1.sort()

        for i in range(0, len(dir_path1)):
            if i % 5 != 0:
                dir1 = dir_path1[i]
                full_dir_path1 = os.path.join(path1, dir1)
                print(dir1)

                file_path1 = os.listdir(full_dir_path1)
                file_path1.sort()

                for file1 in file_path1:
                    full_file_path1 = os.path.join(full_dir_path1, file1)
                    images.append(full_file_path1)

        self.imgs = images
        print(len(images))

    # 返回数据集大小
    def __len__(self):
        return len(self.imgs)

    # 打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        img = self.imgs[index]
        img = Image.open(img)
        img = img.convert('RGB')
        img = self.transform(img)
        return img, 1


class ViT(nn.Module):
    def __init__(self):
        super(ViT, self).__init__()
        self.backbone = timm.models.vit_base_patch16_224(pretrained=True)

    def forward(self, x):
        B = x.shape[0]
        patch_x = self.backbone.patch_embed(x)

        cls_tokens = self.backbone.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        patch_x = torch.cat((cls_tokens, patch_x), dim=1)
        patch_x = patch_x + self.backbone.pos_embed
        patch_x = self.backbone.pos_drop(patch_x)

        for blk in self.backbone.blocks:
            patch_x = blk(patch_x)

        patch_x = self.backbone.norm(patch_x)

        return patch_x[:, 0, :]


def vstackDescriptors(descriptor_list):
    # 45,b,768
    descriptors = []
    for i in range(0, len(descriptor_list)):
        # b,768
        descriptor = descriptor_list[i].tolist()
        # 45*b,768
        descriptors.extend(descriptor)
        if i % 10 == 0:
            print(i)
            print(len(descriptors))

    # 22695,768
    return np.array(descriptors)


def clusterDescriptors(descriptors, no_clusters):
    kmeans = KMeans(n_clusters=no_clusters).fit(descriptors)
    return kmeans


def normalizeFeatures(scale, features):
    return scale.transform(features)


def trainModel(no_clusters):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    kmeansDataset = KmeansDataset(transform=transform)
    kmeansloader = torch.utils.data.DataLoader(
        kmeansDataset, batch_size=512, shuffle=False, drop_last=False)

    print("Train images path detected.")
    model = ViT()
    flops, params = profile(model,
                            (torch.randn((1, 3, 224, 224)), ))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    model = model.cuda()

    torch.set_num_threads(1)
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    model.eval()
    cls_descriptor_list = []

    # with torch.no_grad():
    #     for i, (images, labels) in enumerate(kmeansloader):
    #         images = images.cuda().to(dtype=torch.float32)
    #         # b,3,224,224->b,dim
    #         output1 = model(images)
    #         output1 = output1.cpu().detach().numpy()
    #         # 45,b,dim
    #         cls_descriptor_list.append(output1)
    #
    #         if os.path.exists("../processOrder/kmeans") is False:
    #             os.mkdir("../processOrder/kmeans")
    #         # b,dim
    #         np.save("../processOrder/kmeans/cls_descriptor" + str(i) + ".npy", output1)
    #
    #         if i % 10 == 0:
    #             print(i)

    for i in range(0, 45):
        output1 = np.load("../processOrder/kmeans/cls_descriptor" + str(i) + ".npy")
        # 45,b,768
        cls_descriptor_list.append(output1)
        if i % 10 == 0:
            print(i)

    print(len(cls_descriptor_list))
    patch_descriptor_npy = vstackDescriptors(cls_descriptor_list)
    # 22695,768
    kmeans = clusterDescriptors(patch_descriptor_npy, no_clusters)
    # 22695
    np.save("../processOrder/kmeans/labels.npy", kmeans.labels_)
    # 100*dim
    np.save("../processOrder/kmeans/cluster_centers.npy", kmeans.cluster_centers_)
    print("Descriptors clustered.")


def main(no_clusters):
    trainModel(no_clusters)


def save_files():
    # 22695
    labels = np.load("../processOrder/kmeans/labels.npy")
    # 100*dim
    centers = np.load("../processOrder/kmeans/cluster_centers.npy")

    path1 = "../processOrder/order"

    dir_path1 = os.listdir(path1)
    dir_path1.sort()

    kmeansDataset = KmeansDataset(transform=None)
    print(len(kmeansDataset))
    print(labels.shape)
    j = 0
    for full_file_path1 in kmeansDataset.imgs:
        label = labels[j]
        new_dir = "../processOrder/kmeans_100"
        if os.path.exists(new_dir) is False:
            os.mkdir(new_dir)
        new_dir = "../processOrder/kmeans_100/all_class/"
        if os.path.exists(new_dir) is False:
            os.mkdir(new_dir)
        new_dir = "../processOrder/kmeans_100/all_class/" + str(label)
        if os.path.exists(new_dir) is False:
            os.mkdir(new_dir)
        shutil.copy(full_file_path1, new_dir)
        j += 1


def compute_diff(path1):
    all_lat_var = 0
    all_lon_var = 0

    dir_path1 = os.listdir(path1)
    dir_path1.sort()

    for i in range(0, len(dir_path1)):
        dir1 = dir_path1[i]
        full_dir_path1 = os.path.join(path1, dir1)

        file_path1 = os.listdir(full_dir_path1)
        file_path1.sort()

        pos = []

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

            pos.append([float(lat_pos), float(lon_pos)])

        pos = np.array(pos)
        var = np.std(pos, axis=0)

        all_lat_var += var[0]
        all_lon_var += var[1]

    all_lat_var /= i
    all_lon_var /= i
    print(all_lat_var)
    print(all_lon_var)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_clusters', action="store", dest="no_clusters", default=100)
    args = vars(parser.parse_args())

    # main(int(args['no_clusters']))
    # save_files()
    compute_diff(path1="../processOrder/100/all_class")
    compute_diff(path1="../processOrder/kmeans_100/all_class")
