import argparse
import math
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
from PIL import Image
from selenium import webdriver
import time
from facaformer import FACAFormer

torch.set_printoptions(precision=8)

parser = argparse.ArgumentParser(description='PyTorch Testing')

parser.add_argument('--num_classes1', default=100, type=int,
                    metavar='N', help='the number of milestone labels')
parser.add_argument('--num_classes2', default=2, type=int,
                    metavar='N', help='the number of angle labels (latitude and longitude)')
parser.add_argument('--len', default=6, type=int,
                    metavar='LEN', help='the number of model input sequence length')

parser.add_argument('--resume', default='save11/model_angle_avg_best.pth.tar', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')

parser.add_argument('--start_path', default=["74551/30.3093714,119.9232729.png"],
                    type=list,
                    metavar='PATH',
                    help='the frame path of initial position')
parser.add_argument('--dest_path', default="74551/30.2620077,119.947846.png",
                    type=str,
                    metavar='PATH',
                    help='the frame path of destination')
parser.add_argument('--next_angles', default=[],
                    type=list,
                    metavar='ANGLE',
                    help='input angle sequence')
parser.add_argument('--last_pos', default=[30.3093714, 119.9232729],
                    type=list,
                    metavar='ANGLE',
                    help='last position: lat and lon')
parser.add_argument('--dest_pos', default=[30.2620077, 119.947846],
                    type=list,
                    metavar='ANGLE',
                    help='destination position: lat and lon')


def main():
    """
    real-time testing process control: loading model, dataset, screenshot.
    :return:
    """
    args = parser.parse_args()

    # create model
    print("=> creating model")
    backbone = torchvision_models.mobilenet_v3_small(pretrained=True)

    model = FACAFormer(
        backbone=backbone,
        extractor_dim=576,
        num_classes1=args.num_classes1,
        num_classes2=args.num_classes2,
        len=args.len,
        dim=512,
        depth=6,
        heads=8,
        dim_head=64,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    )

    # load from resume, start training from a certain epoch
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')

            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict
            for key in list(state_dict.keys()):
                new_state_dict[key[len("module."):]] = state_dict[key]
                del new_state_dict[key]
            model.load_state_dict(new_state_dict, strict=False)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc2 = checkpoint['best_acc2']
            best_acc3 = checkpoint['best_acc3']
            print("best_acc1: " + str(best_acc1))
            print("best_acc2: " + str(best_acc2))
            print("best_acc3: " + str(best_acc3))

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # reduce CPU usage, use it after the model is loaded onto the GPU
    torch.set_num_threads(1)
    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    # real-time testing
    real_time_test(val_transform, model, args)


def generate_inputs(val_transform, args):
    """
    generate model inputs.
    :param val_transform: torchvision.transforms.
    :param args:
    :return: frame sequence and angle sequence.
    """
    # slice to avoid modifying the original list
    start_path = args.start_path[:]
    next_angles = args.next_angles[:]

    next_imgs = None

    # original images and angles
    for i in range(0, len(start_path)):
        img = start_path[i]
        img = Image.open(img)
        img = img.convert('RGB')
        img = val_transform(img).unsqueeze(dim=0)
        if next_imgs is None:
            next_imgs = img
        else:
            next_imgs = torch.cat((next_imgs, img), dim=0)
    next_angles.append([0, 0])

    # append destination frame angle destination angle as part of model input
    dest_img = Image.open(args.dest_path)
    dest_img = dest_img.convert('RGB')
    dest_img = val_transform(dest_img).unsqueeze(dim=0)
    next_imgs = torch.cat((next_imgs, dest_img), dim=0)

    dest_angle = [0, 0]
    next_angles.append(dest_angle)
    next_angles = torch.tensor(next_angles, dtype=torch.float)

    # if there are not enough input frames
    for i in range(0, args.len - 1 - len(start_path)):
        next_imgs = torch.cat((next_imgs, torch.zeros((1, 3, 224, 224))), dim=0)
        next_angles = torch.cat((next_angles, torch.zeros((1, 2))), dim=0)

    # add a batch dimension
    return next_imgs.unsqueeze(dim=0), next_angles.unsqueeze(dim=0)


def real_time_test(val_transform, model, args):
    """
    real-time testing process.
    :param val_transform: torchvision.transforms.
    :param model: saved checkpoint.
    :param args:
    :return: actually arrived destination.
    """
    for i in range(0, 220):
        images, next_angles = generate_inputs(val_transform, args)

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            # b,len,3,224,224
            images = images.to(dtype=torch.float32)
            # b,len,2
            next_angles = next_angles.to(dtype=torch.float32)

            # b,len,3,224,224+b,len,2
            output1, output2, output3 = model(images, next_angles)
            _, preds = output1.max(1)
            print("the current milestone: " + str(preds.item()))
            _, preds = output2.max(1)
            print("the next target milestone: " + str(preds.item()))
            output3 = output3.view(-1).numpy().tolist()
            print("the next steering angle: " + str(output3[0]) + "," + str(output3[1]))

            # calculate the next steering angle
            tan = output3[0] / output3[1]
            ang = math.atan(tan) * 190 / math.pi
            if output3[0] >= 0 and output3[1] <= 0:
                ang += 180
            elif output3[0] <= 0 and output3[1] <= 0:
                ang -= 180

            # calculate the moving distance
            dis = 30
            print("dis:" + str(dis))
            lat_delta = dis * output3[0]
            lon_delta = dis * output3[1]
            lat_delta = float(lat_delta / 111000)
            lon_delta = float(lon_delta / 111000 / math.cos(args.last_pos[0] / 180 * math.pi))

            # calculate the new position
            new_lat = args.last_pos[0] + lat_delta
            new_lon = args.last_pos[1] + lon_delta
            print("new_lat: " + str(new_lat))
            print("new_lon: " + str(new_lon))

            # screenshot new frame
            coords = str(new_lat) + "," + str(new_lon)
            path = args.start_path[0][0: 6] + coords + '.png'
            if os.path.exists(path) is False:
                path = screenshot(coords, path, ang)

            # append new frame and new angle
            args.start_path.append(path)
            args.next_angles.append([output3[0], output3[1]])
            args.last_pos = [new_lat, new_lon]
            # discard excess frames
            if len(args.start_path) > 5:
                args.start_path.pop(0)
                args.next_angles.pop(0)

            # If the distance errors of longitude and latitude are both within 20 meters,
            # it means that we have reached the destination.
            if abs(new_lat - args.dest_pos[0]) * 111000 <= 20 and \
                abs(new_lon - args.dest_pos[1]) * 111000 * math.cos(args.last_pos[0]) <= 20:
                print("You reach the destination successfully!")
                break

    return new_lat, new_lon


def screenshot(coords, path, ang):
    """
    screenshot from Google Earth API.
    :param coords: position needed to be screenshot.
    :param path: saved path of new frame.
    :param ang: rotated angle of new frame.
    :return: saved path of new frame.
    """
    DRIVER = 'chromedriver'
    option = webdriver.ChromeOptions()
    option.add_argument('headless')
    option.add_argument('start-maximized')
    driver = webdriver.Chrome(executable_path=DRIVER, options=option)

    zoom = "11.52249204a,151.71390185d,35y"
    url = "https://earth.google.com/web/@" + coords + "," + zoom
    driver.get(url)

    time.sleep(40)
    # driver.quit()

    driver.save_screenshot(path)

    pic = Image.open(path)
    pic = pic.rotate(90 - ang)
    pic = pic.resize((320, 180))
    pic.save(path)

    return path


if __name__ == '__main__':
    main()
