#coding=utf-8
from os.path import join
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import torchvision.transforms as transforms
import os
from scipy.ndimage.morphology import distance_transform_edt
import cv2
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.tif','.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

def calMetric_iou(predict, label):
    tp = np.sum(np.logical_and(predict == 1, label == 1))
    fp = np.sum(predict==1)
    fn = np.sum(label == 1)
    return tp,fp+fn-tp


def getDataList(img_path):
    dataline = open(img_path, 'r').readlines()
    datalist =[]
    for line in dataline:
        temp = line.strip('\n')
        datalist.append(temp)
    return datalist


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result


def onehot_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)
    """
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap


def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)



class LoadDatasetFromFolder(Dataset):
    def __init__(self, args, hr1_path, hr2_path, lab_path):
        super(LoadDatasetFromFolder, self).__init__()
        # 获取图片列表
        datalist = [name for name in os.listdir(hr1_path) for item in args.suffix if
                      os.path.splitext(name)[1] == item]

        self.hr1_filenames = [join(hr1_path, x) for x in datalist if is_image_file(x)]
        self.hr2_filenames = [join(hr2_path, x) for x in datalist if is_image_file(x)]
        self.lab_filenames = [join(lab_path, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()  # only convert to tensor

    def __getitem__(self, index):
        hr1_img = self.transform(Image.open(self.hr1_filenames[index]).convert('RGB'))
        # lr2_img = self.transform(Image.open(self.lr2_filenames[index]).convert('RGB'))
        hr2_img = self.transform(Image.open(self.hr2_filenames[index]).convert('RGB'))

        label = self.label_transform(Image.open(self.lab_filenames[index]))
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)

        return hr1_img, hr2_img, label

    def __len__(self):
        return len(self.hr1_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, args, Time1_dir, Time2_dir, Label_dir):
        super(TestDatasetFromFolder, self).__init__()

        datalist = [name for name in os.listdir(Time1_dir) for item in args.suffix if
                    os.path.splitext(name)[1] == item]

        self.image1_filenames = [join(Time1_dir, x) for x in datalist if is_image_file(x)]
        self.image2_filenames = [join(Time2_dir, x) for x in datalist if is_image_file(x)]
        self.image3_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=True)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()

    def __getitem__(self, index):
        image1 = self.transform(Image.open(self.image1_filenames[index]).convert('RGB'))
        image2 = self.transform(Image.open(self.image2_filenames[index]).convert('RGB'))

        label = self.label_transform(Image.open(self.image3_filenames[index]))
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)

        image_name =  self.image1_filenames[index].split('/', -1)
        image_name = image_name[len(image_name)-1]

        return image1, image2, label, image_name

    def __len__(self):
        return len(self.image1_filenames)



def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return Image.fromarray(mask - mask_erode)


class trainImageAug(object):
    def __init__(self, crop = True, augment = True, angle = 30):
        self.crop =crop
        self.augment = augment
        self.angle = angle

    def __call__(self, image1, image2, mask):
        if self.crop:
            w = np.random.randint(0,512)
            h = np.random.randint(0,512)
            box = (w, h, w+512, h+512)
            image1 = image1.crop(box)
            image2 = image2.crop(box)
            mask = mask.crop(box)
        if self.augment:
            prop = np.random.uniform(0, 1)
            if prop < 0.15:
                image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
                image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            elif prop < 0.3:
                image1 = image1.transpose(Image.FLIP_TOP_BOTTOM)
                image2 = image2.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            elif prop < 0.5:
                image1 = image1.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))
                image2 = image2.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))
                mask = mask.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))

        return image1, image2, mask


def get_transform(convert=True, normalize=False):
    transform_list = []
    if convert:
        transform_list += [
                            transforms.ToTensor(),
                           ]
    if normalize:
        transform_list += [
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


class DA_DatasetFromFolder(Dataset):
    def __init__(self, Image_dir1, Image_dir2, Label_dir, crop=True, augment = True, angle = 30):
        super(DA_DatasetFromFolder, self).__init__()
        # 获取图片列表
        datalist = os.listdir(Image_dir1)
        self.image_filenames1 = [join(Image_dir1, x) for x in datalist if is_image_file(x)]
        self.image_filenames2 = [join(Image_dir2, x) for x in datalist if is_image_file(x)]
        self.label_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]
        self.data_augment = trainImageAug(crop=crop, augment = augment, angle=angle)
        self.img_transform = get_transform(convert=True, normalize=True)
        self.lab_transform = get_transform()

    def __getitem__(self, index):
        image1 = Image.open(self.image_filenames1[index]).convert('RGB')
        image2 = Image.open(self.image_filenames2[index]).convert('RGB')
        label = Image.open(self.label_filenames[index])

        image1, image2, label = self.data_augment(image1, image2, label)
        image1, image2 = self.img_transform(image1), self.img_transform(image2)
        label = self.lab_transform(label)
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)

        return image1, image2, label

    def __len__(self):
        return len(self.image_filenames1)


class trainImageAug2(object):
    def __init__(self, crop = True, augment = True, angle = 30):
        self.crop =crop
        self.augment = augment
        self.angle = angle

    def __call__(self, image1, image2, mask, edge):
        if self.crop:
            w = np.random.randint(0,512)
            h = np.random.randint(0,512)
            box = (w, h, w+512, h+512)
            image1 = image1.crop(box)
            image2 = image2.crop(box)
            mask = mask.crop(box)
            edge = edge.crop(box)
        if self.augment:
            prop = np.random.uniform(0, 1)
            if prop < 0.15:
                image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
                image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
                edge = edge.transpose(Image.FLIP_LEFT_RIGHT)
            elif prop < 0.3:
                image1 = image1.transpose(Image.FLIP_TOP_BOTTOM)
                image2 = image2.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
                edge = edge.transpose(Image.FLIP_TOP_BOTTOM)
            elif prop < 0.5:
                image1 = image1.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))
                image2 = image2.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))
                mask = mask.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))
                edge = edge.rotate(transforms.RandomRotation.get_params([-self.angle, self.angle]))

        return image1, image2, mask, edge

class DA_DatasetFromFolder2(Dataset):
    def __init__(self, Image_dir1, Image_dir2, Label_dir, crop=True, augment = True, angle = 30):
        super(DA_DatasetFromFolder2, self).__init__()
        # 获取图片列表
        datalist = os.listdir(Image_dir1)
        self.image_filenames1 = [join(Image_dir1, x) for x in datalist if is_image_file(x)]
        self.image_filenames2 = [join(Image_dir2, x) for x in datalist if is_image_file(x)]
        self.label_filenames = [join(Label_dir, x) for x in datalist if is_image_file(x)]
        self.data_augment = trainImageAug2(crop=crop, augment = augment, angle=angle)
        self.img_transform = get_transform(convert=True, normalize=True)
        self.lab_transform = get_transform()

    def __getitem__(self, index):
        image1 = Image.open(self.image_filenames1[index]).convert('RGB')
        image2 = Image.open(self.image_filenames2[index]).convert('RGB')
        label = Image.open(self.label_filenames[index])
        edge = mask_to_boundary(np.array(label))

        image1, image2, label, edge = self.data_augment(image1, image2, label, edge)
        image1, image2 = self.img_transform(image1), self.img_transform(image2)
        label = self.lab_transform(label)
        label = make_one_hot(label.unsqueeze(0).long(), 2).squeeze(0)

        edge = self.lab_transform(edge)
        edge = make_one_hot(edge.unsqueeze(0).long(), 2).squeeze(0)

        return image1, image2, label, edge


    def __len__(self):
        return len(self.image_filenames1)

class PredictDatasetFromFolder(Dataset):
    def __init__(self,args, Time1_dir, Time2_dir):
        super(PredictDatasetFromFolder, self).__init__()

        image_sets = [name for name in os.listdir(Time1_dir) for item in args.suffix if
                    os.path.splitext(name)[1] == item]
        self.image1_filenames = [join(Time1_dir, x) for x in image_sets if is_image_file(x)]
        self.image2_filenames = [join(Time2_dir, x) for x in image_sets if is_image_file(x)]

        self.transform = get_transform(convert=True, normalize=False)  # convert to tensor and normalize to [-1,1]
        self.label_transform = get_transform()

    def __getitem__(self, index):
        image1 = self.transform(Image.open(self.image1_filenames[index]).convert('RGB'))
        image2 = self.transform(Image.open(self.image2_filenames[index]).convert('RGB'))

        image_name = self.image1_filenames[index].split('/', -1)
        image_name = image_name[len(image_name) - 1]

        return image1, image2, image_name

    def __len__(self):
        return len(self.image1_filenames)