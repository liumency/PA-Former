# coding=utf-8
import os
import torch.utils.data
from data_utils import TestDatasetFromFolder, calMetric_iou
import cv2
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import numpy as np
from model.network import CDNet


parser = argparse.ArgumentParser(description='Test Change Detection Models')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--model_dir', default='epochs/Google512/paformer/netCD_epoch_247.pth', type=str)
parser.add_argument('--hr1_dir', default='../Data/Google512/test/time1', type=str)
parser.add_argument('--lr2_dir', default='../Data/Google512/test/time2', type=str)
parser.add_argument('--label_dir', default='../Data/Google512/test/label', type=str)
parser.add_argument('--save_cd', default='results/Google512/paformer', type=str)

parser.add_argument('--suffix', default=['.png','.jpg','.tif'], type=list, help='the suffix of the image files.')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(args.save_cd):
    os.mkdir(args.save_cd)

netCD = CDNet(n_class=2).to(device, dtype=torch.float)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    netCD = torch.nn.DataParallel(netCD, device_ids=range(torch.cuda.device_count()))

netCD.load_state_dict(torch.load(args.model_dir))
netCD.eval()

if __name__ == '__main__':
    test_set = TestDatasetFromFolder(args, args.hr1_dir, args.lr2_dir, args.label_dir)
    test_loader = DataLoader(dataset=test_set, num_workers=24, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader)

    inter, unin= 0,0

    for hr_img1, lr_img2, label, image_name in test_bar:

        hr_img1 = hr_img1.to(device, dtype=torch.float)
        lr_img2 = lr_img2.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.float)

        prob = netCD(hr_img1, lr_img2)

        prob=prob.squeeze(0)
        prob = torch.argmax(prob, 0)
        prob = prob.cpu().data.numpy()
        result = np.squeeze(prob)

        label = label.squeeze(0)
        label = torch.argmax(label, 0)
        gt_value = label.cpu().detach().numpy()
        gt_value = np.squeeze(gt_value)

        intr, unn = calMetric_iou(gt_value, result)
        inter = inter + intr
        unin = unin + unn

        test_bar.set_description(
            desc='IoU: %.4f' % (inter * 1.0 / unin))
        #
        cv2.imwrite(args.save_cd + image_name[0], result*255)