import argparse

#training options
parser = argparse.ArgumentParser(description='Training Change Detection Network')

# training parameters
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')
parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
parser.add_argument('--val_batchsize', default=8, type=int, help='batchsize for validation')
parser.add_argument('--num_workers', default=24, type=int, help='num of workers')
parser.add_argument('--n_class', default=2, type=int, help='number of class')
parser.add_argument('--gpu_id', default="0", type=str, help='which gpu to run.')
parser.add_argument('--suffix', default=['.png','.jpg','.tif'], type=list, help='the suffix of the image files.')

parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')

# path for loading data from folder
parser.add_argument('--hr1_train', default='../Data/Google512/train/time1', type=str, help='image at t1 in training set')
parser.add_argument('--hr2_train', default='../Data/Google512/train/time2', type=str, help='image at t2 in training set')
parser.add_argument('--lab_train', default='../Data/Google512/train/label', type=str, help='label image in training set')

parser.add_argument('--hr1_val', default='../Data/Google512/val/time1', type=str, help='image at t1 in validation set')
parser.add_argument('--hr2_val', default='../Data/Google512/val/time2', type=str, help='image at t2 in validation set')
parser.add_argument('--lab_val', default='../Data/Google512/val/label', type=str, help='label image in validation set')

# network saving
parser.add_argument('--model_dir', default='epochs/Google512/paformer/', type=str, help='model save path')
