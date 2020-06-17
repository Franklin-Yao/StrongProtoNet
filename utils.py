import os
import shutil
import time
import pprint
import torch, glob
import numpy as np
import argparse
from enum import Enum
from tensorboardX import SummaryWriter
import os.path as osp

class Model_type(Enum):
    ConvNet = 'ConvNet'
    ResNet12 = 'ResNet12'
    WideResNet = 'WideResNet'

    def __str__(self):
        return self.value


class Method_type(Enum):
    protonet = 'protonet'
    baseline = 'baseline'

    def __str__(self):
        return self.value


def read_many_hdf5(h5file):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read

        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images = np.array(h5file["images"]).astype("uint8")
    labels = np.array(h5file["labels"]).astype("uint8")

    return images, labels


def parse_args(script='train'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_shot', type=int, default=5)
    parser.add_argument('--n_query', type=int, default=15)
    if script == 'train':
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--lr_mul', type=float,
                            default=10)  # lr is the basic learning rate, while lr * lr_mul is the lr for other parts
        parser.add_argument('--weight_decay', type=float, default=0.0005)
        parser.add_argument('--temperature', type=float, default=16)
        parser.add_argument('--balance', type=float, default=10)
        parser.add_argument('--step_size', type=int, default=10)
        parser.add_argument('--n_train_classes', type=int, default=64)
        parser.add_argument('--gamma', type=float, default=0.5)
        parser.add_argument('--method_type', type=Method_type, default=Method_type.baseline, choices=list(Method_type))
        parser.add_argument('--model_type', type=Model_type, default=Model_type.ResNet12, choices=list(Model_type))
        parser.add_argument('--dataset', type=str, default='MiniImageNet',
                            choices=['MiniImageNet', 'CUB', 'TieredImageNet', 'DomainNet'])
        parser.add_argument('--head', type=int, default=1)
        parser.add_argument('--name', default='', help='unique str for saving dir')
        parser.add_argument('--warmup', default=None, help='dir of pretrained weights')
        parser.add_argument('--exp_tag', default='_', choices=['_'],
                            help='different seetings of experiments')
        parser.add_argument('--tsboard', action='store_true')
        parser.add_argument('--train_enc', action='store_true', help='whether to train the encoder')
        parser.add_argument('--resume', action='store_true', help='resume from the last epoch if specified')
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    args.start_epoch = 0
    args.lr = 0.01 if args.method_type is Method_type.baseline else 0.0001

    args.dataset_dir = '/media/frankllin/Data/dataset/crossDomainDatasets/miniImagenet/'
    args.save_dir = 'output'
    args.tf_dir = '%s/log/%s' % (args.save_dir, args.name)
    args.checkpoint_dir = '%s/checkpoints/%s' % (args.save_dir, args.name + '_' + str(args.method_type) + '_' +
                                                 str(args.model_type) + '_' + str(args.n_shot) + 'shot')
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    if args.model_type is Model_type.ResNet12:
        # args.image_size = 80
        args.image_size = 84
    else:
        args.image_size = 84
    args.max_acc = 0.0

    return args


import pickle


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def init():
    args = parse_args(script= 'train')
    print('-----------------------------------------------------------------')
    pretty_print(vars(args))
    print('-----------------------------------------------------------------')

    import multiprocessing
    args.num_workers = multiprocessing.cpu_count()

    if args.tsboard:
        tsboard_dir = osp.join(args.project_dir, 'saves/tsboard/' + args.save_tag)
        ensure_path(tsboard_dir, remove=True)
        writer = SummaryWriter(tsboard_dir)
    else:
        writer = None

    return args, writer

def resume_model(model, optimizer, args, scheduler=None):
    resume_file = os.path.join(args.checkpoint_dir, 'epoch-last.pth')
    if os.path.isfile(resume_file):
        tmp = torch.load(resume_file)
        args.start_epoch = tmp['epoch'] + 1
        args.max_acc = tmp['max_acc']
        model.load_state_dict(tmp['params'])
        optimizer.load_state_dict(tmp['opt_state'])
        if scheduler is not None:
            scheduler.load_state_dict(tmp['scheduler_state'])
        print('Resuming from epoch %d' % (args.start_epoch))
        return True
    else:
        return False

def load_pretrained_weights(model, args):
    print('warmup using %s'%(args.warmup))
    pretrained_dict = torch.load(args.warmup)['params']
    model.load_state_dict(pretrained_dict, strict=False)

def save_model(model, optimizer, args, epoch, max_acc, name, scheduler=None):
    data = {'epoch': epoch,
            'max_acc': max_acc,
            'params': model.state_dict(),
            'opt_state': optimizer.state_dict()}
    if scheduler is not None:
        data['scheduler_state'] = scheduler.state_dict()
    torch.save(data, os.path.join(args.checkpoint_dir, name + '.pth'))

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '[0-9]*.tar'))
    if len(filelist) == 0:
        return None


def ensure_path(path, remove=True):
    if os.path.exists(path):
        if remove:
            # if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()


def pretty_print(x, filename=None):
    _utils_pp.pprint(x)
    if filename is not None:
        with open(filename, "w") as fout:
            pprint.pprint(x, fout)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm
