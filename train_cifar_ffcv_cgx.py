import torch as ch
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms, models
import os
from ffcv.loader import Loader, OrderOption
from torch.utils.data import DataLoader

from typing import List

import torch as ch
import torchvision

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import RandomHorizontalFlip, Cutout, \
    RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter

from tqdm import tqdm

import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)

NCOLS_SCREEN = 85

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset-dir', default=os.path.expanduser('./cifar10'),
                    help='path to training data')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')

parser.add_argument('--batch-size', type=int, default=256,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=256,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.1,
                    help='learning rate for a single GPU')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--dist-backend', choices=['cgx', 'nccl', 'gloo'], default='nccl',
                    help='Backend for torch distributed')
parser.add_argument('--quantization-bits', type=int, default=32,
                    help='Quantization bits for maxmin quantization')
parser.add_argument('--quantization-bucket-size', type=int, default=1024,
                    help='Bucket size for quantization in maxmin quantization')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='Local rank in distributed launch')

args = parser.parse_args()
args.cuda = not args.no_cuda and ch.cuda.is_available()


if "OMPI_COMM_WORLD_SIZE" in os.environ:
    args.local_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4040'
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]

if "WORLD_SIZE" in os.environ:
    import torch_cgx
    args.distributed = int(os.environ["WORLD_SIZE"]) > 1
    local_rank = args.local_rank % ch.cuda.device_count()
    dist.init_process_group(backend=args.dist_backend, init_method="env://")
    args.world_size = ch.distributed.get_world_size()
    rank = ch.distributed.get_rank()
else:
    args.distributed = False
    local_rank = 0
    args.world_size = 1
    rank = 0
print(args)

if args.cuda:
    ch.cuda.set_device(local_rank)
    ch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

verbose = 1 if rank == 0 else 0
ch.set_num_threads(4)


transform_mean, transform_std = CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD
kwargs = {'num_workers': 0} if args.cuda else {}
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(transform_mean, transform_std),
])


CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]
BATCH_SIZE = 256
NUM_GPUS = torch.cuda.device_count()  # specify the number of GPUs available

loaders = {}
for name in ['train', 'test']:
    label_pipeline: List[Operation] = [IntDecoder(), ToTensor(), Squeeze()]  # no ToDevice here yet
    image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

    # Add image transforms and normalization
    if name == 'train':
        image_pipeline.extend([
            RandomHorizontalFlip(),
            RandomTranslate(padding=2),
            Cutout(8, tuple(map(int, CIFAR_MEAN))),  # Note Cutout is done before normalization.
        ])
    image_pipeline.extend([ToTensor(), ToTorchImage(), Convert(ch.float16), 
                       torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

    # Create loaders
    loaders[name] = Loader(f'/tmp/cifar_{name}.beton',
                            batch_size=BATCH_SIZE,
                            num_workers=8,
                            order=OrderOption.RANDOM,
                            drop_last=(name == 'train'),
                            distributed=True, # Enable distributed mode
                            os_cache=True, # Enable OS level caching
                            pipelines={'image': image_pipeline,
                                    'label': label_pipeline})

train_loader = loaders['train']

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(transform_mean, transform_std),
])

val_loader = loaders['test']

    
class Mul(ch.nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight
    def forward(self, x): return x * self.weight

class Flatten(ch.nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

class Residual(ch.nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module
    def forward(self, x): return x + self.module(x)

def conv_bn(channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
    return ch.nn.Sequential(
            ch.nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=False),
            ch.nn.BatchNorm2d(channels_out),
            ch.nn.ReLU(inplace=True)
    )

def construct_model():
    num_class = 10
    model = ch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(ch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        ch.nn.MaxPool2d(2),
        Residual(ch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        ch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        ch.nn.Linear(128, num_class, bias=False),
        Mul(0.2)
    )
    model = model.to(memory_format=ch.channels_last).cuda()
    return model

model = construct_model()

if args.cuda:
    # Move model to GPU.
    model.cuda()

optimizer = optim.SGD(model.parameters(),
                      lr=args.base_lr,
                      momentum=args.momentum, weight_decay=args.wd)
if args.distributed:
    model = DDP(model, device_ids=[local_rank])
    if args.dist_backend == 'cgx':
        assert "OMPI_COMM_WORLD_SIZE" in os.environ, "CGX only works with with mpirun launch"
        from cgx_utils import cgx_hook, CGXState
        state = CGXState(ch.distributed.group.WORLD,
                          compression_params={"bits": args.quantization_bits,
                                              "bucket_size": args.quantization_bucket_size})
        model.register_comm_hook(state, cgx_hook)


def adjust_learning_rate(epoch, batch_idx):
    if epoch < 60:
        lr_adj = 1.
    elif epoch < 120:
        lr_adj = 2e-1
    elif epoch < 160:
        lr_adj = 4e-2
    else:
        lr_adj = 8e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * lr_adj


def train(epoch):
    model.train()
    criterion = ch.nn.CrossEntropyLoss()
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='TEpoch #{}'.format(epoch + 1), disable=not verbose, ncols=NCOLS_SCREEN) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
           
            data = data.float()  # Ensure data is float type
            optimizer.zero_grad()
            output = model(data)
            train_accuracy.update(accuracy(output, target))
            loss = criterion(output, target)
            train_loss.update(loss)
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            loss.backward()
            optimizer.step()
            t.update(1)




def validate(epoch):
    model.eval()
    criterion = ch.nn.CrossEntropyLoss()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')
    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose, ncols=NCOLS_SCREEN) as t:
        with ch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                data = data.float()  # Ensure data is float type
                output = model(data)
                val_loss.update(criterion(output, target))

                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = ch.tensor(0.)
        self.n = ch.tensor(0.)

    def update(self, val):
        self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


num_images = len(train_loader)
for epoch in range(0, args.epochs):
    #if args.world_size > 0:
    #    train_sampler.set_epoch(epoch)
    train(epoch)
    validate(epoch)
    if args.distributed:
        dist.barrier()