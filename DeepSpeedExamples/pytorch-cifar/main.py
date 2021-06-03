'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

import random
import numpy as np

from models import *
from utils import progress_bar

import deepspeed

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--name', type=str, help='test name')
parser.add_argument('--similarity_target', type=float)
parser.add_argument('--batchsize_upper_bound', type=str)
parser.add_argument('--batchsize_lower_bound', type=str)
parser.add_argument('--warmup', action="store_true")
parser.add_argument('--local_rank', type=int, help='for deepspeed')
parser.add_argument('--seed', type=int, help='manual seed')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

if args.batchsize_upper_bound:
    if args.batchsize_upper_bound.endswith("k"):
        args.batchsize_upper_bound=int(args.batchsize_upper_bound.replace("k",""))*1024
    else:
        args.batchsize_upper_bound=int(args.batchsize_upper_bound)
    if args.batchsize_upper_bound<=0:
        args.batchsize_upper_bound=None

if args.batchsize_lower_bound:
    if args.batchsize_lower_bound.endswith("k"):
        args.batchsize_lower_bound=int(args.batchsize_lower_bound.replace("k",""))*1024
    else:
        args.batchsize_lower_bound=int(args.batchsize_lower_bound)
    if args.batchsize_lower_bound<=0:
        args.batchsize_lower_bound=None

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = resnet20_cifar()
# net = net.to(device)
# if device == 'cuda':
    # net = torch.nn.DataParallel(net)
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80, 120], 0.1)

#deepspeed
from torch.utils.tensorboard import SummaryWriter
if args.local_rank==0:
    writer = SummaryWriter(os.path.join("log",args.name))
else:
    writer=None
# Random seed
print('Random Seed is %d' % (args.seed))
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
parameters = filter(lambda p: p.requires_grad, net.parameters())
if args.similarity_target>=0:
    model, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=net,optimizer=optimizer, model_parameters=parameters,
        adaptive_batch_params={"enable_adjust":True,"verbose":True,"similarity_target":args.similarity_target,"max_micro_batch_size":512,"batch_size_upper_bound":args.batchsize_upper_bound,"batch_size_lower_bound":args.batchsize_lower_bound})
else:
    model, optimizer, trainloader, __ = deepspeed.initialize(
        args=args, model=net,optimizer=optimizer, model_parameters=parameters,
        adaptive_batch_params={"enable_adjust":False,"verbose":True,"similarity_target":args.similarity_target,"max_micro_batch_size":512,"batch_size_upper_bound":args.batchsize_upper_bound,"batch_size_lower_bound":args.batchsize_lower_bound})

args.batch_size=model.train_micro_batch_size_per_gpu()


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1,sampler=train_sampler,num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
test_sampler = torch.utils.data.distributed.DistributedSampler(testset)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=512, sampler=test_sampler, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

global_step=0

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    global global_step
    inputs=[]
    targets=[]
    for batch_idx, (input, target) in enumerate(trainloader):
        #qy: adaptive micro batch size
        if len(inputs)<model.train_micro_batch_size_per_gpu():
            inputs.append(input)
            targets.append(target)
            continue
        else:
            inputs=torch.cat(inputs)
            targets=torch.cat(targets)
        inputs, targets = inputs.to(model.local_rank), targets.to(model.local_rank)
        # inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        model.backward(loss)
        model.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if model.is_gradient_accumulation_boundary() and torch.distributed.get_rank()==0:
            global_step+=1
            # print(f"model update at {model.micro_steps}")
            if hasattr(model.optimizer,"optimizer"):
                for param_group in model.optimizer.optimizer.param_groups:
                    lr_this_step=param_group['lr']
            else:
                for param_group in model.optimizer.param_groups:
                    lr_this_step=param_group['lr']
            _epoch=epoch+((batch_idx+1)/len(trainloader))
            writer.add_scalar(f'training_loss', train_loss/(batch_idx+1),
                                        _epoch)
            writer.add_scalar(f'adjusted_learning_rate', lr_this_step*model.adaptive_batch_params["global_lr_modifier"], _epoch)

            writer.add_scalar(f'cos_similarity', model.cos_placeholder if model.cos_placeholder else float('NaN'),
                                        _epoch)
            writer.add_scalar(f'adjust_step', model.gradient_step_size if model.gradient_step_size else float('NaN'),
                                        _epoch)
            writer.add_scalar(f'accumulation_steps',model.gradient_accumulation_steps() , _epoch)
            writer.add_scalar(f'adjusted_batch_size',model.gradient_accumulation_steps()*torch.distributed.get_world_size()*model.train_micro_batch_size_per_gpu() , _epoch)
            writer.add_scalar(f'global_step',global_step, _epoch)
        inputs=[]
        targets=[]
        torch.distributed.barrier()

    progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(model.local_rank), targets.to(model.local_rank)
            # inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    torch.distributed.barrier()
    torch.distributed.all_reduce(torch.Tensor([acc]).to(model.local_rank))
    acc=acc/torch.distributed.get_world_size()
    if torch.distributed.get_rank()==0:
        writer.add_scalar('test_accuracy', acc, epoch + 1)
        writer.add_scalar('test_loss', test_loss/(batch_idx+1), epoch + 1)

def warmup(epoch,optimizer):
    if epoch>int(args.epoch*5.5/100):
        return
    elif epoch==int(args.epoch*5.5/100):
        lr=args.lr
    else:
        lr=0.1+(args.lr-0.1)*(epoch/(args.epoch*5.5/100))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

args.epoch=200
for epoch in range(start_epoch, start_epoch+200):
    if args.warmup:
        warmup(epoch,optimizer)
    train(epoch)
    test(epoch)
    scheduler.step()
    print("Current learning rate",optimizer.param_groups[0]['lr'])


