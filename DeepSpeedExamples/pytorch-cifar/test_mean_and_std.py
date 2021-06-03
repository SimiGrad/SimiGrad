import utils
import torchvision
import torch

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True,
     transform=torchvision.transforms.ToTensor())

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True,
     transform=torchvision.transforms.ToTensor())

trainset_dataloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
testset_dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)
mean = torch.zeros(3)
std = torch.zeros(3)
print('==> Computing mean and std..')
for inputs, targets in trainset_dataloader:
    for i in range(3):
        mean[i] += inputs[:,i,:,:].mean()
        std[i] += inputs[:,i,:,:].std()
# for inputs, targets in testset_dataloader:
#     for i in range(3):
#         mean[i] += inputs[:,i,:,:].mean()
#         std[i] += inputs[:,i,:,:].std()
mean.div_(len(trainset_dataloader)+len(testset_dataloader))
std.div_(len(trainset_dataloader)+len(testset_dataloader))
print(mean, std)
