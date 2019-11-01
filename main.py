from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--output', type=str, default='', metavar='D',
                    help="folder where output should be stored")
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=510, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

torch.manual_seed(args.seed)

# https://pytorch.org/tutorials/beginner/saving_loading_models.html

# Global Values
use_cuda = True
CHECKPOINT_PATH = args.output + 'resumable_model.pth'
CURRENT_EPOCH = 1
first = True

# Data Initialization and Loading
from data import initialize_data, data_transforms, data_resize_crop, \
    data_rotate, data_hvflip, data_hflip, data_vflip, data_color_jitter, data_grayscale

initialize_data(args.data)  # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([
        datasets.ImageFolder(args.data + '/train_images', transform=data_transforms),
        datasets.ImageFolder(args.data + '/train_images', transform=data_resize_crop),
        datasets.ImageFolder(args.data + '/train_images', transform=data_rotate),
        # datasets.ImageFolder(args.data + '/train_images', transform=data_hvflip),
        # datasets.ImageFolder(args.data + '/train_images', transform=data_hflip),
        # datasets.ImageFolder(args.data + '/train_images', transform=data_vflip),
        datasets.ImageFolder(args.data + '/train_images', transform=data_color_jitter),
        datasets.ImageFolder(args.data + '/train_images', transform=data_grayscale),
    ]),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images', transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:  ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')

# Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net

model = Net()
if use_cuda and torch.cuda.is_available():
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

training_loss_values = []
validation_loss_values = []

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    CURRENT_EPOCH = checkpoint['epoch']
    training_loss_values = checkpoint['training_loss']
    validation_loss_values = checkpoint['validation_loss']


def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        if use_cuda and torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * data.size(0)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    training_loss_values.append(running_loss / len(train_loader.dataset))


def plot():
    fig = plt.figure()
    plt.plot(training_loss_values, label='training loss')
    plt.plot(validation_loss_values, label='validation loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(args.output + 'convergence_plot.png')
    plt.close(fig)


def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = Variable(data), Variable(target)
            if use_cuda and torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            validation_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    validation_loss_values.append(validation_loss)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


for epoch in range(CURRENT_EPOCH, args.epochs + 1):
    train(epoch)
    validation()
    plot()
    model_file = args.output + 'model_adam_stn.pth'
    if epoch % 2 == 0:
        model_file = args.output + 'model_all_cnn_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)  # save model for sharing
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_loss': training_loss_values,
        'validation_loss': validation_loss_values
    }, CHECKPOINT_PATH)
    print('\nSaved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file +
          '` to generate the Kaggle formatted csv file')
