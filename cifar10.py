from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
import utils.utils as util
import utils.quantization as q

import numpy as np
import os, time, sys
import copy
import argparse

#########################
# supported model candidates

candidates = [
                'binput-pg', 
             ]
#########################


#----------------------------
# Argument parser.
#----------------------------
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--model_id', '-id', type=int, default=0)
parser.add_argument('--gtarget', '-g', type=float, default=0.0)
parser.add_argument('--init_lr', '-l', type=float, default=1e-3)
parser.add_argument('--batch_size', '-b', type=int, default=128)
parser.add_argument('--num_epoch', '-e', type=int, default=250)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5)
parser.add_argument('--last_epoch', '-last', type=int, default=-1)
parser.add_argument('--finetune', '-f', action='store_true', help='finetune the model')
parser.add_argument('--save', '-s', action='store_true', help='save the model')
parser.add_argument('--test', '-t', action='store_true', help='test only')
parser.add_argument('--resume', '-r', type=str, default=None,
                    help='path of the model checkpoint for resuming training')
parser.add_argument('--data_dir', '-d', type=str, default='/tmp/cifar10_data',
                    help='path to the dataset directory')
parser.add_argument('--which_gpus', '-gpu', type=str, default='0', help='which gpus to use')

args = parser.parse_args()
_ARCH = candidates[args.model_id]
drop_last = True if 'binput' in _ARCH else False


#----------------------------
# Load the CIFAR-10 dataset.
#----------------------------

def load_cifar10():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        ]
    transform_test_list = [transforms.ToTensor()]

    if 'binput' not in _ARCH:
        transform_train_list.append(normalize)
        transform_test_list.append(normalize)

    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose(transform_test_list)

    # pin_memory=True makes transfering data from host to GPU faster
    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2, 
                                              pin_memory=True, drop_last=drop_last)

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2, 
                                             pin_memory=True, drop_last=drop_last)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


#----------------------------
# Define the model.
#----------------------------

def generate_model(model_arch):
    if 'binput-pg' in model_arch:
        import model.binput_resnet20_pg as m
        return m.resnet20(batch_size=args.batch_size, num_gpus=torch.cuda.device_count())
    else:
        raise NotImplementedError("Model architecture is not supported.")


#----------------------------
# Train the network.
#----------------------------

def train_model(trainloader, testloader, net, 
                optimizer, scheduler, start_epoch, device):
    # define the loss function
    criterion = (nn.CrossEntropyLoss().cuda() 
                if torch.cuda.is_available() else nn.CrossEntropyLoss())

    best_acc = 0.0
    best_model = copy.deepcopy(net.state_dict())

    for epoch in range(start_epoch, args.num_epoch): # loop over the dataset multiple times

        # set printing functions
        batch_time = util.AverageMeter('Time/batch', ':.2f')
        losses = util.AverageMeter('Loss', ':6.2f')
        top1 = util.AverageMeter('Acc', ':6.2f')
        progress = util.ProgressMeter(
                        len(trainloader),
                        [losses, top1, batch_time],
                        prefix="Epoch: [{}]".format(epoch+1)
                        )

        # switch the model to the training mode
        net.train()

        print('current learning rate = {}'.format(optimizer.param_groups[0]['lr']))
        
        # each epoch
        end = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            if 'pg' in _ARCH:
                for name, param in net.named_parameters():
                    if 'threshold' in name:
                        loss += (0.00001 * 0.5 *
                                 torch.norm(param-args.gtarget) *
                                 torch.norm(param-args.gtarget))
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            _, batch_predicted = torch.max(outputs.data, 1)
            batch_accu = 100.0 * (batch_predicted == labels).sum().item() / labels.size(0)
            losses.update(loss.item(), labels.size(0))
            top1.update(batch_accu, labels.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 99:    
                # print statistics every 100 mini-batches each epoch
                progress.display(i) # i = batch id in the epoch

        # update the learning rate
        scheduler.step()

        # print test accuracy every few epochs
        if epoch % 1 == 0:
            print('epoch {}'.format(epoch+1))
            epoch_acc = test_accu(testloader, net, device)
            if 'pg' in _ARCH:
                sparsity(testloader, net, device)
            if epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(net.state_dict())
            print("The best test accuracy so far: {:.1f}".format(best_acc))

            # save the model if required
            if args.save:
                print("Saving the trained model and states.")
                this_file_path = os.path.dirname(os.path.abspath(__file__))
                save_folder = os.path.join(this_file_path, 'save_CIFAR10_model')
                util.save_models(best_model, save_folder,
                        suffix=_ARCH+'-finetune' if args.finetune else _ARCH)
                """
                states = {'epoch':epoch+1, 
                          'optimizer':optimizer.state_dict(), 
                          'scheduler':scheduler.state_dict()}
                util.save_states(states, save_folder, suffix=_ARCH)
                """

    print('Finished Training')


#----------------------------
# Test accuracy.
#----------------------------

def test_accu(testloader, net, device):
    correct = 0
    total = 0
    # switch the model to the evaluation mode
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print('Accuracy of the network on the 10000 test images: %.1f %%' % accuracy)
    return accuracy


#----------------------------
# Report sparsity in PG
#----------------------------

def sparsity(testloader, net, device):
    num_out, num_high = [], []

    def _report_sparsity(m):
        classname = m.__class__.__name__
        if isinstance(m, q.PGBinaryConv2d):
            num_out.append(m.num_out)
            num_high.append(m.num_high)

    net.eval()
    # initialize cnt_out, cnt_high
    net.apply(_report_sparsity)
    cnt_out = np.zeros(len(num_out))
    cnt_high = np.zeros(len(num_high))
    num_out, num_high = [], []

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            """ calculate statistics per PG layer """
            net.apply(_report_sparsity)
            cnt_out += np.array(num_out)
            cnt_high += np.array(num_high)
            num_out = []
            num_high = []
    print('Sparsity of the update phase: %.1f %%' %
          (100.0-np.sum(cnt_high)*1.0/np.sum(cnt_out)*100.0))


#----------------------------
# Remove the saved placeholder
#----------------------------

def remove_placeholder(state_dict):
    from collections import OrderedDict
    temp_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if 'encoder.placeholder' in key:
            pass
        else:
            temp_state_dict[key] = value
    return temp_state_dict


#----------------------------
# Main function.
#----------------------------

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Available GPUs: {}".format(torch.cuda.device_count()))

    print("Create {} model.".format(_ARCH))
    net = generate_model(_ARCH)

    if torch.cuda.device_count() > 1:
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        print("Activate multi GPU support.")
        net = nn.DataParallel(net)
    net.to(device)

    #------------------
    # Load model params
    #------------------
    if args.resume is not None:
        model_path = args.resume
        if os.path.exists(model_path):
            print("@ Load trained model from {}.".format(model_path))
            state_dict = torch.load(model_path)
            state_dict = remove_placeholder(state_dict)
            net.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Model not found.")

    #-----------------
    # Prepare Data
    #-----------------
    print("Loading the data.")
    trainloader, testloader, classes = load_cifar10()

    #-----------------
    # Test
    #-----------------
    if args.test:
        print("Mode: Test only.")
        test_accu(testloader, net, device)
        if 'pg' in _ARCH:
            sparsity(testloader, net, device)

    #-----------------
    # Finetune
    #-----------------
    elif args.finetune:
        print("num epochs = {}".format(args.num_epoch))
        initial_lr = args.init_lr
        print("init lr = {}".format(initial_lr))
        optimizer = optim.Adam(net.parameters(),
                          lr = initial_lr,
                          weight_decay=0.)
        lr_decay_milestones = [100, 150, 200]
        print("milestones = {}".format(lr_decay_milestones))
        scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=lr_decay_milestones,
                            gamma=0.1,
                            last_epoch=args.last_epoch)
        start_epoch=0
        print("Start finetuning.")
        train_model(trainloader, testloader, net,
                    optimizer, scheduler, start_epoch, device)
        test_accu(testloader, net, device)

    #-----------------
    # Train
    #-----------------
    else:
        print("num epochs = {}".format(args.num_epoch))
        #-----------
        # Optimizer
        #-----------
        initial_lr = args.init_lr
        optimizer = optim.Adam(net.parameters(),
                          lr = initial_lr,
                          weight_decay=args.weight_decay)

        #-----------
        # Scheduler
        #-----------
        print("Use linear learning rate decay.")
        lambda1 = lambda epoch : (1.0-epoch/args.num_epoch) # linear decay
        #lambda1 = lambda epoch : (0.7**epoch) # exponential decay
        scheduler = optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lr_lambda=lambda1,
                            last_epoch=args.last_epoch)

        start_epoch = 0
        print("Start training.")
        train_model(trainloader, testloader, net, 
                    optimizer, scheduler, start_epoch, device)
        test_accu(testloader, net, device)

if __name__ == "__main__":
    main()

