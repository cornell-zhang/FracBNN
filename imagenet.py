from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import utils.utils as util
import utils.quantization as q

import numpy as np
import os, sys, time
import warnings
import argparse
import copy

# ignore "corrupt EXIF data" warnings in the console
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


#########################
# supported model candidates

candidates = [
                'binput-pg-quant-shortcut',
             ]
#########################


#----------------------------
# Argument parser.
#----------------------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model_id', '-id', type=int, default=0)
parser.add_argument('--gtarget', '-g', type=float, default=0.0)
parser.add_argument('--init_lr', '-lr', type=float, default=5e-4)
parser.add_argument('--batch_size', '-b', type=int, default=256)
parser.add_argument('--num_epoch', '-e', type=int, default=120)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5)
parser.add_argument('--last_epoch', '-last', type=int, default=-1)
parser.add_argument('--finetune', '-f', action='store_true', help='finetune the model')
parser.add_argument('--save', '-s', action='store_true', help='save the model')
parser.add_argument('--test', '-t', action='store_true', help='test only')
parser.add_argument('--resume', '-r', type=str, default=None,
                    help='path of the model for resuming training')
parser.add_argument('--load_states', '-l', type=str, default=None,
                    help='path of states to the optimizer and scheduler')
parser.add_argument('--data_dir', '-d', type=str, 
                    default='/temp/datasets/imagenet-pytorch/',
                    help='path to the dataset directory')
parser.add_argument('--which_gpus', '-gpu', type=str, default='0', help='which gpus to use')

args = parser.parse_args()
_ARCH = candidates[args.model_id]
drop_last = True if 'binput' in _ARCH else False


#----------------------------
# Load the ImageNet dataset.
#----------------------------

def load_dataset():
    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    crop_scale = 0.08
    lighting_param = 0.1
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
        util.Lighting(lighting_param),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transform=train_transforms
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=max(8, 2*torch.cuda.device_count()), 
        pin_memory=True, drop_last=drop_last
    )

    valloader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=max(8, 2*torch.cuda.device_count()), 
        pin_memory=True, drop_last=drop_last
    )

    return trainloader, valloader


#----------------------------
# Define the model.
#----------------------------

def generate_model(model_arch):
    if 'binput-pg-quant-shortcut' in model_arch:
        import model.fracbnn_imagenet as m
        return m.ReActNet(
                batch_size=args.batch_size,
                num_gpus=torch.cuda.device_count()
               )
    else:
        raise NotImplementedError("Model architecture is not supported.")


#----------------------------
# Train the network.
#----------------------------

def train_model(trainloader, testloader, net, 
                optimizer, scheduler, start_epoch, 
                num_epoch, device):
    # define the loss function
    criterion = (nn.KLDivLoss(reduction='batchmean').cuda() 
                if torch.cuda.is_available() else nn.KLDivLoss(reduction='batchmean'))

    best_acc = 0.
    best_model = copy.deepcopy(net.state_dict())
    states = {'epoch':start_epoch,
              'optimizer':optimizer.state_dict(),
              'scheduler':scheduler.state_dict()}

    for epoch in range(start_epoch, num_epoch):

        # set printing functions
        batch_time = util.AverageMeter('Time/batch', ':.2f')
        losses = util.AverageMeter('Loss', ':6.2f')
        top1 = util.AverageMeter('Acc@1', ':6.2f')
        top5 = util.AverageMeter('Acc@5', ':6.2f')
        progress = util.ProgressMeter(
                        len(trainloader),
                        [losses, top1, top5, batch_time],
                        prefix="Epoch: [{}]".format(epoch+1)
                        )

        # switch the model to the training mode
        net.train()

        print('current learning rate = {}'.format(optimizer.param_groups[0]['lr']))

        # each epoch
        end = time.time()
        for i, data in enumerate(trainloader, 0):

            # get the inputs; data is a tuple of (inputs, labels)
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, lessons = net(inputs)
            loss = criterion(outputs.log_softmax(dim=1), lessons.softmax(dim=1))
            if 'pg' in _ARCH:
                for name, param in net.named_parameters():
                    if 'threshold' in name:
                        loss += (0.00001 * 0.5 *
                                 torch.norm(param-args.gtarget) *
                                 torch.norm(param-args.gtarget))
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 99:    
                # print statistics every 100 mini-batches each epoch
                progress.display(i) # i = batch id in the epoch

        # update the learning rate every epoch
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
                states = {'epoch':epoch+1,
                          'optimizer':optimizer.state_dict(),
                          'scheduler':scheduler.state_dict()}
            print("Best test accuracy so far: {:.1f}".format(best_acc))

            # save the model if required
            if args.save:
                print("Saving the trained model.")
                this_file_path = os.path.dirname(os.path.abspath(__file__))
                save_folder = os.path.join(this_file_path, 'save_ImageNet_model')
                util.save_models(best_model, save_folder, 
                                 suffix=_ARCH+'-finetune' if args.finetune else _ARCH)
                util.save_states(states, save_folder, 
                                 suffix=_ARCH+'-finetune' if args.finetune else _ARCH)

    print('Finished Training')


#----------------------------
# Test accuracy.
#----------------------------

def accuracy(outputs, labels, topk=(1,)):
    '''
    Computes the accuracy over the k top predictions for 
    the specified values of k
    '''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def test_accu(testloader, net, device):
    top1 = util.AverageMeter('Acc@1', ':6.2f')
    top5 = util.AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    net.eval()

    with torch.no_grad():
        start = time.time()
        for i, data in enumerate(testloader, 0):
            images, labels = data[0].to(device), data[1].to(device)

            # compute output
            outputs, _ = net(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

        # measure elapsed time
        elapsed_time = time.time() - start

        # print statistics
        print(' * Acc@1 {top1.avg:.1f} Acc@5 {top5.avg:.1f} Elapsed time = {clock:.1f} sec'
              .format(top1=top1, top5=top5, clock=elapsed_time))

    return top1.avg


#----------------------------
# Report sparsity in PG
#----------------------------

def sparsity(testloader, net, device):
    num_out, num_high = [], []

    def _report_sparsity(m):
        classname = m.__class__.__name__
        if isinstance(m, q.PGBinaryConv2d):
            num_out.append(m.num_out.item())
            num_high.append(m.num_high.item())

    net.eval()
    # initialize cnt_out, cnt_high
    net.apply(_report_sparsity)
    cnt_out = np.zeros(len(num_out))
    cnt_high = np.zeros(len(num_high))
    num_out, num_high = [], []

    batch_cnt = 50
    with torch.no_grad():
        start = time.time()
        for data in testloader:
            batch_cnt -= 1
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            """ calculate statistics per PG layer """
            net.apply(_report_sparsity)
            cnt_out += np.array(num_out)
            cnt_high += np.array(num_high)
            num_out = []
            num_high = []
            if batch_cnt == 0:
                break
        # measure elapsed time
        elapsed_time = time.time() - start
    print('Sparsity of the update phase: {:.1f} %  Elapsed time = {clock:.1f} sec'.format(
          (100.0-np.sum(cnt_high)*1.0/np.sum(cnt_out)*100.0), clock=elapsed_time))


#----------------------------
# Remove the saved placeholder
#----------------------------

def remove_placeholder(state_dict):
    from collections import OrderedDict
    temp_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if 'encoder.placeholder' in key:
            pass
        elif 'teacher' in key:
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
            state_dict = torch.load(model_path, map_location=device)
            state_dict = remove_placeholder(state_dict)
            net.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Model not found.")

    #-----------------
    # Prepare Data
    #-----------------
    print("Loading the data.")
    trainloader, testloader = load_dataset()

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

        print("Use linear learning rate decay.")
        lambda1 = lambda epoch : (1.0-epoch/args.num_epoch) # linear decay
        #print("Use exponential learning rate decay. Rate=0.7")
        #lambda1 = lambda epoch : (0.7**epoch) # exponential decay
        scheduler = optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lr_lambda=lambda1,
                            last_epoch=args.last_epoch)

        start_epoch=0
        if args.load_states is not None:
            states_path = args.load_states
            if os.path.exists(states_path):
                print("@ Load training states from {}.".format(states_path))
                states = torch.load(states_path)
                start_epoch = states['epoch']
                optimizer.load_state_dict(states['optimizer'])
                scheduler.load_state_dict(states['scheduler'])
            else:
                raise ValueError("Saved states not found.")

        print("Start finetuning.")
        train_model(trainloader, testloader, net,
                    optimizer, scheduler, start_epoch, 
                    args.num_epoch, device)
        _ = test_accu(testloader, net, device)

    #-----------------
    # Train
    #-----------------
    else:
        print("num epochs = {}".format(args.num_epoch))
        initial_lr = args.init_lr
        print("init lr = {}".format(initial_lr))
        optimizer = optim.Adam(net.parameters(),
                          lr = initial_lr,
                          weight_decay=args.weight_decay)

        # define the shceduler
        print("Use linear learning rate decay.")
        lambda1 = lambda epoch : (1.0-epoch/args.num_epoch)
        scheduler = optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lr_lambda=lambda1,
                            last_epoch=args.last_epoch)

        start_epoch = 0
        # load optimizer and scheduler states if resuming training
        if args.load_states is not None:
            states_path = args.load_states
            if os.path.exists(states_path):
                print("@ Load training states from {}.".format(states_path))
                states = torch.load(states_path)
                start_epoch = states['epoch']
                optimizer.load_state_dict(states['optimizer'])
                scheduler.load_state_dict(states['scheduler'])
            else:
                raise ValueError("Saved states not found.")

        print("Start training.")
        train_model(trainloader, testloader, net,
                    optimizer, scheduler, start_epoch,
                    args.num_epoch, device)
        _ = test_accu(testloader, net, device)


if __name__ == "__main__":
    main()

