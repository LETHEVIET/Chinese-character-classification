# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

from ctc_decoder import ctc_decode

from cangjie_dataset import get_cangjie_val_dataloader, get_cangjie_training_dataloader, Cangjie_Class

import logging
import editdistance

logging.basicConfig(
    level=logging.INFO,  # Set the root logger level (DEBUG logs everything)
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    handlers=[
        # logging.FileHandler("my_app.log"),  # Log to a file
        logging.StreamHandler()             # Log to console 
    ]
)


log = logging.getLogger(__name__)
log.info("Hello, world")

def train(epoch, criterion):
    start = time.time()
    net.train()
    for batch_index, data in enumerate(cangjie_training_loader):

        if args.gpu:
            images, targets, target_lengths = [d.cuda() for d in data]
        else:
            images, targets, target_lengths = [d for d in data]

        optimizer.zero_grad()
        logits = net(images)

        logits = logits.permute(1, 0, 2)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        target_lengths = torch.flatten(target_lengths)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5) # gradient clipping with 5
        optimizer.step()

        n_iter = (epoch - 1) * len(cangjie_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cangjie_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

def calculate_loss(predict, ground_truth):
    T = ground_truth.shape[1]
    N = ground_truth.shape[0]

    target = ground_truth.type(torch.LongTensor)
    inputs = predict.log_softmax(2)

    target_lengths = []

    for img in ground_truth.tolist():
        i = 0
        while i < 5 and img[i] < 26.0:
            i += 1
        target_lengths.append(i)
    target_lengths = torch.Tensor(target_lengths).type(torch.LongTensor)

    inputs = inputs.permute(1, 0, 2) #.contiguous()
    inputs_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    log.info(f"inputs.shape: {inputs.shape}, length shape: {inputs_lengths.shape}")
    log.info(f"target.shape: {target.shape}, length shape: {target_lengths.shape}")
    loss = ctc_loss(inputs, target, inputs_lengths, target_lengths)
    log.info(f"loss: {loss.item()}")
    return loss

@torch.no_grad()
def eval_training(criterion, epoch=0, decode_method='beam_search', beam_size=10, tb=True):

    tot_count = 0
    tot_loss = 0
    tot_correct = 0
    wrong_cases = []

    start = time.time()
    net.eval()

    for i, data in enumerate(cangjie_val_loader):
        device = 'cuda' if next(net.parameters()).is_cuda else 'cpu'

        images, targets, target_lengths = [d.to(device) for d in data]

        logits = net(images)
        logits = logits.permute(1, 0, 2)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        target_lengths = torch.flatten(target_lengths)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size)
        reals = targets.cpu().numpy().tolist()
        target_lengths = target_lengths.cpu().numpy().tolist()

        tot_count += batch_size
        tot_loss += loss.item()
        target_length_counter = 0
        for pred, target_length in zip(preds, target_lengths):
            real = reals[target_length_counter:target_length_counter + target_length]
            target_length_counter += target_length
            if pred == real:
                tot_correct += 1
            else:
                wrong_cases.append((real, pred))

    with open("val.log", "w") as f:
        for case in wrong_cases:
            f.write(f"{case}\n")

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Validation set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        tot_loss / tot_count,
        tot_correct / tot_count,
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Validation/Average loss', tot_loss / tot_count, epoch)
        writer.add_scalar('Validation/Lev Accuracy', tot_correct / tot_count, epoch)

    return tot_correct / tot_count

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    net = get_network(args)

    cangjie_class = Cangjie_Class("etl_952_singlechar_size_64/952_labels.txt")

    log.info("Loading training data... ")
    cangjie_training_loader = get_cangjie_training_dataloader(
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    log.info("Training data loaded!")

    log.info("Loading validation data... ")
    cangjie_val_loader = get_cangjie_val_dataloader(
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    log.info("Validation dataset loaded!")

    # loss_function = nn.CrossEntropyLoss()
    ctc_loss = nn.CTCLoss(reduction='sum', zero_infinity=True)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cangjie_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 1, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(ctc_loss, tb=False) #(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch, ctc_loss)
        acc = eval_training(ctc_loss, epoch)

        #start to save best performance model after learning rate decay to 0.01
        # if epoch > settings.MILESTONES[1] and best_acc < acc:
        if best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
