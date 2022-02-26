# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------
import os

import torch
import torch.utils.data
from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils.dataset import coco
from opt import opt
from models.FastPose import createModel
from utils.eval import DataLogger, accuracy
from utils.img import flip_v, shuffleLR_v
from models.sync_batchnorm import DataParallelWithCallback


if opt.sync:
    DataParallel = DataParallelWithCallback
else:
    DataParallel = torch.nn.DataParallel


def train(train_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.train()

    train_loader_desc = tqdm(train_loader)
    for i, (inps, labels, setMask, imgset) in enumerate(train_loader_desc):
        inps = inps.cuda().requires_grad_()
        labels = labels.cuda()
        setMask = setMask.cuda()
        out = m(inps)

        loss = criterion(out.mul(setMask), labels)

        acc = accuracy(out.data.mul(setMask), labels.data, train_loader.dataset)

        accLogger.update(acc[0], inps.size(0))
        lossLogger.update(loss.item(), inps.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opt.trainIters += 1
        # Tensorboard
        writer.add_scalar(
            'Train/Loss', lossLogger.avg, opt.trainIters)
        writer.add_scalar(
            'Train/Acc', accLogger.avg, opt.trainIters)

        # TQDM
        train_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f}'.format(
                loss=lossLogger.avg,
                acc=accLogger.avg * 100)
        )

    train_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def valid(val_loader, m, criterion, optimizer, writer):
    lossLogger = DataLogger()
    accLogger = DataLogger()
    m.eval()

    val_loader_desc = tqdm(val_loader)

    for i, (inps, labels, setMask, imgset) in enumerate(val_loader_desc):
        inps = inps.cuda()
        labels = labels.cuda()
        setMask = setMask.cuda()

        with torch.no_grad():
            out = m(inps)

            loss = criterion(out.mul(setMask), labels)

            flip_out = m(flip_v(inps, cuda=True))
            flip_out = flip_v(shuffleLR_v(
                flip_out, val_loader.dataset, cuda=True), cuda=True)

            out = (flip_out.cuda() + out) / 2

        acc = accuracy(out.mul(setMask), labels, val_loader.dataset)

        lossLogger.update(loss.item(), inps.size(0))
        accLogger.update(acc[0], inps.size(0))

        opt.valIters += 1

        # Tensorboard
        writer.add_scalar(
            'Valid/Loss', lossLogger.avg, opt.valIters)
        writer.add_scalar(
            'Valid/Acc', accLogger.avg, opt.valIters)

        val_loader_desc.set_description(
            'loss: {loss:.8f} | acc: {acc:.2f}'.format(
                loss=lossLogger.avg,
                acc=accLogger.avg * 100)
        )

    val_loader_desc.close()

    return lossLogger.avg, accLogger.avg


def main():

    # Model Initialize
    m = createModel().cuda()
    if opt.loadModel:
        print('Loading Model from {}'.format(opt.loadModel))
        current_model_weight = m.state_dict()
        weight_save = torch.load(opt.loadModel)
        weight_save_changed = {}
        for k in weight_save:
            if 'conv_out.weight' in k or 'conv_out.bias' in k:
                print(k,'not used')
                continue
            weight_save_changed[k]= weight_save[k]
        current_model_weight.update(weight_save_changed)
        m.load_state_dict(current_model_weight)
        if not os.path.exists("../exp/{}/{}".format(opt.dataset, opt.expID)):
            try:
                os.mkdir("../exp/{}/{}".format(opt.dataset, opt.expID))
            except FileNotFoundError:
                os.mkdir("../exp")
                os.mkdir("../exp/{}".format(opt.dataset))
                os.mkdir("../exp/{}/{}".format(opt.dataset, opt.expID))
    else:
        print('Create new model')
        if not os.path.exists("../exp/{}/{}".format(opt.dataset, opt.expID)):
            try:
                os.mkdir("../exp/{}/{}".format(opt.dataset, opt.expID))
            except FileNotFoundError:
                os.mkdir("../exp")
                os.mkdir("../exp/{}".format(opt.dataset))
                os.mkdir("../exp/{}/{}".format(opt.dataset, opt.expID))

    criterion = torch.nn.MSELoss().cuda()

    if opt.optMethod == 'rmsprop':
        optimizer = torch.optim.RMSprop(m.parameters(), lr=opt.LR)
    elif opt.optMethod == 'adam':
        optimizer = torch.optim.Adam(m.parameters(), lr=opt.LR)
    else:
        raise Exception

    writer = SummaryWriter('.tensorboard/{}/{}'.format(opt.dataset, opt.expID))

    # Prepare Dataset
    if opt.dataset == 'coco':
        train_dataset = coco.Mscoco(train=True, img_folder=opt.img_folder_train, annot_file=opt.annot_file_train, nJoints=opt.nClasses)
        val_dataset = coco.Mscoco(train=False, img_folder=opt.img_folder_val, annot_file=opt.annot_file_val, nJoints=opt.nClasses)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.trainBatch, shuffle=True, num_workers=opt.nThreads, pin_memory=True)
    print('train batch', opt.trainBatch)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.validBatch, shuffle=False, num_workers=opt.nThreads, pin_memory=True)

    # Model Transfer
    m = DataParallel(m).cuda()

    # Start Training
    for i in range(opt.nEpochs+1):
        opt.epoch = i
        print('############# Starting Epoch {} #############'.format(opt.epoch))

        loss, acc = train(train_loader, m, criterion, optimizer, writer)

        print('Train-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=opt.epoch,
            loss=loss,
            acc=acc
        ))
        opt.acc = acc
        opt.loss = loss
        m_dev = m.module

        if (i % opt.snapshot == 0 and i != 0) or i == opt.nEpochs:
            torch.save(m_dev.state_dict(), '../exp/{}/{}/model_{}.pkl'.format(opt.dataset, opt.expID, opt.epoch))
            torch.save(opt, '../exp/{}/{}/option.pkl'.format(opt.dataset, opt.expID, opt.epoch))

        loss, acc = valid(val_loader, m, criterion, optimizer, writer)
        print('Valid-{idx:d} epoch | loss:{loss:.8f} | acc:{acc:.4f}'.format(
            idx=i,
            loss=loss,
            acc=acc
        ))

    writer.close()


if __name__ == '__main__':
    main()
