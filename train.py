import os
import argparse
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ConvLSTM import ConvLSTM
from VideoDataset import VideoDataset

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Toyota Smart Home spatial stream on ConvLSTM")
    parser.add_argument("--frames-path", default="./Data/mp4_frames/", type=str)
    parser.add_argument("--csv-path", default="./Data/Labels/cross_subject/", type=str)
    parser.add_argument("--cross-view", action="store_true")
    parser.add_argument("--frame-size", default=224, type=int)
    parser.add_argument("--sequence-length", default=16, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--gpu-number", default=0, type=int)

    # path check 

    # print args
    args = parser.parse_args()
    print(args)

    # Prepare the loader
    train = VideoDataset(frames_path=args.frames_path, csv_path=args.csv_path + "train.csv", frame_size=args.frame_size, sequence_length=args.sequence_length)
    val = VideoDataset(frames_path=args.frames_path, csv_path=os.path.join(args.csv_path, "val.csv" if args.cross_view else "test.csv"), frame_size=args.frame_size, sequence_length=args.sequence_length)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # get a device
    if torch.cuda.is_available():
        cudnn.benchmark = True
        device = torch.device(f"cuda:{args.gpu_number}")
    else:
        device = torch.device(f"cpu")

    # get a model
    model = ConvLSTM(
        backbone_name="resnet18",
        num_classes=train.num_classes,
        hidden_size=1024,
        num_layers=1,
        bidirectional=True
    ).to(device)

    # get a optimizer and scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=1, verbose=True)

    for e in range(1, args.epochs+1):
        # ================
        # training loop
        # ================
        train_top1 = AverageMeter()
        train_top3 = AverageMeter()
        train_loss = AverageMeter()
        model.train()
        for i, (datas, labels) in enumerate(train_loader):
            datas = datas.to(device); labels = labels.to(device)

            # get a prediction scores and clculate loss
            pred = model(datas)
            loss = criterion(pred, labels)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate loss and acc
            prec1, prec3 = accuracy(pred.data, labels, topk=(1, 3))
            train_loss.update(loss.item(), datas.size(0))
            train_top1.update(prec1.item(), datas.size(0))
            train_top3.update(prec3.item(), datas.size(0))

            # print a message
            print("[train] epochs: {}/{} batch: {}/{} loss: {:.4f}({:.4f}) proc@1: {:.4f}({:.4f}) proc@3: {:.4f}({:.4f})".format(
                e, args.epochs, i, len(train_loader), train_loss.val, train_loss.avg,
                train_top1.val, train_top1.avg, train_top3.val, train_top3.avg,
            ))

        # ================
        # validation loop
        # ================
        val_top1 = AverageMeter()
        val_top3 = AverageMeter()
        val_loss = AverageMeter()
        model.eval()
        for i, (datas, labels) in enumerate(val_loader):
            datas = datas.to(device); labels = labels.to(device)

            # get a prediction scores and clculate loss
            pred = model(datas)
            loss = criterion(pred, labels)

            # calculate loss and acc
            prec1, prec3 = accuracy(pred.data, labels, topk=(1, 3))
            val_loss.update(loss.item(), datas.size(0))
            val_top1.update(prec1.item(), datas.size(0))
            val_top3.update(prec3.item(), datas.size(0))

            print("[val] epochs: {}/{} batch: {}/{} loss: {:.4f}({:.4f}) proc@1: {:.4f}({:.4f}) proc@3: {:.4f}({:.4f})".format(
                e, args.epochs, i, len(val_loader), val_loss.val, val_loss.avg,
                val_top1.val, val_top1.avg, val_top3.val, val_top3.avg,
            ))