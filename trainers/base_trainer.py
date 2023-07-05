import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import wandb
from datasets import IndexedDataset
from utils import GradualWarmupScheduler


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(
        self, val, n=1
    ):  # n is the number of samples in the batch, default to 1
        """Update statistics"""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BaseTrainer:
    def __init__(
        self, 
        args: argparse.Namespace,
        model: nn.Module,
        train_dataset: IndexedDataset,
        val_loader: torch.utils.data.DataLoader,
        train_weights: torch.Tensor = None,
    ):
        """
        Base trainer class
        :param args: arguments
        :param model: model to train
        :param train_dataset: training dataset
        :param val_loader: validation data loader
        :param train_weights: weights for the training data
        """

        self.args = args
        self.model = model

        # if more than one GPU is available, use DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.args.device)

        self.train_dataset = train_dataset
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        self.val_loader = val_loader
        if train_weights is not None:
            self.train_weights = train_weights
        else:
            self.train_weights = torch.ones(len(self.train_dataset))
        self.train_weights = self.train_weights.to(self.args.device)

        # the default optimizer is SGD
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=args.lr_milestones,
            last_epoch=-1,
            gamma=args.gamma,
        )

        self.lr_scheduler = GradualWarmupScheduler(
            self.optimizer,
            multiplier=1,
            total_epoch=args.warm_start_epochs,
            after_scheduler=lr_scheduler,
        )

        self.train_criterion = nn.CrossEntropyLoss(reduction="none").to(args.device)
        self.val_criterion = nn.CrossEntropyLoss().to(args.device)

        self.train_loss = AverageMeter()
        self.train_acc = AverageMeter()

        # record data loading and training time for each batch using average meter
        self.batch_data_time = AverageMeter()
        self.batch_forward_time = AverageMeter()
        self.batch_backward_time = AverageMeter()


    def train(self):
        """
        Train the model
        """

        # load checkpoint if resume is True
        if self.args.resume_from_epoch > 0:
            self._load_checkpoint(self.args.resume_from_epoch)

        for epoch in range(self.args.resume_from_epoch, self.args.epochs):
            self._train_epoch(epoch)
            self._val_epoch(epoch)

            self._log_epoch(epoch)

            if self.args.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "val_loss": self.val_loss,
                        "val_acc": self.val_acc,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    })
                
            self.lr_scheduler.step()

            if (epoch+1) % self.args.save_freq == 0:
                self._save_checkpoint(epoch)
        self._save_checkpoint()


    def _forward_and_backward(self, data, target, data_idx):
        self.optimizer.zero_grad()

        # train model with the current batch and record forward and backward time
        forward_start = time.time()
        output = self.model(data)
        forward_time = time.time() - forward_start
        self.batch_forward_time.update(forward_time)

        loss = self.train_criterion(output, target)
        loss = (loss * self.train_weights[data_idx]).mean()

        backward_start = time.time()
        loss.backward()
        self.optimizer.step()
        backward_time = time.time() - backward_start
        self.batch_backward_time.update(backward_time)

        # update training loss and accuracy
        train_acc = (output.argmax(dim=1) == target).float().mean().item()
        self.train_loss.update(loss.item(), data.size(0))
        self.train_acc.update(train_acc, data.size(0))

        return loss, train_acc

    def _train_epoch(self, epoch):
        self.model.train()
        self._reset_metrics()

        data_start = time.time()
        # use tqdm to display a smart progress bar
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), file=sys.stdout)
        for batch_idx, (data, target, data_idx) in enumerate(pbar):

            # load data to device and record data loading time
            data, target = data.to(self.args.device), target.to(self.args.device)
            data_time = time.time() - data_start
            self.batch_data_time.update(data_time)

            self.optimizer.zero_grad()

            # train model with the current batch and record forward and backward time
            loss, train_acc = self._forward_and_backward(data, target, data_idx)

            # update progress bar
            pbar.set_description(
                "Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.6f}".format(
                    epoch,
                    self.args.epochs,
                    batch_idx * self.args.batch_size + len(data),
                    len(self.train_loader.dataset),
                    100.0 * (batch_idx+1) / len(self.train_loader),
                    loss.item(),
                    train_acc,
                )
            )

            data_start = time.time()


    def _val_epoch(self, epoch):
        self.model.eval()

        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for _, (data, target, _) in enumerate(self.val_loader):
                data, target = data.cuda(), target.cuda()

                output = self.model(data)

                loss = self.val_criterion(output, target)

                val_loss += loss.item() * data.size(0)
                val_acc += (output.argmax(dim=1) == target).float().sum().item()

        val_loss /= len(self.val_loader.dataset)
        val_acc /= len(self.val_loader.dataset)

        self.val_loss = val_loss
        self.val_acc = val_acc

    def _save_checkpoint(self, epoch=None):
        if epoch is not None:
            save_path = self.args.save_dir + "/model_epoch_{}.pt".format(epoch)
        else:
            save_path = self.args.save_dir + "/model_final.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": self.train_loss.avg,
                "train_acc": self.train_acc.avg,
                "val_loss": self.val_loss,
                "val_acc": self.val_acc,
                "args": self.args,
                }, 
            save_path)
        
        self.args.logger.info("Checkpoint saved to {}".format(save_path))
        
    def _load_checkpoint(self, epoch):
        save_path = self.args.save_dir + "/model_epoch_{}.pt".format(epoch)
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_loss = checkpoint["train_loss"]
        self.train_acc = checkpoint["train_acc"]
        self.val_loss = checkpoint["val_loss"]
        self.val_acc = checkpoint["val_acc"]
        self.args = checkpoint["args"]

        self.args.logger.info("Checkpoint loaded from {}".format(save_path))


    def _log_epoch(self, epoch):
        self.args.logger.info(
            "Epoch {}:\tTrain Loss: {:.6f}\tTrain Acc: {:.6f}\tVal Loss: {:.6f}\tVal Acc: {:.6f}".format(
                epoch,
                self.train_loss.avg,
                self.train_acc.avg,
                self.val_loss,
                self.val_acc,
            )
        )
        self.args.logger.info(
            "Epoch {}:\tData Loading Time: {:.6f}\tForward Time: {:.6f}\tBackward Time: {:.6f}".format(
                epoch,
                self.batch_data_time.avg,
                self.batch_forward_time.avg,
                self.batch_backward_time.avg,
            )
        )

    def _reset_metrics(self):
        self.train_loss.reset()
        self.train_acc.reset()
        self.batch_data_time.reset()
        self.batch_forward_time.reset()
        self.batch_backward_time.reset()


    def get_model(self):
        return self.model
    
    def get_train_loss(self):
        return self.train_loss.avg
    
    def get_train_acc(self):
        return self.train_acc.avg
    
    def get_val_loss(self):
        return self.val_loss.avg
    
    def get_val_acc(self):
        return self.val_acc.avg
    
    def get_train_time(self):
        # return a dict of data loading, forward and backward time
        return {
            "data_time": self.batch_data_time.avg,
            "forward_time": self.batch_forward_time.avg,
            "backward_time": self.batch_backward_time.avg,
        }
    
