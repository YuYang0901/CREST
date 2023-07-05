import logging
import os
import time
from warnings import simplefilter

import numpy as np
import torch

from utils import get_args

# Ignore future warnings from numpy
simplefilter(action="ignore", category=FutureWarning)
np.seterr(all="ignore")

args = get_args()

if len(args.gpu) > 0 and ("CUDA_VISIBLE_DEVICES" not in os.environ): 
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
    device_str = ",".join(map(str, args.gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = device_str
    print("Using GPU: {}.".format(os.environ["CUDA_VISIBLE_DEVICES"]))

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision

from datasets import IndexedDataset
from models import *

# Use CUDA if available and set random seed for reproducibility
if torch.cuda.is_available():
    args.device = "cuda"
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    args.device = "cpu"
    torch.manual_seed(args.seed)

if args.use_wandb:
    import wandb
    wandb.init(project="crest", config=args, name=args.save_dir.split('/')[-1])

# Set up logging and output locations
logger = logging.getLogger(args.save_dir.split('/')[-1] + time.strftime("-%Y-%m-%d-%H-%M-%S"))
os.makedirs(args.save_dir, exist_ok=True)

logging.basicConfig(
    filename=f"{args.save_dir}/output.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# define a Handler which writes INFO messages or higher to the sys.stderr
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)
args.logger = logger

# Print arguments
args.logger.info("Arguments: {}".format(args))
args.logger.info("Time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))

def main(args):
    train_dataset = IndexedDataset(args, train=True, train_transform=True)
    args.train_size = len(train_dataset)
    val_loader = torch.utils.data.DataLoader(
        IndexedDataset(args, train=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    if args.arch == 'resnet20':
        model = ResNet20(num_classes=args.num_classes)
    elif args.arch == 'resnet18':
        model = ResNet18(num_classes=args.num_classes)
    elif args.arch == 'resnet50':
        model = torchvision.models.resnet50(num_classes=args.num_classes)
    else:
        raise NotImplementedError(f"Architecture {args.arch} not implemented.")

    if args.selection_method == "none":
        from trainers import BaseTrainer
        trainer = BaseTrainer(
            args,
            model,
            train_dataset,
            val_loader,
        )
    elif args.selection_method == "random":
        from trainers import RandomTrainer
        trainer = RandomTrainer(
            args,
            model,
            train_dataset,
            val_loader,
        )
    elif args.selection_method == "crest":
        from trainers import CRESTTrainer
        trainer = CRESTTrainer(
            args,
            model,
            train_dataset,
            val_loader,
        )
    else:
        raise NotImplementedError(f"Selection method {args.selection_method} not implemented.")
    
    trainer.train()

if __name__ == "__main__":
    main(args)
