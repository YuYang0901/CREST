from enum import Enum

import numpy as np
import torch
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from utils import submodular, craig


def distribute_subset(subset, weight, ordering_time, similarity_time, pred_time, args):
    size = torch.Tensor([len(subset)]).int().cuda()
    subset_sizes = [
        torch.zeros(size.shape, dtype=torch.int32).cuda()
        for _ in range(args.world_size)
    ]
    dist.all_gather(subset_sizes, size)
    max_size = torch.max(torch.cat(subset_sizes)).item()

    subset_list = [
        torch.zeros(max_size, dtype=torch.int64).cuda() for _ in range(args.world_size)
    ]

    subset = (
        np.append(subset, [0] * (max_size - len(subset)))
        if len(subset) != max_size
        else subset
    )

    dist.all_gather(subset_list, torch.from_numpy(subset).cuda())
    subset_list = [
        subset_list[i][: subset_sizes[i].item()] for i in range(args.world_size)
    ]
    subset = torch.cat(subset_list).cpu().numpy()

    if args.weighted and args.greedy:
        weight_list = [
            torch.zeros(max_size, dtype=torch.float32).cuda()
            for _ in range(args.world_size)
        ]

        weight = (
            torch.cat(
                [
                    weight,
                    torch.zeros(max_size - len(weight), dtype=torch.float32).cuda(),
                ]
            )
            if len(weight) != max_size
            else weight
        )

        dist.all_gather(weight_list, weight)
        weight_list = [
            weight_list[i][: subset_sizes[i].item()] for i in range(args.world_size)
        ]
        weight = torch.cat(weight_list)

    reduced_times = (
        torch.Tensor([ordering_time, similarity_time, pred_time]).float().cuda()
    )
    dist.reduce(
        reduced_times, 0, op=dist.ReduceOp.MAX,
    )

    return subset, weight, reduced_times


class SubsetMode(Enum):
    GREEDY = 1
    RANDOM = 2
    CLUSTER = 3


def cluster_features(train_dir, train_num, normalize):
    data = datasets.ImageFolder(
        train_dir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    preds, labels = (
        np.reshape(data.imgs, (train_num, -1)),
        data.targets,
    )

    return preds, labels


class SubsetGenerator:
    def __init__(self, greedy, smtk):
        self.mode = self._get_mode(greedy)
        self.smtk = smtk

    def _get_mode(self, greedy):
        if not greedy:
            return SubsetMode.RANDOM
        else:
            return SubsetMode.GREEDY

    def _random_subset(self, B, idx):
        # order = np.arange(0, TRAIN_NUM)
        # order = idx  # todo
        # np.random.shuffle(order)  # todo: with replacement
        # subset, weight = order[:B], None
        rnd_idx = np.random.randint(0, len(idx), B)
        subset, weight = idx[rnd_idx], None

        return subset, weight, 0, 0, 0

    def _greedy_features(
        self,
        epoch,
        train_dataset,
        idx,
        batch_size,
        workers,
        class_num,
        targets,
        predict_function,
        predict_args,
        pred_loader=None,
    ):
        if pred_loader is None:
            idx_subset = torch.utils.data.Subset(train_dataset, indices=idx)
            pred_loader = DataLoader(
                idx_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=True,
            )  # (Note): shuffle=False

        preds, pred_time = predict_function(pred_loader, *predict_args)
        preds = preds[idx]
        print(f"Epoch [{epoch}] [Greedy], pred size: {np.shape(preds)}")
        if np.shape(preds)[-1] == class_num:
            preds -= np.eye(class_num)[targets[idx]]

        return preds, pred_time

    def generate_subset(
        self,
        preds,
        epoch,
        B,
        idx,
        targets,
        subset_printer=None,
        mode="dense",
        num_n=128,
        use_submodlib=True,
    ):
        if subset_printer is not None:
            subset_printer.print_selection(self.mode, epoch)

        fl_labels = targets[idx] - np.min(targets[idx])

        if len(fl_labels) > 50000:
            (
                subset,
                weight,
                _,
                _,
                ordering_time,
                similarity_time,
            ) = submodular.greedy_merge(preds, fl_labels, B, 5, "euclidean",)
        else:
            if use_submodlib:
                (
                    subset,
                    weight,
                    _,
                    _,
                    ordering_time,
                    similarity_time,
                ) = submodular.get_orders_and_weights(
                    B,
                    preds,
                    "euclidean",
                    y=fl_labels,
                    equal_num=True,
                    mode=mode,
                    num_n=num_n,
                )
            else:
                (
                    subset,
                    weight,
                    _,
                    _,
                    ordering_time,
                    similarity_time,
                ) = craig.get_orders_and_weights(
                    B, preds, "euclidean", y=fl_labels, equal_num=True, smtk=self.smtk
                )

            subset = np.array(idx[subset])  # (Note): idx
            weight = np.array(weight)

        return subset, weight, ordering_time, similarity_time


class WeightedSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.
    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices, weights) -> None:
        self.dataset = dataset
        self.indices = indices
        self.weights = weights

    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        target = self.dataset[self.indices[idx]][1]
        return image, target, self.weights[idx]

    def __len__(self):
        return len(self.indices)
