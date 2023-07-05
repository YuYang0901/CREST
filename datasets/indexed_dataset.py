import torchdatasets as td
from datasets.datasets import get_dataset


class IndexedDataset(td.Dataset):
    def __init__(self, args, train=True, train_transform=False):
        super().__init__()
        self.dataset = get_dataset(args, train=train, train_transform=train_transform)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

    def clean(self):
        self._cachers = []