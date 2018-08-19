import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from definitions import ROOT_DIR
from utils.helpers import device


def get_train_valid_loaders(dataset, valid_size=0.1, shuffle=True, seed=1234, batch_size=32):
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)

    valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    return train_loader, valid_loader


def load_data(dirname=ROOT_DIR + '/data/'):
    """
    Load our data, returns a 3 Dataset objects.
    :param dirname:
    :return:
    """
    train_dataset = SketchDataset()
    valid_dataset = SketchDataset()
    test_dataset = SketchDataset()

    label_mapping = []
    for file in os.listdir(dirname):
        if file.endswith(".npz"):
            data = np.load(dirname + '/' + file, encoding='bytes')
            label = os.path.splitext(file)[0]
            label_mapping.append(label)

            index = len(label_mapping) - 1
            train_data = data['train']
            train_labels = np.array([index] * len(train_data))
            train_dataset.push(train_data, train_labels)

            valid_data = data['valid']
            valid_labels = np.array([index] * len(valid_data))
            valid_dataset.push(valid_data, valid_labels)

            test_data = data['test']
            test_labels = np.array([index] * len(test_data))
            test_dataset.push(test_data, test_labels)

    train_dataset.add_mapping(label_mapping)
    valid_dataset.add_mapping(label_mapping)
    test_dataset.add_mapping(label_mapping)

    return train_dataset, valid_dataset, test_dataset


class SketchDataset(Dataset):
    def __init__(self):
        super(SketchDataset, self).__init__()

        self.mapping = []
        self.data = []
        self.labels = []

    def add_mapping(self, mapping):
        self.mapping = mapping

    def push(self, data, labels):
        self.data = np.concatenate((self.data, data))
        self.labels = np.concatenate((self.labels, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        label = self.labels[idx]

        return torch.tensor(example, dtype=torch.long, device=device), \
            torch.tensor(label, dtype=torch.long, device=device)

if __name__ == '__main__':
    from utils.draw import draw_strokes
    train_dataset, valid_dataset, test_dataset = load_data()
    data, label = train_dataset[0]
    draw_strokes(data)
