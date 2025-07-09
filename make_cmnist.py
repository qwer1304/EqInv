import argparse
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset
from PIL import Image
import pandas as pd
## Progress bar
from tqdm.auto import tqdm
import numpy as np
import random


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes

class ColoredMNIST(MultipleEnvironmentMNIST):

    def __init__(self, root, include_color=False):
        ENVIRONMENTS = ['p90', 'p85', 'p80', 'p75', 'm90']
        #                                 (root, environments,                dataset_transform,  input_shape,  num_classes)
        super(ColoredMNIST, self).__init__(root, [0.1, 0.15, 0.2, 0.25, 0.9], self.color_dataset, (3, 28, 28,), 2)

        self.input_shape = (3, 28, 28,)
        self.num_classes = 2
        self.N_WORKERS = 1
        self.environments = ENVIRONMENTS
        self.include_color = include_color

    def color_dataset(self, images, labels, environment):
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        
        images = torch.stack([images, images, torch.zeros_like(images)], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        if self.include_color:
            y = colors.view(-1).long() * 2 + labels.view(-1).long()
        else:
            y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

def create_labels_dir(dir, labels):
    for label in labels:
        save_dir_label = dir + str(label) + '/'
        os.makedirs(save_dir_label, exist_ok=True)

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    # datasets is a list of per-environment TensorDatasets (x,y)
    output_dir = os.path.join(args.output_dir, str(args.target_image_size) + '/')
    # torchvision downloads raw data to <root>/MNIST/raw/
    save_dir_raw = args.output_dir
    os.makedirs(save_dir_raw, exist_ok=True)
    save_dir_train = output_dir + 'train/'
    os.makedirs(save_dir_train, exist_ok=True)
    save_dir_val = output_dir + 'val/'
    os.makedirs(save_dir_val, exist_ok=True)
    save_dir_testgt = output_dir + 'testgt/'
    os.makedirs(save_dir_testgt, exist_ok=True)
    
    # datasets is a list of datasets, each one x, y
    datasets = ColoredMNIST(save_dir_raw, include_color=args.add_color)

    labels = list(range(datasets.num_classes))
    create_labels_dir(save_dir_train, labels)
    create_labels_dir(save_dir_val, labels)
    create_labels_dir(save_dir_testgt, labels)

    for d, dataset in tqdm(enumerate(datasets), leave=False, total=len(datasets)):
        for idx, (img_tensor, label) in tqdm(enumerate(dataset), desc=f"Dataset {datasets.environments[d]}", leave=False, total=len(dataset)):
            # image_tensor and label are a single example
            if d == args.test_domain:
                save_dir_domain = save_dir_testgt
            elif d == args.val_domain:
                save_dir_domain = save_dir_val
            else:
                save_dir_domain = save_dir_train
                
            save_dir_label = save_dir_domain + str(label.item()) + '/'

            # Create filename
            filename = f"{idx:05d}.jpg"
            filepath = os.path.join(save_dir_label, filename)

            if args.target_image_size is not None:
                resize = transforms.Resize((args.target_image_size, args.target_image_size))
                img_tensor = resize(img_tensor)

            # Convert to PIL image
            pil_img = transforms.ToPILImage()(img_tensor)

            # Save using PIL
            pil_img.save(filepath, "JPEG")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create CMNIST dataset')
    parser.add_argument('--output_dir', type=str, default="./data/DataSets/CMNIST/")
    parser.add_argument('--target_image_size', type=int, default=224)
    parser.add_argument('--val_domain', type=int)
    parser.add_argument('--test_domain', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--add_color', action='store_true', help='include color in label')
    args = parser.parse_args()
    
    main(args)


