import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.distributed as dist
from torch import nn


import os

from ImageClassifier import ImageClassifier
    
image_path = "./data"

HOME_PATH = os.path.expanduser("~");

train_dir = HOME_PATH+"/python/apples_oranges/data/training_set/training_set"
test_dir =  HOME_PATH+"/python/apples_oranges/data/test_set/test_set"
train_dir, test_dir

import random
from PIL import Image
import glob
from pathlib import Path
from timeit import default_timer as timer;
# Set seed
random.seed(42) 

# 1. Get all image paths (* means "any combination")
image_path_list= glob.glob(f"{image_path}/*/*/*/*.jpg")

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = Path(random_image_path).parent.stem

# 4. Open image
img = Image.open(random_image_path)




import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()



import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)


# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to IMAGE_SIZE xIMAGE_SIZE 
    transforms.Resize(size=IMAGE_SIZE),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()])

from torchvision import datasets

train_dir += "/"
# Creating training set
train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=train_transform, # transforms to perform on data (images)
                                  target_transform=None) # transforms to perform on labels (if necessary)
#Creating test set
test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)

# Set image size.
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

# Create training transform with TrivialAugment
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor()])

'''
ImageFolder(root=root_dir, transform=transforms.ToTensor())
'''

loss_fn = nn.CrossEntropyLoss();


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        local_rank: int,
        world_rank: int,

    ) -> None:
        self.local_rank = local_rank
        self.global_rank = global_rank

        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = loss_fn(output,targets);
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs():
    # train_set = MyTrainDataset()  # load your dataset
    train_set = datasets.ImageFolder(train_dir, transform=train_transform)

    model = ImageClassifier()  # load your model
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(save_every: int, total_epochs: int, batch_size: int, local_rank: int, world_rank: int, snapshot_path: str):
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)

    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path, local_rank, global_rank)

    trainer.train(total_epochs)

    dist.destroy_process_group()


start_time = timer();
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument("--master_addr", type=str, required=True)
    parser.add_argument("--master_port", type=str, required=True)

    args = parser.parse_args()

    num_gpus_per_node = torch.cuda.device_count()
    print ("num_gpus_per_node = " + str(num_gpus_per_node), flush=True)

    from mpi4py import MPI
    import os
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    global_rank = rank = comm.Get_rank()
    local_rank = int(rank) % int(num_gpus_per_node) # local_rank and device are 0 when using 1 GPU per task
    backend = None
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(global_rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = str(args.master_addr)
    os.environ['MASTER_PORT'] = str(args.master_port)
    os.environ['NCCL_SOCKET_IFNAME'] = 'hsn0'

    # Path of snapshot for every single job
    snapshot_path = str(os.environ["MODEL_SNAPSHOT_PATH"])
    print(snapshot_path, " is snapshot path")
    
    import datetime;
    dist.init_process_group(
        backend="nccl",
        #init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
        init_method='env://',
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=900000)
    )

    torch.cuda.set_device(local_rank)

    main(args.save_every, args.total_epochs, args.batch_size, local_rank, global_rank,snapshot_path=snapshot_path)

end_time = timer();
print(f"Total training time: {end_time-start_time:.3f} seconds")