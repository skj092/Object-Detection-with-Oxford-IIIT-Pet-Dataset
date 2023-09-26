from fastai.vision.all import *
from torchvision.transforms import v2 as T
import torch
from torch.utils.data import random_split, DataLoader, Subset
from dataset import CustomObjectDetectionDataset
from models import model
from engine import train_one_epoch, evaluate


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    # download the dataset
    path = untar_data(URLs.PETS)
    Path.BASE_PATH  = path
    #print(path.ls())

    # dataset and dataloader
    ds = CustomObjectDetectionDataset(path, transform=get_transform(train=True))

    # Define the dataset size and the desired split ratio
    dataset_size = len(ds)
    validation_split = 0.2  # 20% of the data will be used for validation

    # Calculate the sizes of the training and validation sets
    valid_size = int(validation_split * dataset_size)
    train_size = dataset_size - valid_size

    # Use random_split to split the dataset into train and validation subsets
    train_subset, valid_subset = random_split(ds, [train_size, valid_size])

    # Create DataLoader objects for train and validation sets
    batch_size = 4  # You can adjust this to your preference
    train_loader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)

    # Optionally, you can create a complete dataset for train and validation
    train_ds = Subset(ds, train_subset.indices)
    valid_ds = Subset(ds, valid_subset.indices)

    #print(len(train_ds), len(valid_ds))
    xb, yb = next(iter(train_loader))
    #print(xb[0].shape)
    #print(yb)

    # testing on one batch
    #images, targets = next(iter(train_loader))
    #images = list(image for image in images)
    #target = [{k: v for k, v in t.items()} for t in targets]
    #model.to(device)
    #output = model(images, target)  # Returns losses and detections
    #print(output)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )

    # let's train it for 5 epochs
    num_epochs = 1

    evaluate(model, valid_loader, device=device)

   # for epoch in range(num_epochs):
   #     # train for one epoch, printing every 10 iterations
   #     train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
   #     # update the learning rate
   #     lr_scheduler.step()
   #     # evaluate on the test dataset
   #     evaluate(model, valid_loader, device=device)

   # print("That's it!")
