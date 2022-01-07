#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import os
import logging
import sys


# handles  https://discuss.pytorch.org/t/oserror-image-file-is-truncated-150-bytes-not-processed/64445
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, testloader):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("BEGIN TESTING")
    model.to("cpu")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)

    print(f"Test Loss: {test_loss:.4f}, Accuracy: {correct}/{len(testloader.dataset)} ({100.0 * correct / len(testloader.dataset):.0f}%)")
#     logger.info(f"Test Loss: {test_loss:.4f}, Accuracy: {100.0 * correct / len(testloader.dataset):.0f}%")
    logger.info("COMPLETE TESTING")

    
def train(model, trainloader, validloader, criterion, optimizer, epochs, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("BEGIN TRAINING")
    for i in range(epochs):
        model.train()
        train_loss = 0
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in validloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
        
        print(f"Epoch {i}: Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
#         logger.info(f"Epoch {i}: Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
    logger.info("COMPLETE TRAINING")

        
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    logger.info("INITIALIZE MODEL FOR FINETUNING")
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        
    inf = model.fc.in_features
    model.fc = nn.Linear(inf, 133)
    
    return model
           

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    logger.info("CREATING DATA LOADERS")
    traindata = os.path.join(data, "train")
    validdata = os.path.join(data, "valid")
    testdata = os.path.join(data, "test")
    
    
    transform = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406),
                                                          (0.229, 0.224, 0.225))
                                   ])
    
    trainset = torchvision.datasets.ImageFolder(traindata, transform=transform)
    validset = torchvision.datasets.ImageFolder(validdata, transform=transform)
    testset = torchvision.datasets.ImageFolder(testdata, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    
    return trainloader, validloader, testloader

def save_model(model, model_dir):
    logger.info("SAVING MODEL WEIGHTS")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    # set up device for gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"DEVICE: {device}")
    model=net()
    model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    trainloader, validloader, testloader = create_data_loaders(args.data_dir, args.batch_size)
    train(model, trainloader, validloader, criterion, optimizer, args.epochs, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, testloader)
    
    '''
    TODO: Save the trained model
    '''
    save_model(model, args.model_dir)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument( "--momentum", type=float, default=0.9, metavar="M", help="momentum (default: 0.8)")
    
     # Container environment
#     parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
#     parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
#     parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args=parser.parse_args()
    
    main(args)
