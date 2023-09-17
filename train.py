import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.functional as F
from torch import optim
from torchvision import datasets, transforms, models
import torchvision.models as models
from collections import OrderedDict
import json

with open('cat_to_name.json') as label_file:
    cat_to_name = json.load(label_file, strict=False)

#transforms
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(250),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(250),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform = test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size = 32, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size = 32)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 32 )

def classifier(model, hidden_layers, dropout):
    if model == 'vgg19':
        model = models.vgg19(pretrained = True)
        input_size = 25088
    elif model == 'resnet18':
        model = models.resnet18(pretrained = True)
        input_size = 1024
    elif model == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
        
    for param in model.parameters():
        param.requires_grad = False
    torch.cuda.empty_cache()
    
    classifier = nn.Sequential(OrderedDict([
        ('dropout': nn.Dropout(dropout)),
        ('fc1': nn.Linear(input_size, hidden_layers)),
        ('relu': nn.ReLU()),
        ('fc2':nn.Linear(hidden_layers, 256)),
        ('output': nn.Linear(256, 102)),                                              
        ('softmax':nn.LogSoftmax(dim=1))]))
    model.classifier = classifier
    return model
                               
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str. help = 'Choose a model')
parser.add_argument('--hidden_layers', type=int, help= 'Number of hidden layers')
parser.add_argument('--dropout', type=float, default=0.5, help= 'Dropout')
parser.add_argument('--data_dir', type=str, default= "./flowers")
parser.add_argument('--checkpoint', type=str, default="./checkpoint.pth")
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=5)
args = parser.parse_args()
#variables
model_arch = args.model
hidden_layers = args.hidden_layers
dropout = args.dropout
data_dir = args.data_dir
checkpoint = args.checkpoint
learning_rate = args.learning_rate
epochs = args.epochs
# model, criterion and optimizer
model = classifier(model_arch, hidden_layers, dropout)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

# training the model

def training(model, trainloader, criterion, epochs):
    running_loss = 0
    print_every = 10
    steps = 0
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    for e in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            steps+=1
            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            if steps % print_every == 0:
                valid_loss = 0
                valid_accuracy = 0
                batch_loss = 0
                model.eval()
                with torch.no_grad():  
                    # calculate test loss and accuracy    
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        valid_loss += batch_loss.item()
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch: {e+1}/{epochs}.."
                  f"Training Loss: {running_loss/print_every:.3f}.."
                  f"Validation Loss: {valid_loss/len(validloader):.3f}.."
                  f"Validation Accuracy: {(valid_accuracy/len(validloader))*100:.2f}%..")
                running_loss = 0
                model.train()
            
#training model
training_model(model, trainloader, criterion, optimizer, epochs)

#function for testing model
def test_model(testloader):
    test_accuracy = 0
    total_labels = 0
    correct_pred = 0
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    with torch.no_grad():  
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _,prediction = torch.max(output.data, 1)
            correct_pred += (prediction == labels).sum().item()
            total_labels += labels.size(0)
        test_accuracy = (correct_pred/total_labels)*100
        print(f"Test Accuracy: {test_accuracy:.2f}%..")
            
# checking accuracy on test data
test_model(testloader)
                               
#saving a checkpoint   
model.class_to_idx = training_data.class_to_idx
checkpoint1 = {'model': model_arch,
              'learning_rate': learning_rate,
              'dropout': dropout,
              'output_size': 102,
              'hidden_layers': hidden_layers,
              'state_dict': model.state_dict(),
              'epochs': epochs,
              'class_to_idx': model.class_to_idx}
torch.save(checkpoint1, checkpoint)
