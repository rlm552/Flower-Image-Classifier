import argparse
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
import time

def arg_parser():
    cwd = os.getcwd()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dir', help='The directory containing images to train network')
    parser.add_argument('--save_dir', default=cwd, help='Directory for saving model to checkpoint. Default is current working directory')
    parser.add_argument('--arch', type=str, default='vgg16', help='Architecture used for training. Support architectures: alexnet, vgg11, vgg13, vgg16, vgg19')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate used in training. Defualt is 0.01')
    parser.add_argument('--hidden_units', default=512, help='Number of hidden units used in hidden layer. Default is 512')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs used in training. Default is 4')
    parser.add_argument('--gpu', action='store_const', dest='device', default='cpu', const='cuda:0', help='Device on which to train. Default is cpu.')
    
    args = parser.parse_args()
    
    return args

args = arg_parser()

device = torch.device(args.device)

def trainloader(crop_resize, mean, std, batch_size):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(crop_resize),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=mean,
                                                            std =std)])
    train_data = datasets.ImageFolder(args.train_dir, transform=train_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return trainloader, train_data

trainloader, train_data = trainloader(crop_resize = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], batch_size = 64)

def Network():
    method_to_call = getattr(models, args.arch)
    model = method_to_call(pretrained=True)
    
    if args.arch in 'alexnet':
        in_features = model.classifier[1].in_features
    elif args.arch in ['vgg11', 'vgg13', 'vgg16', 'vgg19']:
        in_features = model.classifier[0].in_features
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                                ('fc1', nn.Linear(in_features, args.hidden_units)),
                                ('relu', nn.ReLU()),
                                ('Dropout', nn.Dropout(p=0.5)),
                                ('fc2', nn.Linear(args.hidden_units,102)),
                                ('output', nn.LogSoftmax(dim=1))
                            ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    return model, criterion, optimizer, in_features

model, criterion, optimizer, in_features = Network()

def validloader(crop_resize, mean, std, batch_size):
    valid_dir = 'flowers/valid'
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(crop_resize),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=mean,
                                                            std =std)])
    
    valid_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    return validloader

validloader = validloader(crop_resize = 224, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], batch_size = 32)

def validation(model, validloader, criterion, inputs, labels):
    correct = 0
    total = 0
    valid_loss = 0

    with torch.no_grad():
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        total += labels.size(0)
        correct += equality.type(torch.FloatTensor).sum()   

    return correct, total

def train(epochs):
    
    print_every = 40
    steps = 0

    model.to(device)
    model.train()

    for e in range(epochs):    
        start = time.time()
        running_loss = 0
        batches = 0
        correct = 0
        total = 0
        # Loop through batches
        for i, (inputs, labels) in enumerate(trainloader):        
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)      
            
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            ps = torch.exp(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            #Find accuracy for training data
            equality = (labels.data == ps.max(dim=1)[1])
            total += labels.size(0)
            correct += equality.type(torch.FloatTensor).sum()
            batches+=1
            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))            
                running_loss = 0
        print(f"Time per batch: {(time.time() - start)/batches:.3f} seconds")
        accuracy = 100 * correct/total
        print('Training Accuracy: %d %%' % (accuracy), 'Epoch: ', e+1)
        # Model in inference mode for validation, dropout is off
        model.eval()
        correct = 0
        total = 0
        for ii, (inputs, labels) in enumerate(validloader):
            # TODO:   Try your model on validation data and print relevant results
            batch_correct, batch_total = validation(model, validloader, criterion, inputs, labels)
            correct+=batch_correct
            total+=batch_total
        accuracy = 100 * correct/total
        print('Accuracy of the network on the',total,'validation images: %d %%' % (accuracy))
        # Make sure dropout is on for training
        model.train()
    return model

model = train(args.epochs)

def save_checkpoint():
    checkpoint = {'model': args.arch,
                  'input_size': in_features,
                  'output_size': 102,
                  'hidden_layer': args.hidden_units,
                  'dropout' : 0.5,
                  'epochs': args.epochs,
                  'idx_to_class': {v: k for k, v in train_data.class_to_idx.items()},
                  'optim_state': optimizer.state_dict(),
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, args.save_dir + '/' + 'checkpoint.pth')

save_checkpoint()