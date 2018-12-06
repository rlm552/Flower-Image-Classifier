import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import models
from torch import nn
from collections import OrderedDict
import json

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', type=str, help='The filepath to the image used for inference')
    parser.add_argument('checkpoint', help='The filepath to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Using this option will print the top k classes for the image. K must be less than or equal to 102. Defaults to 5.')
    parser.add_argument('-category_names', default=None, help='A JSON file containing a dictionary with the indices as key and the class(flower) as value')
    parser.add_argument('--gpu', action='store_const', dest='device', default='cpu', const='cuda:0', help='Device on which to train. Default is cpu.')
    
    args = parser.parse_args()
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    cat_to_name_f = args.category_names
    device = args.device
    
    return image_path, checkpoint, top_k, cat_to_name_f, device

image_path, checkpoint, top_k, cat_to_name_f, device = arg_parser()

cat_to_name = None
if(cat_to_name_f is not None):
    with open(cat_to_name_f, 'r') as f:
        cat_to_name = json.load(f)
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    # Define new model using input size, output size, and hidden layers in checkpoint
    ''''''
    method_to_call = getattr(models, checkpoint['model'])
    model = method_to_call(pretrained=True)
    
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer'])),
                            ('relu', nn.ReLU()),
                            ('Dropout', nn.Dropout(p=checkpoint['dropout'])),
                            ('fc2', nn.Linear(checkpoint['hidden_layer'],checkpoint['output_size'])),
                            ('output', nn.LogSoftmax(dim=1))
                        ]))

    model.classifier = classifier
    
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load model state variables
    model.epochs = checkpoint['epochs']
    model.optim_state = checkpoint['optim_state']
    model.idx_to_class = checkpoint['idx_to_class']
    
    return model

model = load_checkpoint(checkpoint)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    im = Image.open(image)
    im_copy = im.copy()    
    im.close()
    
    width = im_copy.size[0]
    height = im_copy.size[1]
    
    if width > height:
        size = int(width/height*256), 256
    else:
        size = 256, int(height/width*256)
        
    im_copy.thumbnail(size)
    
    width = im_copy.size[0]
    height = im_copy.size[1]
    
    left_crop = int((width - 224)/2)
    left_crop_remainder = (width - 224)/2 - left_crop
    right_crop = int(width - left_crop - 2*left_crop_remainder)
    bottom_crop = int((height - 224)/2)
    bottom_crop_remainder = (height - 224)/2 - bottom_crop
    top_crop = int(height - bottom_crop - 2* bottom_crop_remainder)
    crop_size = left_crop, bottom_crop, right_crop, top_crop      
    im_copy = im_copy.crop(crop_size)
    
    np_image = np.array(im_copy)/255
    im_copy.close()
    
    means = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means)/std    
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image    

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # DONE: Implement the code to predict the class from an image file
    # Pre-process image to numpy array
    np_image = process_image(image_path)
    torch_image = torch.from_numpy(np_image).float()

    torch_image.unsqueeze_(0)
        
    # Run in eval mode to turn off dropout
    model.eval()
    with torch.no_grad():    
        outputs = model.forward(torch_image)
    ps = torch.exp(outputs).data
    largest = ps.topk(topk)
    prob = largest[0].numpy()[0]
    idx = largest[1].numpy()[0]
    classes = [model.idx_to_class[x] for x in idx]
    
    return prob, classes

prob, classes = predict(image_path, model, topk=top_k)

def print_info(probs, classes):
        
    if (cat_to_name is not None):
        cat_names = []
        for ii in range(top_k):
            cat_names.append(cat_to_name[classes[ii]])
        classes = cat_names
        
    print('The top', top_k, 'predicted classes for the flower are', classes, 'with class probablities of', 
          prob)
    
print_info(prob, classes)