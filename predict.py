import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torch import optim
import json
import argparse
from PIL import Image
from torchvision import dataset, transforms, models
from torch import __version__
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--testfile', type=str, default = 'flowers/test/11/image_03130.jpg')
parser.add_argument('--jsonfile', type=str, default = 'cat_to_name.json')
parser.add_argument('--checkpointfile', type=str, default="checkpoint.pth")
parser.add_argument('--topk', type=int, default=5)

#variables
args = parser.parse_args()
test_file = args.testfile
json_file = args.jsonfile
checkpointf = args.checkpointfile
topk=args.topk

with open('cat_to_name.json') as label_file:
    cat_to_name = json.load(label_file, strict=False)

def load_checkpoint(file):
    checkpoint = torch.load(file)
    model_arch = checkpoint['model']
    learning_rate = checkpoint['learning_rate']
    dropout = checkpoint['dropout']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']
    state_dict = checkpoint['state_dict']
    epochs = checkpoint['epochs']
    class_to_idx = checkpoint['class_to_idx']
    
    if model == 'vgg19':
        model = models.vgg19(pretrained = True)
    elif model == 'resnet18':
        model = models.resnet18(pretrained = True)
    elif model == 'alexnet':
        model = models.alexnet(pretrained=True)

    model.classifier = classifier
    model.class_to_idx = class_to_idx
    model.load_state_dict(state_dict)

    for param in model.parameters():
        param.requires_grad = False
    return model

#loading model
model_load = load_checkpoint()

#process images
def process_image(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # TODO: Process a PIL image for use in a PyTorch model
    preprocess = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    pil_image = Image.open(image)
    preprocessed_img = preprocess(np_image).float()
    return preprocessed_img

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    image = process_image(image_path)
    img = image.unsqueeze_(0).float()
    top_class=[]
    with torch.no_grad():
        image = img.to(device)
        output = model.forward(image)
    predict = torch.exp(output)
    
    probs, idxs = predict.topk(topk, dim=1)
    top_probs = probs.cpu().numpy()[0]
    idxs = idxs.cpu().numpy()[0]
    idxs_to_class = {i:c for c,i in model.class_to_idx.items()}
    top_class = [idxs_to_class[x] for x in idxs]
    
    return top_probs, top_class
#prediction
probs, classes = predict(test, model_load, topk)
print(probs)
print(classes)
