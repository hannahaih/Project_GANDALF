#################################################################################################################################
#################################################################################################################################
#                                                        "GAN-DALF"                                                             #
#                                                  Author: Hannah Reber                                                         #
#                                       https://github.com/hannahaih/Project_GANDALF                                            #
#################################################################################################################################
#################################################################################################################################

import os, cv2, glob, sys, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, imsave

#################################################################################################################################
#################################################################################################################################
#                                                       GANDALF TOOLS                                                           #
#################################################################################################################################
#################################################################################################################################

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_DIM = (128, 128, 3)

def crop_and_resize(img, w, h):
    im_h, im_w, channels = img.shape
    res_aspect_ratio = w/h
    input_aspect_ratio = im_w/im_h
    if input_aspect_ratio > res_aspect_ratio:
        im_w_r = int(input_aspect_ratio*h)
        im_h_r = h
        img = cv2.resize(img, (im_w_r , im_h_r))
        x1 = int((im_w_r - w)/2)
        x2 = x1 + w
        img = img[:, x1:x2, :]
    if input_aspect_ratio < res_aspect_ratio:
        im_w_r = w
        im_h_r = int(w/input_aspect_ratio)
        img = cv2.resize(img, (im_w_r , im_h_r))
        y1 = int((im_h_r - h)/2)
        y2 = y1 + h
        img = img[y1:y2, :, :]
    if input_aspect_ratio == res_aspect_ratio:
        img = cv2.resize(img, (w, h))
    return img


def generate_image(G, n_samples, n_noise):
    z = torch.randn(n_samples, n_noise).to(DEVICE)
    y_hat = G(z).view(n_samples, IMAGE_DIM[2], IMAGE_DIM[0], IMAGE_DIM[1]).permute(0, 2, 3, 1)
    result = (y_hat.detach().cpu().numpy()+1)/2.
    return result


def makesavepath(SAVE_DIR):
    pathid = SAVE_DIR.split("\\")[-2] + "\\"
    pathway = SAVE_DIR.replace(pathid,"")
    if pathid.replace("\\","") not in os.listdir(pathway):
        os.makedirs(SAVE_DIR)
        print("made folder",SAVE_DIR)
    else:
        print("found folder")
              
def preprocess_step1(files, RESIZED_IMAGES):   
    for po in files:
        jpg_name = RESIZED_IMAGES + po.split("\\")[-1]
        img = cv2.imread(po)
        img = crop_and_resize(img,IMAGE_DIM[0],IMAGE_DIM[1])
        cv2.imwrite(jpg_name,img)
        
def preprocess_step2(dst, src):
    if not os.path.exists(dst):
        os.mkdir(dst)
    for each in os.listdir(src):
        png = Image.open(os.path.join(src,each))
        # print each
        if png.mode == 'RGBA':
            png.load() # required for png.split()
            background = Image.new("RGB", png.size, (0,0,0))
            background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
            background.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')
        else:
            png.convert('RGB')
            png.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')           
            
class Discriminator(nn.Module):
    def __init__(self, in_channel=1, num_classes=1):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
        )        
        self.fc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x, y=None):
        y_ = self.conv(x)
        y_ = y_.view(y_.size(0), -1)
        y_ = self.fc(y_)
        return y_
    
class Generator(nn.Module):
    def __init__(self, out_channel=1, input_size=100, num_classes=784):
        super(Generator, self).__init__()
        assert IMAGE_DIM[0] % 2**4 == 0, 'Should be divided 16'
        self.init_dim = (IMAGE_DIM[0] // 2**4, IMAGE_DIM[1] // 2**4)
        self.fc = nn.Sequential(
            nn.Linear(input_size, self.init_dim[0]*self.init_dim[1]*512),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, out_channel, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        
    def forward(self, x, y=None):
        x = x.view(x.size(0), -1)
        y_ = self.fc(x)
        y_ = y_.view(y_.size(0), 512, self.init_dim[0], self.init_dim[1])
        y_ = self.conv(y_)
        return y_
    
    
class IMAGES(Dataset):
    
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.fpaths = sorted(glob.glob(os.path.join(data_path, '*.jpg'))) 
        
    def __getitem__(self, idx):
        img = self.transform(Image.open(self.fpaths[idx]))
        return img
    
    def __len__(self):
        return len(self.fpaths)

    
def make_mp4(piclist, path, fps=24,title="video"):
    imagelist, frames = piclist, piclist
    #imagelist = [x for x in os.listdir(path) if ".jpg" in x]
    #frames = [x for x in os.listdir(path) if ".jpg" in x]
    fone = frames[0]
    frameone = cv2.imread(path+fone)
    height, width, channels = frameone.shape
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    video_name = path + title + ".mp4"
    height, width, layers = frameone.shape
    print("writing video",title,"...")
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height)) 
    for image in imagelist: 
        video.write(cv2.imread(os.path.join(path, image))) 
    cv2.destroyAllWindows() 
    video.release() 
    print("done with",video_name)
    

def cnn_styletransfer(contentpath,stylepath,num_epochs,cnn_output_path,sessionnum,example_id=None):
        
    vgg = models.vgg19(pretrained=True).features ###### get features-portion of VGG19 (needed for classifier portion)
    for param in vgg.parameters(): ###################### freeze all VGG parameters, only optimizing the target image
        param.requires_grad_(False)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    vgg.to(device)
    if torch.cuda.is_available()==False:
        print("ATTENTION: something wrong with CUDA")
        
    output_pics = []
    jpg_id = contentpath.split("\\")[-1]
    content = load_image(contentpath).to(device) ########################################### load content + style pic
    style = load_image(stylepath, shape=content.shape[-2:]).to(device) ################ resize style to match content
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3)) ######################################### display both pics
    ax1.imshow(im_convert(content))                                    
    ax1.set_title("Content Image "+example_id,fontsize = 10)
    ax2.imshow(im_convert(style))
    ax2.set_title("Style Image "+example_id, fontsize = 10)
    plt.show()
    ###################################################################################### def features, grams, target
    content_features = get_features(content, vgg) ########### get content and style features only once before training
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features} # gram matrices style layers
    target = content.clone().requires_grad_(True).to(device) ##################### start with target = copy of content
    style_weights = {'conv1_1': 1., ############################## initialize weights for all layers EXCLUDING conv4_2
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}
    content_weight = 1 ######################################################################################## alpha
    style_weight = 1e9 ######################################################################################### beta
    optimizer = optim.Adam([target], lr=0.003) ############################################################ optimizer
    ####################################################################################### transfer style to content    
    for epoch in range(1, num_epochs+1):
        target_features = get_features(target, vgg) ######################################## get features from target
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2) ####### content-loss
        style_loss = 0 ################################################################ initialize style-loss to zero
        for layer in style_weights: ################################################## add gram matrix loss of layers
            target_feature = target_features[layer] ############################## get style representation for layer
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer] ################################################ get style representation
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2) ## weighted styleloss
            style_loss += layer_style_loss / (d * h * w) ########################################### added style-loss
        total_loss = content_weight * content_loss + style_weight * style_loss ########################### total loss
        optimizer.zero_grad() ##################################################################### target pic update
        total_loss.backward()
        optimizer.step()
        ############################################################################################## get result pic
        if  epoch == num_epochs:                                                
            picname = cnn_output_path + jpg_id
            plt.imshow(im_convert(target))
            plt.axis("off")
            plt.savefig(picname,dpi=600,bbox_inches="tight", pad_inches=0,)
            print("done with",picname)
            
#################################################################################################################################
#################################################################################################################################
#                                                 CNN STYLE TRANSFER TOOLS                                                      #
#################################################################################################################################
#################################################################################################################################

# func to plot loss
def plotloss(list_of_losses):
    plt.figure(figsize=(6,3))
    plt.plot(list_of_losses)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    picname = "loss.jpg"
    plt.savefig(picname,dpi=300)

# func to load & preprocess pics
def load_image(img_path, max_size=400, shape=None):    
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape      
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])
            # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)  
    return image

# func to un-normalize pics + conv fromm tensor to np.array
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

# func to get features
def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  # content representation
                  '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():   # model._modules = dict holding each module in the model
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# func to get gram matrix
def gram_matrix(tensor):
    _, d, h, w = tensor.size()           # get the batch_size, depth, height, and width of the Tensor
    tensor = tensor.view(d, h * w)       # reshape so we're multiplying the features for each channel
    gram = torch.mm(tensor, tensor.t())  # calculate the gram matrix  
    return gram


# func for choosing stylepath
def choose_match(contenpath, stylepath):
    content = load_image(contenpath).to(device)                         # load content + style pic
    style = load_image(stylepath, shape=content.shape[-2:]).to(device)  # resize style to match content
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))               # display both pics
    ax1.imshow(im_convert(content))                                    
    ax1.set_title("Content Image ",fontsize = 10)
    ax2.imshow(im_convert(style))
    ax2.set_title("Style Image ",fontsize = 10)
    plt.show()

sessionnum = 0

# func to apply styletransfer
def initialize_styletransfer(contentpath,stylepath,num_epochs,cnn_output_path,sessionnum):
    losses = []
    output_pics = []
    jpg_id = contentpath.split("\\")[-1]
    content = load_image(contentpath).to(device)                             # load content + style pic
    style = load_image(stylepath, shape=content.shape[-2:]).to(device)       # resize style to match content
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))                    # display both pics
    ax1.imshow(im_convert(content))                                    
    ax1.set_title("Content Image "+example_id,fontsize = 20)
    ax2.imshow(im_convert(style))
    ax2.set_title("Style Image "+example_id, fontsize = 20)
    plt.show()
    # def features, grams, target
    content_features = get_features(content, vgg)                 # get content and style features only once before training
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}   # gram matrices for style layers
    target = content.clone().requires_grad_(True).to(device)                           # start with target = copy of content
    style_weights = {'conv1_1': 1.,                                    # initialize weights for all layers EXCLUDING conv4_2
                     'conv2_1': 0.75,
                     'conv3_1': 0.2,
                     'conv4_1': 0.2,
                     'conv5_1': 0.2}
    content_weight = 1                          # alpha
    style_weight = 1e9                          # beta
    optimizer = optim.Adam([target], lr=0.003)  # optimizer
    if sessionnum > 0: 
        print("loading model")
        checkpoint = torch.load(session_modelpath)
        vgg.load_state_dict(checkpoint['vgg_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("session number: ",sessionnum)
        vgg.train()
        
    for epoch in range(1, num_epochs+1):
        target_features = get_features(target, vgg)                            # get the features from your target image
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)          # content-loss
        style_loss = 0                                                                  # initialize the style-loss to 0
        for layer in style_weights:                                   # then add to it for each layer's gram matrix loss
            target_feature = target_features[layer]                # get the "target" style representation for the layer
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = style_grams[layer]                                       # get the "style" style representation
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)     # styleloss weighted
            style_loss += layer_style_loss / (d * h * w)                                              # added style-loss
        total_loss = content_weight * content_loss + style_weight * style_loss                    # calculate total-loss
        losses.append(total_loss)                                                                          # save Losses
        optimizer.zero_grad()                                                                      # update target image
        total_loss.backward()
        optimizer.step()
        
        if  epoch == num_epochs:                                                
            picname = cnn_output_path + jpg_id
            plt.imshow(im_convert(target))
            plt.axis("off")
            plt.savefig(picname,dpi=600,bbox_inches="tight", pad_inches=0,)
            