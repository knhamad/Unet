from Model import UNetmodel
from torch.utils import data
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
cuda =  torch.cuda.is_available() and True


def test_visiualization(Unet=UNetmodel.UNet,loader_test=data.DataLoader,criterion=nn):
    
    Unet.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (img, label) in enumerate(loader_test):

            if cuda:
                img = img.cuda()
                label = label.cuda()
            
            output = Unet(img) 
            print(img.shape)
            #print(output.shape)
            # plt.figure(1)
            # plt.imshow(img[1,2].cpu())
            label=label/255
            #print(label.shape)
            
            label=label.reshape((output.shape[0], 1, 608, 968))
            #print(label.shape)
            #print(output.shape)
            # plt.figure(2)
            # plt.imshow(label[1,0].cpu())
            label=label.float()
            test_loss += criterion(output, label).item()
            # plt.figure(3)
            # plt.imshow(output[1,0].cpu())
            output=(output-output.min())/(output.max()-output.min())
            output[output<0.4]=0
            output[output>=0.4]=1
            plt.figure(i)
            plt.imshow(output[0,0].cpu())
            #print(torch.unique(output))
            # plt.figure(i+2)
            # plt.imshow(output[0,0].cpu())
            #print(test_loss)
            #print(output[1,0])
        
    test_loss /= len(loader_test.dataset)
    print (" loss=%.8f" %(test_loss))